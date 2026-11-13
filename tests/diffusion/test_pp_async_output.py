# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PP workers must accept the next RPC while a terminal output is packed."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm_omni.diffusion.distributed.parallel_state as parallel_state
import vllm_omni.diffusion.ipc as diffusion_ipc
import vllm_omni.diffusion.worker.diffusion_worker as worker_module
from vllm_omni.diffusion.data import DiffusionOutput, RpcReply
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm, unpack_diffusion_output_shm
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _worker(monkeypatch, outputs):
    proc = WorkerProc.__new__(WorkerProc)
    proc.gpu_id = 1
    proc.od_config = SimpleNamespace(
        streaming_output=False,
        video_output_transport=SimpleNamespace(enable_registered_shm=False),
    )
    proc.result_mq = queue.Queue()
    proc._enqueue_result = proc.result_mq.put
    proc._async_output_queue = queue.Queue()
    proc._async_output_done = threading.Condition()
    proc._async_output_pending = 0
    proc.worker = SimpleNamespace(execute_method=lambda *args, **kwargs: next(outputs))
    monkeypatch.setattr(proc, "_is_stage_leader", lambda: True)
    monkeypatch.setattr(parallel_state, "get_pipeline_parallel_rank", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_pipeline_parallel_world_size", lambda: 2)
    fake_torch = MagicMock()
    fake_torch.accelerator.current_accelerator.return_value.type = "cpu"
    monkeypatch.setattr(worker_module, "torch", fake_torch)
    monkeypatch.setattr(worker_module.current_omni_platform, "record_device_event", lambda: object())
    return proc, fake_torch.Stream.return_value


def _rpc(proc, rpc_id):
    proc._execute_role_rpc(
        {"method": "execute_stepwise", "exec_all_ranks": True, "reply_roles": {"entry": "e", "result": rpc_id}}
    )


def _output(request_id, *, finished=True, dtype=torch.float32):
    return BatchRunnerOutput.from_list(
        [
            RunnerOutput(
                request_id=request_id,
                step_index=4,
                finished=finished,
                result=DiffusionOutput(output=torch.arange(4096, dtype=dtype)) if finished else None,
            )
        ]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_terminal_pack_does_not_block_next_step_and_preserves_output(monkeypatch, dtype):
    monkeypatch.setattr(diffusion_ipc, "_SHM_TENSOR_THRESHOLD", 1)
    terminal = _output("q0", dtype=dtype)
    expected = terminal.runner_outputs[0].result.output.clone()
    next_step = _output("q1", finished=False)
    proc, stream = _worker(monkeypatch, iter([terminal, next_step]))
    entered, release = threading.Event(), threading.Event()

    def pack(output, **kwargs):
        if output is terminal:
            entered.set()
            assert release.wait(2)
        return pack_diffusion_output_shm(output)

    monkeypatch.setattr(worker_module, "pack_diffusion_output_shm", pack)
    monkeypatch.setattr(diffusion_ipc, "pack_diffusion_output_shm", pack)
    producer = threading.Thread(target=_rpc, args=(proc, "r0"), daemon=True)
    producer.start()
    producer.join(0.5)
    try:
        assert not producer.is_alive(), "terminal D2H/SHM packing blocks the worker RPC loop"
        assert proc.result_mq.empty(), "step_done must wait until the output is ready"
        background = threading.Thread(target=proc._async_output_loop, daemon=True)
        background.start()
        assert entered.wait(1)
        assert proc.drain_async_outputs(timeout=0) is False
        _rpc(proc, "r1")
        reply = proc.result_mq.get(timeout=1)
        assert isinstance(reply, RpcReply) and reply.rpc_id == "r1"
        assert reply.result is next_step
        assert proc.result_mq.empty()
        release.set()
        reply = proc.result_mq.get(timeout=2)
        assert isinstance(reply, RpcReply) and reply.rpc_id == "r0" and reply.error is None
        assert reply.result.runner_outputs[0].result.output["__tensor_shm__"] is True
        unpack_diffusion_output_shm(reply.result)
        assert torch.equal(reply.result.runner_outputs[0].result.output, expected)
        assert reply.result.runner_outputs[0].result.output.dtype == dtype
        assert reply.result.runner_outputs[0].request_id == "q0"
        assert proc.drain_async_outputs(timeout=1)
        stream.wait_event.assert_called_once()
        stream.synchronize.assert_called_once()
    finally:
        release.set()
        producer.join(2)
        proc._async_output_queue.put(None)
        if "background" in locals():
            background.join(2)


def test_terminal_pack_failure_completes_original_rpc_with_error(monkeypatch):
    proc, _ = _worker(monkeypatch, iter([_output("q0")]))

    def fail(*args, **kwargs):
        raise RuntimeError("packing failed")

    monkeypatch.setattr(worker_module, "pack_diffusion_output_shm", fail)
    monkeypatch.setattr(diffusion_ipc, "pack_diffusion_output_shm", fail)
    _rpc(proc, "r0")
    proc._async_output_queue.put(None)
    proc._async_output_loop()
    reply = proc.result_mq.get_nowait()
    assert isinstance(reply, RpcReply) and reply.rpc_id == "r0"
    assert reply.error and reply.result is None
    assert proc.drain_async_outputs(timeout=0)
    assert proc.result_mq.empty()


@pytest.mark.parametrize("streaming,mixed", [(True, False), (False, True)])
def test_mutable_streaming_and_mixed_batches_keep_synchronous_transport(monkeypatch, streaming, mixed):
    result = _output("q0")
    if mixed:
        result.runner_outputs.extend(_output("q1", finished=False).runner_outputs)
    proc, _ = _worker(monkeypatch, iter([result]))
    proc.od_config.streaming_output = streaming
    monkeypatch.setattr(worker_module, "pack_diffusion_output_shm", lambda output: output)
    monkeypatch.setattr(diffusion_ipc, "pack_diffusion_output_shm", lambda output: output)
    _rpc(proc, "r0")
    assert proc.result_mq.get_nowait().result is result
    assert proc._async_output_queue.empty()


def test_side_stream_materializes_small_device_tensors_before_queue_serialization(monkeypatch):
    tensor = MagicMock(spec=torch.Tensor)
    tensor.device = torch.device("cuda:1")
    tensor.nelement.return_value = 1
    tensor.element_size.return_value = 4
    tensor.untyped_storage.return_value.nbytes.return_value = 4
    stream = object()
    handle = {"__tensor_shm__": True, "name": "test-only"}
    copy = MagicMock(return_value=handle)
    monkeypatch.setattr(diffusion_ipc, "_tensor_to_shm", copy)
    output = DiffusionOutput(output=tensor)
    pack_diffusion_output_shm(output, d2h_stream=stream)
    assert output.output is handle, "small device tensors must not fall through to synchronous pickle D2H"
    assert copy.call_args.kwargs["d2h_stream"] is stream


def test_empty_output_shm_round_trip_preserves_shape_and_dtype():
    # Side-stream transport also materializes empty device tensor fields.
    # POSIX SHM still requires a positive allocation size for their encoding.
    array = np.empty((0, 3), dtype=np.float32)
    handle = diffusion_ipc._array_to_shm(array)
    restored = diffusion_ipc._array_from_shm(handle)
    assert restored.shape == array.shape
    assert restored.dtype == array.dtype
