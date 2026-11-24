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


def test_last_stage_defers_non_terminal_result_until_device_event_without_blocking_next_rpc(monkeypatch):
    step0 = _output("q0", finished=False)
    step1 = _output("q1", finished=False)
    step2 = _output("q2", finished=False)
    proc, stream = _worker(monkeypatch, iter([step0, step1, step2]))
    monkeypatch.setattr(
        worker_module.current_omni_platform,
        "record_device_event",
        MagicMock(side_effect=AssertionError("non-terminal steps must not use blocking D2H record_device_event")),
    )
    pack = MagicMock(side_effect=AssertionError("non-terminal steps have no output tensors to pack"))
    monkeypatch.setattr(worker_module, "pack_diffusion_output_shm", pack)

    event0, event1 = MagicMock(), MagicMock()
    worker_module.torch.Event.side_effect = [event0, event1]
    entered, release = threading.Event(), threading.Event()

    def synchronize_event0():
        entered.set()
        assert release.wait(2)

    event0.synchronize.side_effect = synchronize_event0

    _rpc(proc, "r0")
    event0.record.assert_called_once()
    assert proc.result_mq.empty()

    background = threading.Thread(target=proc._async_output_loop, daemon=True)
    background.start()
    try:
        assert entered.wait(2)
        assert proc.drain_async_outputs(timeout=0) is False
        assert proc.result_mq.empty()

        # Worker RPC loop accepts the next step while r0 waits on its device event.
        _rpc(proc, "r1")
        event1.record.assert_called_once()
        assert proc.result_mq.empty()

        release.set()
        reply0 = proc.result_mq.get(timeout=2)
        reply1 = proc.result_mq.get(timeout=2)
        assert isinstance(reply0, RpcReply) and reply0.rpc_id == "r0" and reply0.result is step0
        assert isinstance(reply1, RpcReply) and reply1.rpc_id == "r1" and reply1.result is step1
        event1.synchronize.assert_called_once()
        stream.wait_event.assert_not_called()
        stream.synchronize.assert_not_called()
        assert proc.drain_async_outputs(timeout=1)

        # Without an accelerator, non-terminal replies fall back to immediate delivery.
        worker_module.torch.accelerator.is_available.return_value = False
        _rpc(proc, "r2")
        reply2 = proc.result_mq.get_nowait()
        assert isinstance(reply2, RpcReply) and reply2.rpc_id == "r2" and reply2.result is step2
        assert proc._async_output_queue.empty()
    finally:
        release.set()
        proc._async_output_queue.put(None)
        background.join(2)


def test_deferred_step_event_failure_completes_result_role_with_error(monkeypatch):
    proc, _ = _worker(monkeypatch, iter([_output("q0", finished=False)]))
    failing_event = MagicMock()
    failing_event.synchronize.side_effect = RuntimeError("device step failed")
    worker_module.torch.Event.return_value = failing_event

    _rpc(proc, "r0")
    assert proc.result_mq.empty()
    proc._async_output_queue.put(None)
    proc._async_output_loop()

    reply = proc.result_mq.get_nowait()
    assert isinstance(reply, RpcReply) and reply.rpc_id == "r0"
    assert reply.error and "device step failed" in reply.error
    assert proc.drain_async_outputs(timeout=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_terminal_pack_does_not_block_next_step_and_preserves_output(monkeypatch, dtype):
    monkeypatch.setattr(diffusion_ipc, "_SHM_TENSOR_THRESHOLD", 1)
    terminal = _output("q0", dtype=dtype)
    expected = terminal.runner_outputs[0].result.output.clone()
    next_step = _output("q1", finished=False)
    proc, stream = _worker(monkeypatch, iter([terminal, next_step]))
    next_event = MagicMock()
    worker_module.torch.Event.return_value = next_event
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
        next_producer = threading.Thread(target=_rpc, args=(proc, "r1"), daemon=True)
        next_producer.start()
        next_producer.join(0.5)
        assert not next_producer.is_alive(), "next non-terminal RPC must not block behind terminal packing"
        next_event.record.assert_called_once()
        assert proc.result_mq.empty()
        release.set()
        reply = proc.result_mq.get(timeout=2)
        assert isinstance(reply, RpcReply) and reply.rpc_id == "r0" and reply.error is None
        assert reply.result.runner_outputs[0].result.output["__tensor_shm__"] is True
        unpack_diffusion_output_shm(reply.result)
        assert torch.equal(reply.result.runner_outputs[0].result.output, expected)
        assert reply.result.runner_outputs[0].result.output.dtype == dtype
        assert reply.result.runner_outputs[0].request_id == "q0"
        next_reply = proc.result_mq.get(timeout=2)
        assert isinstance(next_reply, RpcReply) and next_reply.rpc_id == "r1" and next_reply.error is None
        assert next_reply.result is next_step
        assert proc.drain_async_outputs(timeout=1)
        stream.wait_event.assert_called_once()
        stream.synchronize.assert_called_once()
        next_event.synchronize.assert_called_once()
    finally:
        release.set()
        producer.join(2)
        if "next_producer" in locals():
            next_producer.join(2)
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
