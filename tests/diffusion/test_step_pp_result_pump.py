# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
from types import SimpleNamespace

import pytest
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

from vllm_omni.diffusion.data import DiffusionParallelConfig, RpcReply
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("step_execution", "pp_size", "n_local_reader", "expected_busy_loop_s"),
    [
        pytest.param(True, 2, 1, 0.0, id="step-pp"),
        pytest.param(True, 1, 1, 1.0, id="step-pp1"),
        pytest.param(False, 2, 1, 1.0, id="request-mode"),
        pytest.param(True, 2, 0, None, id="remote-result-reader"),
    ],
)
def test_result_queue_busy_loop(step_execution, pp_size, n_local_reader, expected_busy_loop_s):
    writer = MessageQueue(
        n_reader=1, n_local_reader=n_local_reader, max_chunk_bytes=1024, max_chunks=2, connect_ip="127.0.0.1"
    )
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(
        step_execution=step_execution, parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size)
    )
    reader = executor._init_result_queue(writer.export_handle())
    condition = reader._spin_condition
    assert (condition.busy_loop_s if condition is not None else None) == expected_busy_loop_s


class _Queue:
    def __init__(self):
        self.queue: queue.Queue[object] = queue.Queue()

    def dequeue(self, timeout):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError from exc

    def export_handle(self):
        return None


@pytest.mark.parametrize("worker_error", [None, ""])
def test_step_pp_receives_both_roles_then_control_reply(monkeypatch, worker_error):
    writers = [MessageQueue(n_reader=1, n_local_reader=1, max_chunk_bytes=4096, max_chunks=4) for _ in range(4)]
    handles = [writer.export_handle() for writer in writers]
    expected = BatchRunnerOutput.from_list([])
    roles = {}

    def enqueue(message, **kwargs):
        if isinstance(message, dict) and "reply_roles" in message:
            roles.update(message["reply_roles"])
            writers[0].enqueue(RpcReply(rpc_id=roles["entry"], result=None))
        elif isinstance(message, dict) and message.get("method") == "synchronize_device":
            writers[0].enqueue(True)

    broadcast = SimpleNamespace(export_handle=lambda: None, enqueue=enqueue)
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_init_broadcast_queue", lambda *_: broadcast)
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_launch_workers", lambda *_: ([], handles))
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_start_worker_monitor", lambda _: None)
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(
        num_gpus=4,
        step_execution=True,
        parallel_config=SimpleNamespace(pipeline_parallel_size=2, tensor_parallel_size=2),
    )
    executor._init_executor()
    pump_threads = executor._result_pump_threads
    try:
        futures = executor.submit_step(SimpleNamespace(scheduled_request_ids=["q0"]))
        assert set(roles) == {"entry", "result"}
        assert futures.pp0_done.result(timeout=2) is None
        assert not futures.step_done.done()
        writers[2].enqueue(
            RpcReply(
                roles["result"],
                result=expected if worker_error is None else None,
                error=worker_error,
            )
        )
        if worker_error is None:
            assert futures.step_done.result(timeout=2) == expected
        else:
            with pytest.raises(RuntimeError):
                futures.step_done.result(timeout=2)
        assert executor.collective_rpc("synchronize_device", timeout=2) == [True]
    finally:
        executor.shutdown()
    assert all(not thread.is_alive() for thread in pump_threads)


def test_step_pp_kv_timeout_and_shutdown(monkeypatch):
    queues = [_Queue(), _Queue()]
    roles_list: list[dict[str, str]] = []

    def enqueue(message, **kwargs):
        if isinstance(message, dict) and "reply_roles" in message:
            roles_list.append(dict(message["reply_roles"]))

    broadcast = SimpleNamespace(export_handle=lambda: None, enqueue=enqueue)
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_init_broadcast_queue", lambda *_: broadcast)
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_launch_workers", lambda *_: ([], [0, 1]))
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_init_result_queue", lambda _, index: queues[index])
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_start_worker_monitor", lambda _: None)
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(
        num_gpus=2,
        step_execution=True,
        parallel_config=DiffusionParallelConfig(pipeline_parallel_size=2),
    )
    executor._init_executor()
    try:
        monkeypatch.setattr("vllm_omni.diffusion.executor.multiproc_executor._DLO_DP_WAVE_TIMEOUT_S", 0.15)
        kv_step = SimpleNamespace(
            scheduled_request_ids=["q0"],
            has_nonblocking_kv_work=True,
            kv_finished_request_ids=set(),
        )
        futures = executor.submit_step(kv_step)
        roles = roles_list[-1]
        assert set(roles) == {"entry", "result"}
        queues[0].queue.put(RpcReply(roles["result"], BatchRunnerOutput.from_list([]), kv_rank=0))
        with pytest.raises(TimeoutError, match="Missing diffusion KV step ACK"):
            futures.step_done.result(timeout=2)
        assert roles["result"] not in executor._kv_step_reports
        assert roles["result"] not in executor._rpc_futures

        # Shutdown while step_done is pending must fail both futures cleanly without deadlock.
        monkeypatch.setattr("vllm_omni.diffusion.executor.multiproc_executor._DLO_DP_WAVE_TIMEOUT_S", 10.0)
        pending = executor.submit_step(SimpleNamespace(scheduled_request_ids=["q1"]))
        executor.shutdown()
        with pytest.raises(RuntimeError, match="Executor shut down"):
            pending.step_done.result(timeout=2)
        with pytest.raises(RuntimeError, match="Executor shut down"):
            pending.pp0_done.result(timeout=2)
    finally:
        if not executor._closed:
            executor.shutdown()
