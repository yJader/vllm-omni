# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import RpcReply
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


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
    queues = [_Queue(), _Queue()]
    expected = BatchRunnerOutput.from_list([])

    def enqueue(message, **kwargs):
        roles = message.get("reply_roles") if isinstance(message, dict) else None
        if roles:
            queues[0].queue.put(RpcReply(rpc_id=roles["entry"], result=None))
            queues[1].queue.put(
                RpcReply(rpc_id=roles["result"], result=expected if worker_error is None else None, error=worker_error)
            )
        elif isinstance(message, dict) and message.get("method") == "synchronize_device":
            queues[0].queue.put(True)

    broadcast = SimpleNamespace(export_handle=lambda: None, enqueue=enqueue)
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_init_broadcast_queue", lambda *_: broadcast)
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_launch_workers", lambda *_: ([], [0, 1]))
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_init_result_queue", lambda _, index: queues[index])
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_start_worker_monitor", lambda _: None)
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(
        num_gpus=2,
        step_execution=True,
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )
    executor._init_executor()
    try:
        futures = executor.submit_step(SimpleNamespace())
        assert futures.pp0_done.result(timeout=2) is None
        if worker_error is None:
            assert futures.step_done.result(timeout=2) is expected
        else:
            with pytest.raises(RuntimeError):
                futures.step_done.result(timeout=2)
        assert executor.collective_rpc("synchronize_device", timeout=2) == [True]
    finally:
        executor._pump_stop.set()
        for thread in getattr(executor, "_result_pump_threads", []):
            thread.join(timeout=2)
        executor._finalizer.detach()
