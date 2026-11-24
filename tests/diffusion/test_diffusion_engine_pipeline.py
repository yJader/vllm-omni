# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
from collections import deque

import pytest

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode, _RpcTask
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.executor.abstract import PipelineStepFutures
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _PipelineExecutor:
    def __init__(self) -> None:
        self.batches: list[tuple[object, concurrent.futures.Future, concurrent.futures.Future]] = []
        self.rpc_calls: list[str] = []

    def submit_step(self, sched_output) -> PipelineStepFutures:
        pp0_done: concurrent.futures.Future[None] = concurrent.futures.Future()
        step_done: concurrent.futures.Future = concurrent.futures.Future()
        self.batches.append((sched_output, pp0_done, step_done))
        return PipelineStepFutures(pp0_done=pp0_done, step_done=step_done)

    def collective_rpc(self, method: str, **kwargs):
        self.rpc_calls.append(method)
        return [None]

    def shutdown(self) -> None:
        return None


def _make_engine(loop: asyncio.AbstractEventLoop, *, max_concurrent_batches: int = 2) -> DiffusionEngine:
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = OmniDiffusionConfig(
        model="test",
        max_num_seqs=1,
        max_concurrent_batches=max_concurrent_batches,
        streaming_output=False,
        parallel_config={"pipeline_parallel_size": 2},
        diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
    )
    engine.executor = _PipelineExecutor()
    engine.scheduler = StepScheduler()
    engine.scheduler.initialize(engine.od_config)
    engine.scheduler.max_num_resident_reqs = max_concurrent_batches
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    engine._pipeline_async_enabled = True
    engine._max_concurrent_batches = max_concurrent_batches
    engine._pending_batches = deque()
    engine._pp0_ready = True
    engine._control_gate_closed = False
    engine._pipeline_failure = None
    engine._next_pipeline_batch_id = 0
    engine._deferred_pipeline_aborts = set()
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine._out_streams = {}
    engine._request_chunk_states = {}
    engine._unclaimed_async_outputs = {}
    engine._shutdown_output_futures = {}
    engine._closed = False
    engine._shutting_down = False
    engine._shutdown_complete = False
    engine.abort_queue = queue.Queue()
    engine._rpc_queue = queue.Queue()
    engine._scheduling_paused = False
    engine.dp_concurrent = False
    engine._streaming_playback = {}
    engine._scheduler_num_waiting_reqs = 0
    engine._request_cancellations = None
    engine.main_loop = loop
    engine.stop_event = threading.Event()
    engine.worker_thread = threading.Thread(target=engine._busy_loop, daemon=True)
    engine.worker_thread.start()
    return engine


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, 2), (1, 1), (3, 3)],
)
def test_diffusion_config_resolves_max_concurrent_batches(configured, expected):
    config = OmniDiffusionConfig(
        model="test",
        parallel_config={"pipeline_parallel_size": 2},
        max_concurrent_batches=configured,
    )

    assert config.max_concurrent_batches == expected


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_diffusion_config_rejects_invalid_max_concurrent_batches(invalid):
    with pytest.raises(ValueError, match="max_concurrent_batches"):
        OmniDiffusionConfig(model="test", max_concurrent_batches=invalid)


def _add_request(engine: DiffusionEngine, request_id: str) -> asyncio.Queue:
    request = OmniDiffusionRequest(
        request_id=request_id,
        prompt="test",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )
    engine.add_request(request)
    return engine._out_streams[request_id]


def _step_output(request_id: str) -> BatchRunnerOutput:
    return BatchRunnerOutput.from_list(
        [
            RunnerOutput(
                request_id=request_id,
                step_index=1,
                finished=True,
                result=DiffusionOutput(finished=True),
            )
        ]
    )


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("Timed out waiting for pipeline Engine state")


def _stop_engine(engine: DiffusionEngine) -> None:
    with engine._cv:
        engine.stop_event.set()
        engine._cv.notify_all()
    engine.worker_thread.join(timeout=3)
    assert not engine.worker_thread.is_alive()
    engine.executor.shutdown()
    engine.scheduler.close()


@pytest.mark.asyncio
async def test_pp0_readiness_submits_next_request_before_fifo_result_retirement():
    engine = _make_engine(asyncio.get_running_loop())
    output_a = _add_request(engine, "a")
    output_b = _add_request(engine, "b")
    try:
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        _, pp0_a, step_a = engine.executor.batches[0]
        pp0_a.set_result(None)
        await _wait_until(lambda: len(engine.executor.batches) == 2)

        sched_b, pp0_b, step_b = engine.executor.batches[1]
        assert sched_b.scheduled_request_ids == ["b"]
        pp0_b.set_result(None)
        step_b.set_result(_step_output("b"))
        await asyncio.sleep(0.03)
        assert output_b.empty()

        step_a.set_result(_step_output("a"))
        await asyncio.sleep(0.03)
        assert not output_a.empty() and not output_b.empty()
        result_a, result_b = output_a.get_nowait(), output_b.get_nowait()
        assert result_a.finished and result_b.finished
        assert engine.scheduler.has_requests() is False
    finally:
        _stop_engine(engine)


@pytest.mark.asyncio
async def test_c8_dispatches_waiting_request_until_continuation_latent_is_ready():
    engine = _make_engine(asyncio.get_running_loop(), max_concurrent_batches=8)
    batches = engine.executor.batches
    try:
        for request_id, size, steps in (("q0", 1024, 2), ("q1", 512, 1), ("q2", 768, 1)):
            engine.add_request(
                OmniDiffusionRequest(
                    request_id=request_id,
                    prompt="test",
                    sampling_params=OmniDiffusionSamplingParams(height=size, width=size, num_inference_steps=steps),
                )
            )
        await _wait_until(lambda: len(batches) == 1)
        assert batches[0][0].scheduled_request_ids == ["q0"]
        batches[0][1].set_result(None)
        await _wait_until(lambda: len(batches) == 2)
        assert batches[1][0].scheduled_request_ids == ["q1"]
        batches[1][1].set_result(None)
        await _wait_until(lambda: len(batches) == 3)
        assert batches[2][0].scheduled_request_ids == ["q2"]

        batches[0][2].set_result(BatchRunnerOutput.from_list([RunnerOutput("q0", step_index=1, finished=False)]))
        batches[2][1].set_result(None)
        await _wait_until(lambda: len(batches) == 4)
        assert batches[3][0].scheduled_cached_reqs.request_ids == ["q0"]
    finally:
        for batch, entry, result in list(batches):
            if not entry.done():
                entry.set_result(None)
            if not result.done():
                result.set_result(_step_output(batch.scheduled_request_ids[0]))
        _stop_engine(engine)


@pytest.mark.asyncio
async def test_worker_control_rpc_drains_submitted_pipeline_batches():
    engine = _make_engine(asyncio.get_running_loop())
    output = _add_request(engine, "a")
    try:
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        _, pp0_done, step_done = engine.executor.batches[0]
        pp0_done.set_result(None)
        task = _RpcTask(method="update_weights", args=(), kwargs=None, deadline=None, unique_reply_rank=None)
        with engine._cv:
            engine._rpc_queue.put(task)
            engine._cv.notify_all()

        await asyncio.sleep(0.03)
        assert not task.future.done()
        assert engine.executor.rpc_calls == []

        step_done.set_result(_step_output("a"))
        await asyncio.sleep(0.03)
        assert task.future.result(timeout=0) == [None]
        assert engine.executor.rpc_calls == ["update_weights"]
        assert output.get_nowait().finished
    finally:
        _stop_engine(engine)


@pytest.mark.asyncio
async def test_pause_scheduler_drains_pipeline_batches_and_returns_none():
    engine = _make_engine(asyncio.get_running_loop())
    output = _add_request(engine, "a")
    try:
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        _, pp0_done, step_done = engine.executor.batches[0]
        pp0_done.set_result(None)
        task = _RpcTask(
            method="pause_scheduler",
            args=(),
            kwargs={"mode": "keep"},
            deadline=None,
            unique_reply_rank=None,
        )
        with engine._cv:
            engine._scheduling_paused = True
            engine._rpc_queue.put(task)
            engine._cv.notify_all()

        await asyncio.sleep(0.03)
        assert not task.future.done()
        assert engine.executor.rpc_calls == []

        step_done.set_result(_step_output("a"))
        await asyncio.sleep(0.03)
        assert task.future.result(timeout=0) is None
        assert engine.executor.rpc_calls == ["synchronize_device"]
        assert output.get_nowait().finished
    finally:
        _stop_engine(engine)


@pytest.mark.asyncio
async def test_engine_shutdown_drains_submitted_pipeline_batches():
    engine = _make_engine(asyncio.get_running_loop())
    _add_request(engine, "a")
    try:
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        _, _, step_done = engine.executor.batches[0]
        with engine._cv:
            engine.stop_event.set()
            engine._cv.notify_all()
        await asyncio.sleep(0.03)
        assert engine.worker_thread.is_alive()

        step_done.set_result(_step_output("a"))
        await _wait_until(lambda: not engine.worker_thread.is_alive())
        assert not engine.worker_thread.is_alive()
        assert engine.scheduler.has_requests() is False
    finally:
        _stop_engine(engine)
