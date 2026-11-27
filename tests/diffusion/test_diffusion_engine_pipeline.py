# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import queue
import threading
import time
from typing import Any

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
    supports_async_step_pp: bool = True

    def __init__(self) -> None:
        self.batches: list[tuple[Any, concurrent.futures.Future, concurrent.futures.Future]] = []
        self.rpc_calls: list[str] = []
        self.shutdown_calls: int = 0

    def submit_step(self, sched_output) -> PipelineStepFutures:
        pp0_done: concurrent.futures.Future[None] = concurrent.futures.Future()
        step_done: concurrent.futures.Future = concurrent.futures.Future()
        self.batches.append((sched_output, pp0_done, step_done))
        return PipelineStepFutures(pp0_done=pp0_done, step_done=step_done)

    def collective_rpc(self, method: str, **kwargs):
        self.rpc_calls.append(method)
        return [None]

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        return None


@contextlib.contextmanager
def _running_engine(*, max_concurrent_batches: int = 2, max_num_seqs: int = 1):
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = OmniDiffusionConfig(
        model="test",
        max_num_seqs=max_num_seqs,
        max_concurrent_batches=max_concurrent_batches,
        streaming_output=False,
        parallel_config={"pipeline_parallel_size": 2},
        diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
    )
    engine.executor = _PipelineExecutor()
    engine.scheduler = StepScheduler()
    engine.scheduler.initialize(engine.od_config)
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    engine._pipeline_async_enabled = True
    engine._init_runtime_state()
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
    engine._streaming_playback = {}
    engine._scheduler_num_waiting_reqs = 0
    engine._request_cancellations = None
    engine.main_loop = asyncio.get_running_loop()
    engine.stop_event = threading.Event()
    engine._loop_started = True
    engine.worker_thread = threading.Thread(target=engine._busy_loop, daemon=True)
    engine.worker_thread.start()
    try:
        yield engine
    finally:
        for batch, entry, result in list(engine.executor.batches):
            if not entry.done():
                entry.set_result(None)
            if not result.done():
                result.set_result(_step_output(batch.scheduled_request_ids[0]))
        with engine._cv:
            engine.stop_event.set()
            engine._cv.notify_all()
        engine.worker_thread.join(timeout=3)
        assert not engine.worker_thread.is_alive()
        engine.executor.shutdown()
        engine.scheduler.close()


@pytest.mark.parametrize(("configured", "expected"), [(None, 2), (1, 1), (3, 3)])
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


def _add_request(
    engine: DiffusionEngine,
    request_id: str,
    *,
    steps: int = 1,
    size: int | None = None,
    emit_lifecycle: bool = False,
) -> asyncio.Queue:
    params_kwargs: dict = {"num_inference_steps": steps, "emit_request_lifecycle": emit_lifecycle}
    if size is not None:
        params_kwargs.update(height=size, width=size)
    engine.add_request(
        OmniDiffusionRequest(
            request_id=request_id,
            prompt="test",
            sampling_params=OmniDiffusionSamplingParams(**params_kwargs),
        )
    )
    return engine._out_streams[request_id]


def _step_output(*request_ids: str, finished: bool = True) -> BatchRunnerOutput:
    return BatchRunnerOutput.from_list(
        [
            RunnerOutput(
                request_id=rid,
                step_index=1,
                finished=finished,
                result=DiffusionOutput(finished=True) if finished else None,
            )
            for rid in request_ids
        ]
    )


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("Timed out waiting for pipeline Engine state")


@pytest.mark.asyncio
async def test_pp0_readiness_submits_next_request_before_fifo_result_retirement():
    with _running_engine() as engine:
        output_a = _add_request(engine, "a")
        output_b = _add_request(engine, "b")
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
        assert output_a.get_nowait().finished and output_b.get_nowait().finished
        assert engine.scheduler.has_requests() is False


@pytest.mark.asyncio
async def test_c8_dispatches_waiting_request_until_continuation_latent_is_ready():
    with _running_engine(max_concurrent_batches=8) as engine:
        batches = engine.executor.batches
        for request_id, size, steps in (("q0", 1024, 2), ("q1", 512, 1), ("q2", 768, 1)):
            _add_request(engine, request_id, size=size, steps=steps)
        await _wait_until(lambda: len(batches) == 1)
        assert batches[0][0].scheduled_request_ids == ["q0"]
        batches[0][1].set_result(None)
        await _wait_until(lambda: len(batches) == 2)
        assert batches[1][0].scheduled_request_ids == ["q1"]
        batches[1][1].set_result(None)
        await _wait_until(lambda: len(batches) == 3)
        assert batches[2][0].scheduled_request_ids == ["q2"]

        batches[0][2].set_result(_step_output("q0", finished=False))
        batches[2][1].set_result(None)
        await _wait_until(lambda: len(batches) == 4)
        assert batches[3][0].scheduled_cached_reqs.request_ids == ["q0"]


@pytest.mark.asyncio
async def test_worker_control_rpc_drains_submitted_pipeline_batches():
    with _running_engine() as engine:
        output = _add_request(engine, "a")
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


@pytest.mark.asyncio
@pytest.mark.parametrize("fails_at_entry", [True, False])
async def test_pipeline_failure_during_control_drain_propagates_to_rpc_and_request(fails_at_entry):
    with _running_engine() as engine:
        output = _add_request(engine, "a")
        error = ValueError("pipeline worker failed")
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        _, pp0_done, step_done = engine.executor.batches[0]
        if not fails_at_entry:
            pp0_done.set_result(None)
        task = _RpcTask(method="update_weights", args=(), kwargs=None, deadline=None, unique_reply_rank=None)
        with engine._cv:
            engine._rpc_queue.put(task)
            engine._cv.notify_all()
        await _wait_until(lambda: engine._rpc_queue.empty())

        (pp0_done if fails_at_entry else step_done).set_exception(error)

        await _wait_until(lambda: not engine.worker_thread.is_alive())
        assert task.future.exception(timeout=0) is error
        assert engine.executor.rpc_calls == []
        assert engine._shutdown_complete
        assert (await asyncio.wait_for(output.get(), timeout=1)).error == str(error)


@pytest.mark.asyncio
async def test_late_rpc_enqueued_during_pipeline_failure_shutdown_is_failed_not_executed():
    with _running_engine() as engine:
        _add_request(engine, "a")
        error = ValueError("pipeline worker failed")
        late_task = _RpcTask(method="late_rpc", args=(), kwargs=None, deadline=None, unique_reply_rank=None)
        late_enqueued = False

        def _shutdown_and_enqueue_late_rpc() -> None:
            nonlocal late_enqueued
            if not late_enqueued:
                late_enqueued = True
                engine._rpc_queue.put(late_task)

        engine.executor.shutdown = _shutdown_and_enqueue_late_rpc
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        _, pp0_done, step_done = engine.executor.batches[0]
        pp0_done.set_result(None)
        task = _RpcTask(method="update_weights", args=(), kwargs=None, deadline=None, unique_reply_rank=None)
        with engine._cv:
            engine._rpc_queue.put(task)
            engine._cv.notify_all()
        await _wait_until(lambda: engine._rpc_queue.empty())

        step_done.set_exception(error)

        await _wait_until(lambda: not engine.worker_thread.is_alive())
        assert task.future.exception(timeout=0) is error
        assert isinstance(late_task.future.exception(timeout=0), RuntimeError)
        assert engine.executor.rpc_calls == []


@pytest.mark.asyncio
async def test_engine_shutdown_drains_submitted_pipeline_batches():
    with _running_engine() as engine:
        _add_request(engine, "a")
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        _, _, step_done = engine.executor.batches[0]
        with engine._cv:
            engine.stop_event.set()
            engine._cv.notify_all()
        await asyncio.sleep(0.03)
        assert engine.worker_thread.is_alive()

        step_done.set_result(_step_output("a"))
        await _wait_until(lambda: not engine.worker_thread.is_alive())
        assert engine.scheduler.has_requests() is False


@pytest.mark.asyncio
async def test_step_pp_emits_request_started_once():
    with _running_engine() as engine:
        stream = _add_request(engine, "req-lifecycle", emit_lifecycle=True)
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        _, pp0_done, step_done = engine.executor.batches[0]
        pp0_done.set_result(None)
        step_done.set_result(_step_output("req-lifecycle"))
        await _wait_until(lambda: not engine.scheduler.has_requests())

        items = []
        while not stream.empty():
            items.append(stream.get_nowait())
        assert [item.request_started for item in items] == [True, False]
        assert [item.finished for item in items] == [False, True]


@pytest.mark.asyncio
async def test_deferred_abort_in_flight_batch_retires_as_aborted():
    with _running_engine(max_num_seqs=2) as engine:
        stream_a = _add_request(engine, "a", steps=2)
        stream_b = _add_request(engine, "b", steps=2)
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        sched_0, pp0_0, step_0 = engine.executor.batches[0]
        assert sched_0.scheduled_request_ids == ["a", "b"]

        engine.abort("a")
        await asyncio.sleep(0.03)
        # Abort of an in-flight batch member stays deferred until step retirement.
        assert stream_a.empty()
        assert "a" in engine._deferred_pipeline_aborts

        pp0_0.set_result(None)
        step_0.set_result(_step_output("a", "b", finished=False))
        await _wait_until(lambda: len(engine.executor.batches) == 2)
        out_a = stream_a.get_nowait()
        assert out_a.aborted is True and out_a.finished is True
        assert "a" not in engine._deferred_pipeline_aborts

        sched_1, pp0_1, step_1 = engine.executor.batches[1]
        assert sched_1.scheduled_request_ids == ["b"]
        pp0_1.set_result(None)
        step_1.set_result(_step_output("b"))
        await _wait_until(lambda: not engine.scheduler.has_requests())
        out_b = stream_b.get_nowait()
        assert out_b.aborted is False and out_b.finished is True


@pytest.mark.parametrize(
    ("pipeline_cls", "executor_supports_async_pp", "vae_patch_parallel_size", "error_match"),
    [
        (type("_SupportedPipeline", (), {"supports_step_pipeline_parallel": True}), True, 1, None),
        (
            type("_SupportedPipeline", (), {"supports_step_pipeline_parallel": True}),
            False,
            1,
            "asynchronous Step-PP support",
        ),
        (
            type("_UnsupportedPipeline", (), {"supports_step_pipeline_parallel": False}),
            True,
            1,
            "asynchronous Step-PP support",
        ),
        (type("_PlainPipeline", (), {}), True, 1, "asynchronous Step-PP support"),
        (None, True, 1, "asynchronous Step-PP support"),
        (
            type("_SupportedPipeline", (), {"supports_step_pipeline_parallel": True}),
            True,
            2,
            "vae_patch_parallel_size=1",
        ),
    ],
)
def test_step_pp_startup_capability_validation(
    monkeypatch,
    pipeline_cls,
    executor_supports_async_pp,
    vae_patch_parallel_size,
    error_match,
):
    import vllm_omni.diffusion.diffusion_engine as engine_module
    from vllm_omni.diffusion.executor.abstract import DiffusionExecutor

    constructed = False

    class _FakeExecutor:
        supports_async_step_pp = executor_supports_async_pp

        def __init__(self, _od_config) -> None:
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(engine_module, "resolve_pipeline_cls", lambda _od_config: pipeline_cls)
    monkeypatch.setattr(DiffusionExecutor, "get_class", staticmethod(lambda _od_config: _FakeExecutor))

    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = OmniDiffusionConfig(
        model="test",
        step_execution=True,
        parallel_config={
            "pipeline_parallel_size": 2,
            "vae_patch_parallel_size": vae_patch_parallel_size,
        },
    )
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    engine.step_execution = True

    if error_match is not None:
        with pytest.raises(ValueError, match=error_match):
            engine._init_executor(engine.od_config)
        assert not constructed
        return

    engine._init_executor(engine.od_config)
    assert constructed
    assert engine._pipeline_async_enabled is True
    engine.scheduler = StepScheduler()
    engine.scheduler.initialize(engine.od_config)
    engine._init_runtime_state()
    assert engine.scheduler.step_pp_enabled is True


@pytest.mark.parametrize(
    ("execution_mode", "pp_size", "expected_execute_method"),
    [
        (DiffusionExecutionMode.STEP_BATCH, 1, "execute_step"),
        (DiffusionExecutionMode.REQUEST_BATCH, 2, "execute_batch"),
    ],
)
def test_non_step_pp_modes_keep_sync_executor_path(monkeypatch, execution_mode, pp_size, expected_execute_method):
    import vllm_omni.diffusion.diffusion_engine as engine_module
    from vllm_omni.diffusion.executor.abstract import DiffusionExecutor

    class _SyncOnlyExecutor:
        supports_async_step_pp = False

        def __init__(self, _od_config) -> None:
            pass

        def execute_step(self, _sched_output):
            return None

        def execute_batch(self, _sched_output):
            return None

    monkeypatch.setattr(engine_module, "resolve_pipeline_cls", lambda _od_config: None)
    monkeypatch.setattr(DiffusionExecutor, "get_class", staticmethod(lambda _od_config: _SyncOnlyExecutor))

    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = OmniDiffusionConfig(
        model="test",
        step_execution=execution_mode == DiffusionExecutionMode.STEP_BATCH,
        parallel_config={"pipeline_parallel_size": pp_size},
    )
    engine.execution_mode = execution_mode
    engine.step_execution = execution_mode == DiffusionExecutionMode.STEP_BATCH
    engine._init_executor(engine.od_config)
    engine.scheduler = StepScheduler()
    engine.scheduler.initialize(engine.od_config)
    engine._init_runtime_state()
    engine._init_execute_fn()

    assert engine._pipeline_async_enabled is False
    assert engine.scheduler.step_pp_enabled is False
    assert engine.execute_fn.__name__ == expected_execute_method


@pytest.mark.asyncio
async def test_submit_step_failure_fails_engine_and_propagates_to_request_and_rpc():
    before_submit = threading.Event()
    allow_submit = threading.Event()
    submit_error = RuntimeError("broadcast enqueue failed")
    with _running_engine() as engine:

        def gated_prepare(_sched_output):
            before_submit.set()
            assert allow_submit.wait(2)

        def failing_submit(_sched_output):
            raise submit_error

        engine._prepare_kv_for_forward = gated_prepare  # type: ignore[method-assign]
        engine.executor.submit_step = failing_submit  # type: ignore[method-assign]
        output = _add_request(engine, "a")
        try:
            assert await asyncio.to_thread(before_submit.wait, 2)
            rpc_task = asyncio.create_task(engine.async_collective_rpc("update_weights"))
            await _wait_until(lambda: not engine._rpc_queue.empty())
            allow_submit.set()

            await _wait_until(lambda: not engine.worker_thread.is_alive())
            assert (await asyncio.wait_for(output.get(), timeout=1)).error == str(submit_error)
            with pytest.raises(RuntimeError, match="broadcast enqueue failed"):
                await asyncio.wait_for(rpc_task, timeout=1)
            assert engine.executor.shutdown_calls == 1
        finally:
            allow_submit.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_pp0", [False, True])
async def test_completion_between_retire_scan_and_wait_does_not_hang(fail_pp0):
    scan_saw_pending = threading.Event()
    allow_wait_phase = threading.Event()
    with _running_engine(max_concurrent_batches=1) as engine:
        orig_retire = engine._retire_ready_results
        armed = False

        def gated_retire():
            nonlocal armed
            orig_retire()
            if not armed and engine._pending_batches:
                armed = True
                scan_saw_pending.set()
                assert allow_wait_phase.wait(2)

        engine._retire_ready_results = gated_retire  # type: ignore[method-assign]
        output = _add_request(engine, "a")
        try:
            assert await asyncio.to_thread(scan_saw_pending.wait, 2)
            _, pp0_done, step_done = engine.executor.batches[0]
            if fail_pp0:
                pp0_done.set_exception(RuntimeError("pp0 crashed before wait"))
            else:
                pp0_done.set_result(None)
                step_done.set_result(_step_output("a"))
            allow_wait_phase.set()

            result = await asyncio.wait_for(output.get(), timeout=1)
            if fail_pp0:
                assert result.error == "pp0 crashed before wait"
                await _wait_until(lambda: not engine.worker_thread.is_alive())
            else:
                assert result.finished is True and result.error is None
        finally:
            allow_wait_phase.set()


@pytest.mark.asyncio
async def test_queued_abort_before_final_retirement_wins_over_completion():
    with _running_engine(max_num_seqs=2) as engine:
        stream_a = _add_request(engine, "a")
        stream_b = _add_request(engine, "b")
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        sched_0, pp0_0, step_0 = engine.executor.batches[0]
        assert sched_0.scheduled_request_ids == ["a", "b"]

        # Hold the condition lock so abort and final step completion become visible
        # together before the busy loop enters Phase 1 retirement.
        with engine._cv:
            engine.abort("a")
            pp0_0.set_result(None)
            step_0.set_result(_step_output("a", "b"))

        out_a = await asyncio.wait_for(stream_a.get(), timeout=1)
        out_b = await asyncio.wait_for(stream_b.get(), timeout=1)
        assert out_a.aborted is True and out_a.finished is True
        assert out_b.aborted is False and out_b.finished is True

        # Aborting after terminal commit must not duplicate output delivery.
        engine.abort("b")
        await asyncio.sleep(0.03)
        assert stream_a.empty() and stream_b.empty()


@pytest.mark.asyncio
async def test_abort_defers_inflight_before_kv_drain_reentry():
    reentered = False
    with _running_engine(max_concurrent_batches=1) as engine:
        output_a = _add_request(engine, "a")
        output_c = _add_request(engine, "c")
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        _, pp0_a, step_a = engine.executor.batches[0]
        pp0_a.set_result(None)
        step_a.set_result(_step_output("a"))

        orig_poll = engine._poll_native_kv

        def reentrant_poll(*, drain_request_ids=None):
            nonlocal reentered
            if drain_request_ids and not reentered:
                reentered = True
                engine._retire_ready_results()
            return orig_poll(drain_request_ids=drain_request_ids)

        engine._poll_native_kv = reentrant_poll  # type: ignore[method-assign]
        with engine._cv:
            engine._abort_requests(["a", "c"])

        assert reentered
        assert (await asyncio.wait_for(output_a.get(), timeout=1)).aborted is True
        assert (await asyncio.wait_for(output_c.get(), timeout=1)).aborted is True


@pytest.mark.asyncio
async def test_public_pause_and_resume_scheduler_for_step_pp():
    with _running_engine() as engine:
        output_a = _add_request(engine, "a")
        await _wait_until(lambda: len(engine.executor.batches) == 1)
        _, pp0_a, step_a = engine.executor.batches[0]
        pp0_a.set_result(None)

        pause_task = asyncio.create_task(engine.async_collective_rpc("pause_scheduler", kwargs={"mode": "keep"}))
        await asyncio.sleep(0.03)
        assert not pause_task.done()
        assert engine.executor.rpc_calls == []

        output_b = _add_request(engine, "b")
        await asyncio.sleep(0.03)
        assert len(engine.executor.batches) == 1

        step_a.set_result(_step_output("a"))
        assert await asyncio.wait_for(pause_task, timeout=1) is None
        assert engine.executor.rpc_calls == ["synchronize_device"]
        assert (await asyncio.wait_for(output_a.get(), timeout=1)).finished is True
        assert len(engine.executor.batches) == 1

        assert await asyncio.wait_for(engine.async_collective_rpc("resume_scheduler"), timeout=1) is None
        await _wait_until(lambda: len(engine.executor.batches) == 2)
        _, pp0_b, step_b = engine.executor.batches[1]
        pp0_b.set_result(None)
        step_b.set_result(_step_output("b"))
        assert (await asyncio.wait_for(output_b.get(), timeout=1)).finished is True


def test_pause_scheduler_validation_rejects_sync_step_and_unknown_mode():
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    engine._pipeline_async_enabled = False

    with pytest.raises(ValueError, match="mode='keep'"):
        engine._check_pause_request({"mode": "drop"})
    with pytest.raises(NotImplementedError, match="asynchronous Step-PP"):
        engine._check_pause_request({"mode": "keep"})
