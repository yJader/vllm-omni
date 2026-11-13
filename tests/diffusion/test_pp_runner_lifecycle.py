# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pending PP feedback must survive step completion and drain before retirement."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed import pipeline_parallel as pp_module
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents, PipelineParallelMixin
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _Work:
    def __init__(self):
        self.waits = 0
        self.completed = False

    def wait(self):
        self.waits += 1
        self.completed = True

    def is_completed(self):
        return self.completed


@pytest.fixture
def pending_feedback():
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = SimpleNamespace()
    state = StepRequestState(
        request_id="request",
        sampling=OmniDiffusionSamplingParams(),
        latents=torch.zeros(1, 4, 2),
        timesteps=torch.arange(3),
    )
    runner.state_cache = {state.request_id: state}
    runner.input_batch = None
    _, batch, _ = runner._prepare_batch_inputs([state], [])
    work = _Work()
    state.latents = AsyncLatents({"latents": torch.full((1, 4, 2), 7.0)}, [work], [])
    return runner, state, batch, work


def test_step_tail_defers_feedback_until_next_prepare(pending_feedback):
    runner, state, batch, work = pending_feedback
    runner._update_states_after([state], batch)

    assert work.waits == 0
    assert isinstance(state.latents, AsyncLatents)
    _, next_batch, _ = runner._prepare_batch_inputs([state], [])
    assert work.waits == 1
    assert isinstance(state.latents, torch.Tensor)
    torch.testing.assert_close(next_batch.latents, torch.full((1, 4, 2), 7.0))


def test_release_drains_feedback_before_model_hook(pending_feedback):
    """Mid-batch abort still drains to avoid leaking comm handles."""
    runner, state, _, work = pending_feedback
    observations = []
    runner.pipeline.release_step_request = lambda state: observations.append(work.waits)

    assert not state.request_denoise_completed

    runner.release_step_requests([state.request_id])

    assert work.waits == 1
    assert observations == [1]
    assert state.request_id not in runner.state_cache
    assert runner.input_batch is None


@pytest.mark.parametrize("terminal", ["completed", "interrupted", "failed"])
def test_terminal_step_drains_feedback_before_model_hook(pending_feedback, terminal):
    """Unused feedback still owns receive storage until retirement drains it."""
    runner, state, batch, work = pending_feedback
    observations = []

    def release(released_state):
        assert released_state is state
        assert work.waits == 1
        assert isinstance(released_state.latents, torch.Tensor)
        torch.testing.assert_close(released_state.latents, torch.full((1, 4, 2), 7.0))
        observations.append(released_state.request_id)

    runner.pipeline.release_step_request = release
    if terminal == "completed":
        state.step_index = state.total_steps
        assert state.request_denoise_completed
    elif terminal == "failed":
        # Per-request failure removes the cache entry before the step tail;
        # the cached batch must retain the state until its feedback is drained.
        runner.state_cache.pop(state.request_id)
        assert batch.states[0] is state

    runner._update_states_after([state], batch, interrupted=terminal == "interrupted")

    assert work.waits == 1
    assert observations == [state.request_id]
    assert state.request_id not in runner.state_cache
    assert runner.input_batch is None

    runner.release_step_requests([state.request_id])
    assert work.waits == 1
    assert observations == [state.request_id]


def test_plain_tensor_updates_refresh_cached_batch(pending_feedback):
    runner, state, batch, _ = pending_feedback
    state.latents = torch.full((1, 4, 2), 9.0)

    runner._update_states_after([state], batch)

    assert runner.input_batch is batch
    assert state.request_id in runner.state_cache
    torch.testing.assert_close(batch.latents, torch.full((1, 4, 2), 9.0))


def test_release_drains_pp_sends_after_feedback(pending_feedback):
    """Pipeline PP sends are drained by the pipeline hook, not the runner."""

    class Pipeline(PipelineParallelMixin, CFGParallelMixin):
        pass

    runner, state, _, receive_work = pending_feedback
    assert not state.request_denoise_completed

    runner.pipeline = Pipeline()
    send_work = _Work()
    runner.pipeline._pp_send_work = [send_work]

    runner.release_step_requests([state.request_id])

    assert receive_work.waits == 1
    assert send_work.waits == 1
    assert runner.pipeline._pp_send_work == []
    assert runner.input_batch is None


def test_last_stage_prunes_completed_feedback_without_waiting(monkeypatch):
    class Pipeline(PipelineParallelMixin, CFGParallelMixin):
        supports_step_pipeline_parallel = True

    class Scheduler:
        def step(self, noise_pred, timestep, latents, return_dict):
            return (latents + noise_pred,)

    works = []

    class LastGroup:
        is_last_rank = True

        def isend_tensor(self, tensor, dst):
            work = _Work()
            works.append(work)
            return [work]

    monkeypatch.setattr(pp_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pp_module, "get_pp_group", LastGroup)
    pipeline = Pipeline()
    pipeline.od_config = SimpleNamespace(step_execution=True)
    for _ in range(3):
        if works:
            works[-1].completed = True
        pipeline.scheduler_step_maybe_with_cfg(
            torch.ones(1), torch.tensor(1.0), torch.zeros(1), False, per_request_scheduler=Scheduler()
        )
        assert pipeline._pp_send_work == [works[-1]]
        assert all(work.waits == 0 for work in works)
