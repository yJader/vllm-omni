# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regressions for Qwen's two evolving streams crossing a PP boundary."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

import vllm_omni.diffusion.models.qwen_image.qwen_image_transformer as model_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed import pipeline_parallel as pp_module
from vllm_omni.diffusion.models.qwen_image import pipeline_qwen_image as pipeline_module
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformer2DModel
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _ImagePrepare(nn.Module):
    def forward(self, hidden, *args):
        return hidden + 10, None, None


class _TextPrepare(nn.Module):
    def forward(self, hidden):
        return hidden + 20


class _Modulate(nn.Module):
    def forward(self, timestep, img_shapes):
        return timestep, None


class _TimeEmbed(nn.Module):
    def forward(self, *args):
        return torch.ones(1, 4)


class _DualStreamBlock(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        text = encoder_hidden_states + hidden_states.mean(dim=1, keepdim=True)
        return text, hidden_states + text.mean(dim=1, keepdim=True)


class _Norm(nn.Module):
    def forward(self, hidden, temb):
        return hidden


def _model(start, end):
    model = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
    nn.Module.__init__(model)
    model.parallel_config = SimpleNamespace(sequence_parallel_size=1)
    model.image_rope_prepare = _ImagePrepare()
    model.modulate_index_prepare = _Modulate()
    model.txt_norm = nn.Identity()
    model.txt_in = _TextPrepare()
    model.time_text_embed = _TimeEmbed()
    model.start_layer, model.end_layer = start, end
    model.transformer_blocks = nn.ModuleList([_DualStreamBlock() for _ in range(4)])
    model.norm_out, model.proj_out = _Norm(), nn.Identity()
    model.zero_cond_t = False
    return model


def test_pp_boundary_preserves_image_and_updated_text_streams(monkeypatch):
    stage = SimpleNamespace(first=True, last=True)
    monkeypatch.setattr(model_module, "is_pipeline_first_stage", lambda: stage.first)
    monkeypatch.setattr(model_module, "is_pipeline_last_stage", lambda: stage.last)
    monkeypatch.setattr(model_module, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(model_module, "_packed_qk_norm_rope_table", lambda *args: None)
    kwargs = dict(
        hidden_states=torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        encoder_hidden_states=torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
        timestep=torch.ones(1),
        img_shapes=[(1, 1, 3)],
    )
    expected = _model(0, 4)(**kwargs).sample
    stage.last = False
    intermediate = _model(0, 2)(**kwargs)
    assert isinstance(intermediate, IntermediateTensors)
    assert set(intermediate.tensors) == {"hidden_states", "encoder_hidden_states"}
    stage.first, stage.last = False, True
    actual = _model(2, 4)(**kwargs, intermediate_tensors=intermediate).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="requires intermediate_tensors"):
        _model(2, 4)(**kwargs)


def test_terminal_steps_drain_existing_sends_before_decode(monkeypatch):
    class Work:
        waited = False

        def wait(self):
            self.waited = True

        def is_completed(self):
            return self.waited

    works = []

    class LastGroup:
        is_last_rank = True
        is_first_rank = False
        world_size = 2

        def isend_tensor(self, tensor, dst):
            assert dst == 0
            work = Work()
            works.append(work)
            return [work]

    class Scheduler:
        def step(self, noise_pred, timestep, latents, return_dict):
            return (latents + noise_pred,)

    monkeypatch.setattr(pp_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pp_module, "get_pp_group", LastGroup)
    monkeypatch.setattr(pipeline_module, "is_pipeline_last_stage", lambda: True)
    pipeline = pipeline_module.QwenImagePipeline.__new__(pipeline_module.QwenImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(step_execution=True)
    pipeline._interrupt = False

    def decode(latents, height, width, output_type):
        assert all(work.waited for work in works)
        return DiffusionOutput(output=latents)

    pipeline._decode_latents = decode
    for index in range(3):
        # An older transfer still needs its drain even when the current final
        # step itself does not post feedback.
        pending = Work()
        works.append(pending)
        pipeline._pp_send_work = [pending]
        state = StepRequestState(
            request_id=str(index),
            sampling=OmniDiffusionSamplingParams(height=16, width=16, output_type="latent"),
            timesteps=torch.tensor([1.0]),
            latents=torch.zeros(1, 4, 2),
            scheduler=Scheduler(),
        )
        pipeline.step_scheduler(state, torch.ones_like(state.latents))
        result = pipeline.post_decode(state)
        torch.testing.assert_close(result.output, torch.ones_like(state.latents))
        assert pipeline._pp_send_work == []


@pytest.mark.parametrize("last_stage", [False, True])
def test_output_layers_are_allocated_only_on_last_stage(monkeypatch, last_stage):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(model_module, "is_pipeline_last_stage", lambda: last_stage)
    monkeypatch.setattr(model_module, "make_layers", lambda *args, **kwargs: (0, 0, nn.ModuleList()))
    allocated = []
    linear = model_module.ReplicatedLinear

    def recording_linear(*args, **kwargs):
        allocated.append(kwargs.get("prefix"))
        return linear(*args, **kwargs)

    monkeypatch.setattr(model_module, "ReplicatedLinear", recording_linear)
    model = QwenImageTransformer2DModel(
        SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1)),
        in_channels=4,
        out_channels=4,
        num_layers=0,
        num_attention_heads=1,
        attention_head_dim=8,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    )

    assert ("norm_out.linear" in allocated) is last_stage
    assert ("proj_out" in allocated) is last_stage
    assert isinstance(model.norm_out, PPMissingLayer) is not last_stage
    assert isinstance(model.proj_out, PPMissingLayer) is not last_stage


@pytest.mark.parametrize("first_stage", [False, True])
@pytest.mark.parametrize("step_index", [0, 1], ids=["continuation", "terminal"])
def test_qwen_posts_feedback_only_when_another_step_needs_it(monkeypatch, first_stage, step_index):
    """Omitting unused final feedback must agree on its sending and receiving ranks."""
    from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents

    feedback = []

    class Group:
        is_first_rank = first_stage
        is_last_rank = not first_stage
        world_size = 2

        def isend_tensor(self, tensor, dst):
            feedback.append(("send", dst))
            return []

        def irecv_tensor(self, tensor, src):
            feedback.append(("receive", src))
            return []

    class Scheduler:
        def step(self, noise_pred, timestep, latents, return_dict):
            return (latents + noise_pred,)

    monkeypatch.setattr(pp_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pp_module, "get_pp_group", Group)
    pipeline = object.__new__(pipeline_module.QwenImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(step_execution=True)
    pipeline._interrupt = False
    state = StepRequestState(
        request_id="request",
        sampling=OmniDiffusionSamplingParams(),
        timesteps=torch.tensor([1.0, 0.5]),
        step_index=step_index,
        latents=torch.zeros(1, 4, 2),
        scheduler=Scheduler(),
    )
    pipeline.step_scheduler(state, None if first_stage else torch.ones_like(state.latents))

    assert state.step_index == step_index + 1
    if step_index == 0:
        assert feedback == ([("receive", 1)] if first_stage else [("send", 0)])
        assert isinstance(state.latents, AsyncLatents) is first_stage
    else:
        assert feedback == [], "unused final feedback serializes the next batch through its receive stream wait"
        assert isinstance(state.latents, torch.Tensor)
    if not first_stage:
        torch.testing.assert_close(state.latents, torch.ones(1, 4, 2))
