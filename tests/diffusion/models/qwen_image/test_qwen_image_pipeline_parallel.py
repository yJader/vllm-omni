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
    model.reference_precision = False
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


def test_mixfusion_hooks_skip_pp_missing_layers_and_match_single_stage(monkeypatch):
    """MixFusion hooks mount on PP stages with PPMissingLayer and match single-stage output."""
    from vllm_omni.diffusion.forward_context import ForwardContext, MixFusionStep, override_forward_context
    from vllm_omni.diffusion.hooks.mixfusion import apply_mixfusion, remove_mixfusion

    class _MixFusionImagePrepare(nn.Module):
        def forward(self, hidden_states, img_shapes, txt_seq_lens):
            seq_len = hidden_states.shape[1]
            vid_freqs = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1).expand(-1, 2)
            txt_freqs = torch.ones(max(txt_seq_lens), 2, dtype=torch.float32)
            return hidden_states + 10.0, vid_freqs, txt_freqs

    class _MixFusionTimeEmbed(nn.Module):
        def forward(self, timestep, hidden_states, additional_t_cond=None):
            return timestep.unsqueeze(-1).expand(-1, 4).to(dtype=hidden_states.dtype)

    class _MixFusionBlock(nn.Module):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.img_mod = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.img_mod.weight.fill_(scale)

        def forward(self, hidden_states, encoder_hidden_states, temb, **kwargs):
            mod = self.img_mod(temb)
            assert mod.shape[0] == hidden_states.shape[0]
            ctx = model_module.get_forward_context()
            step = getattr(ctx, "mixfusion_step", None)
            if step is not None and step.plan is not None:
                # Re-index text stream per chunk so dual-stream coupling matches per-request execution.
                chunk_text = encoder_hidden_states.index_select(0, step.plan.chunk_to_request)
                img_out = hidden_states + mod.unsqueeze(1) + chunk_text.mean(dim=1, keepdim=True)
                # Update each request's text stream from its own image chunks only.
                txt_updates = [
                    hidden_states[start:end].mean(dim=(0, 1), keepdim=True)
                    for start, end in step.plan.request_chunk_ranges
                ]
                txt_out = encoder_hidden_states + torch.cat(txt_updates, dim=0)
            else:
                img_out = hidden_states + mod.unsqueeze(1) + encoder_hidden_states.mean(dim=1, keepdim=True)
                txt_out = encoder_hidden_states + hidden_states.mean(dim=1, keepdim=True)
            return txt_out, img_out

    class _MixFusionNorm(nn.Module):
        def forward(self, x, conditioning_embedding):
            assert conditioning_embedding.shape[0] == x.shape[0]
            return x + conditioning_embedding.unsqueeze(1)

    def _build_stage(start: int, end: int, *, last_stage: bool) -> QwenImageTransformer2DModel:
        model = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
        nn.Module.__init__(model)
        model.parallel_config = SimpleNamespace(sequence_parallel_size=1)
        model.reference_precision = True
        model.image_rope_prepare = _MixFusionImagePrepare()
        model.modulate_index_prepare = _Modulate()
        model.txt_norm = nn.Identity()
        model.txt_in = _TextPrepare()
        model.time_text_embed = _MixFusionTimeEmbed()
        model.start_layer, model.end_layer = start, end
        model.transformer_blocks = nn.ModuleList(
            [_MixFusionBlock(0.1 * (i + 1)) if start <= i < end else PPMissingLayer() for i in range(4)]
        )
        if last_stage:
            model.norm_out = _MixFusionNorm()
            model.proj_out = nn.Identity()
        else:
            model.norm_out = PPMissingLayer()
            model.proj_out = PPMissingLayer()
        model.zero_cond_t = False
        return model

    stage = SimpleNamespace(first=True, last=True)
    monkeypatch.setattr(model_module, "is_pipeline_first_stage", lambda: stage.first)
    monkeypatch.setattr(model_module, "is_pipeline_last_stage", lambda: stage.last)

    torch.manual_seed(0)
    latents = [torch.randn(1, 2, 4), torch.randn(1, 4, 4)]
    img_shapes = [[(1, 1, 2)], [(1, 2, 2)]]
    encoder_hidden_states = torch.randn(2, 3, 4)
    encoder_hidden_states_mask = torch.ones(2, 3, dtype=torch.bool)
    timestep = torch.tensor([1.0, 2.0])

    # Single-stage unbatched reference per request.
    reference = _build_stage(0, 4, last_stage=True)
    expected = []
    with override_forward_context(ForwardContext()):
        for idx in range(2):
            out = reference(
                hidden_states=latents[idx],
                encoder_hidden_states=encoder_hidden_states[idx : idx + 1],
                encoder_hidden_states_mask=encoder_hidden_states_mask[idx : idx + 1],
                timestep=timestep[idx : idx + 1],
                img_shapes=img_shapes[idx],
                txt_seq_lens=[3],
            )
            expected.append(out.sample)

    # 2-stage PP with PPMissingLayer on both stages + MixFusion hooks mounted.
    stage0 = _build_stage(0, 2, last_stage=False)
    stage1 = _build_stage(2, 4, last_stage=True)
    apply_mixfusion(stage0, stage0._mixfusion_plan)
    apply_mixfusion(stage1, stage1._mixfusion_plan)

    packed_kwargs = dict(
        hidden_states=latents,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        timestep=timestep,
        img_shapes=img_shapes,
        txt_seq_lens=[3, 3],
    )
    stage.first, stage.last = True, False
    with override_forward_context(ForwardContext(mixfusion_step=MixFusionStep())):
        intermediate = stage0(**packed_kwargs)
    assert isinstance(intermediate, IntermediateTensors)
    assert tuple(intermediate["hidden_states"].shape) == (3, 2, 4)
    assert tuple(intermediate["encoder_hidden_states"].shape) == (2, 3, 4)

    stage.first, stage.last = False, True
    with override_forward_context(ForwardContext(mixfusion_step=MixFusionStep())):
        actual = stage1(**packed_kwargs, intermediate_tensors=intermediate)
    assert isinstance(actual, list) and len(actual) == 2
    for got, exp in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, exp, rtol=0, atol=1e-6)

    remove_mixfusion(stage0, stage0._mixfusion_plan)
    remove_mixfusion(stage1, stage1._mixfusion_plan)


def test_pipeline_parallel_predict_noise_preserves_mixfusion_list_and_cfg_branch(monkeypatch):
    """Last PP stage must keep list outputs (even len=1) and stamp transformer.cfg_branch."""

    class _FakePPGroup:
        is_first_rank = False
        is_last_rank = True
        world_size = 2

        def irecv_tensor_dict(self):
            return {"hidden_states": torch.zeros(1)}, [], []

    monkeypatch.setattr(pp_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pp_module, "get_pp_group", _FakePPGroup)
    monkeypatch.setattr(pp_module, "get_classifier_free_guidance_world_size", lambda: 1)

    pipeline = object.__new__(pipeline_module.QwenImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(step_execution=True)
    pipeline.transformer = nn.Module()
    seen_branches: list[str | None] = []

    def fake_predict_noise(**kwargs):
        seen_branches.append(getattr(pipeline.transformer, "cfg_branch", None))
        return kwargs["out"]

    pipeline.predict_noise = fake_predict_noise

    # 1-request MixFusion batch without CFG must stay a list.
    single_out = [torch.ones(1, 2, 4)]
    got_single = pipeline.predict_noise_maybe_with_cfg(
        do_true_cfg=False,
        true_cfg_scale=1.0,
        positive_kwargs={"out": single_out},
        negative_kwargs=None,
        cfg_normalize=False,
    )
    assert isinstance(got_single, list) and len(got_single) == 1
    assert torch.equal(got_single[0], single_out[0])
    assert seen_branches == ["positive"]

    # 2-request MixFusion batch with sequential CFG stamps both branches and returns a list.
    seen_branches.clear()
    pos_out = [torch.full((1, 2, 4), 2.0), torch.full((1, 4, 4), 3.0)]
    neg_out = [torch.ones(1, 2, 4), torch.ones(1, 4, 4)]
    got_cfg = pipeline.predict_noise_maybe_with_cfg(
        do_true_cfg=True,
        true_cfg_scale=2.0,
        positive_kwargs={"out": pos_out},
        negative_kwargs={"out": neg_out},
        cfg_normalize=False,
    )
    assert seen_branches == ["positive", "negative"]
    assert isinstance(got_cfg, list) and len(got_cfg) == 2
    torch.testing.assert_close(got_cfg[0], torch.full((1, 2, 4), 3.0))
    torch.testing.assert_close(got_cfg[1], torch.full((1, 4, 4), 5.0))
