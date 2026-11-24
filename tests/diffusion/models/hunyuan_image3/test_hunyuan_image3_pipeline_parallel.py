# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

import vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer as transformer_module
import vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 as pipeline_module
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import PipelineParallelMixin
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import HunyuanImage3Model
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import HunyuanImage3Pipeline
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _AddOneLayer(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return (hidden_states + 1,)


class _PPGroup:
    is_first_rank = False
    is_last_rank = False


class _FinalPPGroup:
    is_first_rank = False
    is_last_rank = True


def _model() -> HunyuanImage3Model:
    model = torch.nn.Module.__new__(HunyuanImage3Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        use_return_dict=True,
        hidden_size=2,
        tie_word_embeddings=False,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_experts=1,
        use_cla=False,
    )
    model.start_layer = 2
    model.end_layer = 4
    model.layers = torch.nn.ModuleList([PPMissingLayer(), PPMissingLayer(), _AddOneLayer(), _AddOneLayer()])
    model.embed_tokens = PPMissingLayer()
    model.norm = PPMissingLayer()
    model.pre_processor = torch.nn.Identity()
    model.unifiled_cat = torch.nn.Identity()
    model.post_processor = torch.nn.Identity()
    return model


def test_hunyuan_image3_pipeline_uses_pipeline_parallel_mixin():
    assert issubclass(HunyuanImage3Pipeline, PipelineParallelMixin)
    assert HunyuanImage3Pipeline.mro().index(PipelineParallelMixin) < HunyuanImage3Pipeline.mro().index(
        CFGParallelMixin
    )


def test_hunyuan_decoder_consumes_intermediate_states_and_runs_only_local_layers(monkeypatch):
    monkeypatch.setattr(
        transformer_module.current_omni_platform, "reset_diffusion_fused_moe_forward_context", lambda: None
    )
    monkeypatch.setattr(transformer_module, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(transformer_module, "get_pp_group", lambda: _PPGroup())

    hidden_states = torch.zeros(1, 3, 2)
    outputs = _model()(
        input_ids=torch.ones(1, 3, dtype=torch.long),
        inputs_embeds=None,
        intermediate_tensors=IntermediateTensors({"hidden_states": hidden_states}),
        attention_mask=None,
        position_ids=torch.arange(3).unsqueeze(0),
        custom_pos_emb=(torch.ones(1, 3, 2), torch.ones(1, 3, 2)),
        query_lens=[3],
        seq_lens=[3],
        return_dict=True,
    )

    assert isinstance(outputs, IntermediateTensors)
    torch.testing.assert_close(outputs["hidden_states"], torch.full((1, 3, 2), 2.0))


@pytest.mark.parametrize("checkpoint_name", ["wte.weight", "ln_f.weight"])
def test_hunyuan_loader_skips_pp_missing_embedding_and_norm(checkpoint_name):
    model = _model()

    loaded = model.load_weights([(checkpoint_name, torch.ones(2, 2))])

    assert loaded == set()


def test_hunyuan_forward_call_runs_ragged_final_layer_after_last_pp_stage(monkeypatch):
    class _FinalModel(torch.nn.Module):
        def forward(self, **kwargs):
            assert isinstance(kwargs["intermediate_tensors"], IntermediateTensors)
            return BaseModelOutputWithPast(last_hidden_state=torch.full((1, 4, 2), 3.0))

    pipeline = torch.nn.Module.__new__(HunyuanImage3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.config = SimpleNamespace(use_return_dict=True)
    pipeline.hf_config = SimpleNamespace(patch_size=1)
    pipeline.vllm_config = object()
    pipeline.model = _FinalModel()
    final_layer_args: dict[str, object] = {}

    def final_layer(hidden_states, image_mask, timestep, token_h, token_w, first_step, num_special_tokens):
        final_layer_args.update(
            hidden_states=hidden_states,
            token_h=token_h,
            token_w=token_w,
            first_step=first_step,
        )
        return torch.ones(1, 1, 2, 2)

    pipeline.ragged_final_layer = final_layer
    monkeypatch.setattr(transformer_module, "get_pp_group", lambda: _FinalPPGroup())
    monkeypatch.setattr("vllm.forward_context.set_forward_context", lambda *_args, **_kwargs: nullcontext())

    result = HunyuanImage3Pipeline.forward_call(
        pipeline,
        input_ids=None,
        attention_mask=torch.ones(1, 1, 4, 4),
        position_ids=torch.arange(4).unsqueeze(0),
        custom_pos_emb=(torch.ones(1, 4, 2), torch.ones(1, 4, 2)),
        mode="gen_image",
        first_step=True,
        images=torch.ones(1, 1, 2, 2),
        image_mask=torch.tensor([[False, False, True, True]]),
        timestep=torch.tensor([1.0]),
        gen_timestep_scatter_index=torch.tensor([[1]]),
        query_lens=[4],
        seq_lens=[4],
        intermediate_tensors=IntermediateTensors({"hidden_states": torch.zeros(1, 4, 2)}),
    )

    assert result.diffusion_prediction.shape == (1, 1, 2, 2)
    assert final_layer_args["token_h"] == 2
    assert final_layer_args["token_w"] == 2
    assert final_layer_args["first_step"] is True
    torch.testing.assert_close(final_layer_args["hidden_states"], torch.full((1, 4, 2), 3.0))


@pytest.mark.parametrize("missing_prefix_layers", [0, 2])
def test_step_cache_split_uses_local_layer_lengths(missing_prefix_layers):
    from vllm_omni.diffusion.models.hunyuan_image3 import pipeline_hunyuan_image3 as pipeline_module

    pipeline = torch.nn.Module.__new__(HunyuanImage3Pipeline)
    torch.nn.Module.__init__(pipeline)
    cache = [None] * missing_prefix_layers + [{"lens": torch.tensor([3])}]
    state = SimpleNamespace(
        extra={
            pipeline_module._STEP_CFG_FACTOR: 1,
            pipeline_module._STEP_MODEL_KWARGS: {},
            pipeline_module._STEP_PROMPT_KV: cache,
        }
    )
    mask = torch.arange(16).reshape(1, 1, 2, 8)

    pipeline._split_merged_kwargs_to_states(
        [state],
        {"attention_mask": mask},
        [0],
        [0],
    )

    result = state.extra[pipeline_module._STEP_MODEL_KWARGS]
    torch.testing.assert_close(result["attention_mask"], torch.cat([mask[..., :3], mask[..., 6:]], dim=-1))
    assert result["query_lens"] == [2]
    assert result["seq_lens"] == [5]


def test_hunyuan_feedback_preserves_latent_dtype_before_fixed_shape_send(monkeypatch):
    from vllm_omni.diffusion.distributed import pipeline_parallel as pp_module
    from vllm_omni.diffusion.models.hunyuan_image3 import pipeline_hunyuan_image3 as pipeline_module

    sent = []

    class _LastGroup:
        is_last_rank = True
        is_first_rank = False
        world_size = 2

        def isend_tensor(self, tensor, dst):
            assert dst == 0
            sent.append(tensor)
            return []

    monkeypatch.setattr(pipeline_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pp_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pp_module, "get_pp_group", lambda: _LastGroup())
    scheduler = pipeline_module.HunyuanImage3FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(2)
    pipeline = torch.nn.Module.__new__(HunyuanImage3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(step_execution=True)
    state = StepRequestState(
        request_id="request",
        sampling=OmniDiffusionSamplingParams(),
        extra={pipeline_module._STEP_GENERATOR: torch.Generator().manual_seed(42)},
        latents=torch.zeros(1, 32, 2, 2, dtype=torch.bfloat16),
        timesteps=scheduler.timesteps,
        scheduler=scheduler,
    )

    pipeline.step_scheduler(state, torch.ones_like(state.latents, dtype=torch.float32))

    assert len(sent) == 1
    assert sent[0].dtype == torch.bfloat16
    assert state.latents.dtype == torch.bfloat16
    assert state.step_index == 1
    assert not state.denoise_completed


@pytest.mark.parametrize("last_stage", [False, True])
def test_mixed_phase_batch_executes_every_group_and_preserves_output_order(last_stage):
    pipeline = torch.nn.Module.__new__(HunyuanImage3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._uses_scheduler_paged_kv = lambda: False
    pipeline._ensure_grouped_attention_backend_supported = lambda count: None
    states = [
        StepRequestState(
            request_id=str(index),
            sampling=OmniDiffusionSamplingParams(),
            step_index=step,
            timesteps=torch.arange(3),
            latents=torch.full((1, 4, 2, 2), float(index)),
            extra={
                pipeline_module._STEP_CFG_FACTOR: 1,
                pipeline_module._STEP_MODEL_KWARGS: {"num_image_tokens": 4},
            },
        )
        for index, step in enumerate((0, 1, 0))
    ]
    calls = []

    def denoise_group(group):
        calls.append([state.request_id for state in group])
        return torch.cat([state.latents for state in group]) if last_stage else None

    pipeline._denoise_step_group = denoise_group
    output = pipeline.denoise_step(InputBatch.make_batch(states))

    assert calls == [["0", "2"], ["1"]]
    if last_stage:
        torch.testing.assert_close(output, torch.cat([state.latents for state in states]))
    else:
        assert output is None


@pytest.mark.parametrize("first_step", [True, False])
@pytest.mark.parametrize("return_dict", [True, False])
def test_forward_call_preserves_single_stage_and_split_stage_outputs(monkeypatch, first_step, return_dict):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding.from_pretrained(torch.arange(8.0).reshape(4, 2))
            self.increment = 2
            self.last_stage = True

        def forward(self, inputs_embeds, intermediate_tensors=None, return_dict=True, **kwargs):
            hidden = inputs_embeds if intermediate_tensors is None else intermediate_tensors["hidden_states"]
            hidden = hidden + self.increment
            if not self.last_stage:
                return IntermediateTensors({"hidden_states": hidden})
            return BaseModelOutputWithPast(last_hidden_state=hidden) if return_dict else (hidden,)

    pipeline = torch.nn.Module.__new__(HunyuanImage3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.config = SimpleNamespace(use_return_dict=True, hidden_size=2)
    pipeline.hf_config = SimpleNamespace(patch_size=1)
    pipeline.vllm_config = object()
    pipeline.model = Model()
    pipeline._uses_scheduler_paged_kv = lambda: False
    pipeline.instantiate_vae_image_tokens = lambda embeds, *args: (embeds, 2, 2)
    pipeline.instantiate_timestep_tokens = lambda embeds, *args: embeds + 1
    pipeline.instantiate_guidance_tokens = lambda embeds, *args: embeds
    pipeline.instantiate_timestep_r_tokens = lambda embeds, *args: embeds
    pipeline.time_embed = lambda timestep: timestep
    pipeline.patch_embed = lambda images, timestep: (images.flatten(2).transpose(1, 2).expand(-1, -1, 2), 2, 2)
    pipeline.timestep_emb = lambda timestep: torch.ones(1, 1, 2)
    pipeline.ragged_final_layer = lambda hidden, *args: hidden
    monkeypatch.setattr("vllm.forward_context.set_forward_context", lambda *args, **kwargs: nullcontext())
    token_count = 4 if first_step else 5
    kwargs = dict(
        input_ids=torch.arange(4).reshape(1, 4) if first_step else None,
        position_ids=torch.arange(token_count).reshape(1, token_count),
        images=torch.ones(1, 1, 2, 2),
        image_mask=torch.ones(1, 4, dtype=torch.bool),
        timestep=torch.tensor([1.0]),
        gen_timestep_scatter_index=torch.tensor([[0]]),
        custom_pos_emb=(torch.ones(1, token_count, 2), torch.ones(1, token_count, 2)),
        mode="gen_image",
        first_step=first_step,
        return_dict=return_dict,
    )
    reference = pipeline.forward_call(**kwargs)
    expected = reference.diffusion_prediction if return_dict else reference[-1]

    pipeline.model.increment = 1
    pipeline.model.last_stage = False
    intermediate = pipeline.forward_call(**kwargs)
    assert isinstance(intermediate, IntermediateTensors)
    pipeline.model.last_stage = True
    result = pipeline.forward_call(**kwargs, intermediate_tensors=intermediate)

    torch.testing.assert_close(result.diffusion_prediction, expected, rtol=0, atol=0)


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("seed", [None, 42])
def test_scheduler_uses_request_state_and_preserves_dtype(monkeypatch, pp_size, seed):
    from vllm_omni.diffusion.distributed import pipeline_parallel as pp_module

    class LastGroup:
        is_last_rank = True

        def isend_tensor(self, tensor, dst):
            assert dst == 0
            return []

    monkeypatch.setattr(pp_module, "get_pipeline_parallel_world_size", lambda: pp_size)
    monkeypatch.setattr(pp_module, "get_pp_group", LastGroup)
    pipeline = torch.nn.Module.__new__(HunyuanImage3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(step_execution=True)
    schedulers = [pipeline_module.HunyuanImage3FlowMatchEulerDiscreteScheduler() for _ in range(2)]
    for scheduler in schedulers:
        scheduler.set_timesteps(3)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    latents = torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16)
    state = StepRequestState(
        request_id="request",
        sampling=OmniDiffusionSamplingParams(),
        timesteps=schedulers[0].timesteps,
        scheduler=schedulers[0],
        latents=latents.clone(),
        extra={pipeline_module._STEP_GENERATOR: generator},
    )
    for step, timestep in enumerate(state.timesteps):
        prediction = torch.full(latents.shape, float(step + 1))
        latents = schedulers[1].step(prediction, timestep, latents, return_dict=False)[0].to(torch.bfloat16)
        pipeline.step_scheduler(state, prediction)
        assert state.step_index == step + 1
        assert state.latents.dtype == torch.bfloat16
        torch.testing.assert_close(state.latents, latents, rtol=0, atol=0)


@pytest.mark.parametrize("first_stage", [False, True], ids=["sender", "receiver"])
@pytest.mark.parametrize("step_index", [0, 1], ids=["continuation", "terminal"])
def test_hunyuan_posts_feedback_only_when_another_step_needs_it(monkeypatch, first_stage, step_index):
    from vllm_omni.diffusion.distributed import pipeline_parallel as pp_module
    from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents

    feedback = []

    class Group:
        is_first_rank = first_stage
        is_last_rank = not first_stage
        world_size = 2

        def isend_tensor(self, tensor, dst):
            assert dst == 0
            feedback.append("send")
            assert tensor.dtype == torch.bfloat16
            return []

        def irecv_tensor(self, tensor, src):
            assert src == 1
            feedback.append("receive")
            assert tensor.dtype == torch.bfloat16
            return []

    class Scheduler:
        def step(self, noise_pred, timestep, latents, return_dict):
            return (latents + noise_pred,)

    monkeypatch.setattr(pp_module, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pp_module, "get_pp_group", Group)
    pipeline = torch.nn.Module.__new__(HunyuanImage3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(step_execution=True)
    latents = torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16)
    state = StepRequestState(
        request_id="request",
        sampling=OmniDiffusionSamplingParams(),
        timesteps=torch.tensor([1.0, 0.5]),
        step_index=step_index,
        latents=latents,
        scheduler=Scheduler(),
    )
    prediction = None if first_stage else torch.ones_like(latents, dtype=torch.float32)

    pipeline.step_scheduler(state, prediction)

    assert state.step_index == step_index + 1
    if step_index == 0:
        assert feedback == (["receive"] if first_stage else ["send"])
        assert isinstance(state.latents, AsyncLatents) is first_stage
    else:
        assert feedback == [], "terminal latents are consumed only on the last PP stage"
        assert isinstance(state.latents, torch.Tensor)
        if first_stage:
            assert state.latents is latents
    if not first_stage:
        assert state.latents.dtype == torch.bfloat16
        torch.testing.assert_close(state.latents, torch.ones_like(latents), rtol=0, atol=0)
