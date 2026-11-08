# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A DP group of size one has rank zero and must emit PP role replies."""

import pytest
import vllm.distributed.parallel_state as vllm_parallel_state

import vllm_omni.diffusion.distributed.parallel_state as parallel_state
from vllm_omni.diffusion.data import RpcReply
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    "tp,sp,cfg,dp,expected", [(0, 0, 0, 0, True), (1, 0, 0, 0, False), (0, 1, 0, 0, False), (0, 0, 1, 0, False)]
)
def test_stage_leader_emits_once_for_dp_rank_zero(monkeypatch, tp, sp, cfg, dp, expected):
    monkeypatch.setattr(vllm_parallel_state, "get_tensor_model_parallel_rank", lambda: tp)
    monkeypatch.setattr(parallel_state, "get_sequence_parallel_rank", lambda: sp)
    monkeypatch.setattr(parallel_state, "get_classifier_free_guidance_rank", lambda: cfg)
    monkeypatch.setattr(parallel_state, "get_data_parallel_rank", lambda: dp)
    worker = WorkerProc.__new__(WorkerProc)
    assert worker._is_stage_leader() is expected


def test_last_stage_role_reply_retains_local_runner_result(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput

    monkeypatch.setattr(vllm_parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_pipeline_parallel_rank", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_pipeline_parallel_world_size", lambda: 2)
    result = BatchRunnerOutput.from_list([])
    worker = WorkerProc.__new__(WorkerProc)
    worker.gpu_id = 1
    worker.result_mq = object()
    worker.worker = SimpleNamespace(execute_method=lambda *args, **kwargs: result)
    replies: list[RpcReply] = []
    worker._enqueue_result = replies.append

    worker._execute_role_rpc(
        {
            "method": "execute_stepwise",
            "exec_all_ranks": True,
            "reply_roles": {"entry": "e", "result": "r"},
        }
    )

    assert len(replies) == 1
    assert replies[0].rpc_id == "r"
    assert replies[0].result is result
