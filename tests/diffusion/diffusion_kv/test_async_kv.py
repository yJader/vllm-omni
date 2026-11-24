# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for KV steps, rank ACKs and shared-reader lifetimes."""

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing import shared_memory

import pytest
import torch
from vllm.v1.outputs import KVConnectorOutput

from tests.diffusion.diffusion_kv.test_native_prefetch import _request, _scheduler
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, RpcReply
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.diffusion_kv.kv_connector import KVReceiveProgress
from vllm_omni.diffusion.executor.abstract import _KVOutputAggregator
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.sched import DiffusionRequestStatus, RequestScheduler, StepScheduler
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerProc
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVCacheConfig,
    OmniKVTransferManager,
    ReceiveRole,
    _LegacyKVPayload,
    _TransferTopoConfig,
)
from vllm_omni.distributed.omni_connectors.utils.kv_utils import PER_SAMPLE_KV_COUNT_KEY
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _step(**kwargs):
    kwargs.setdefault("kv_required_request_ids", set())
    kwargs.setdefault("finished_req_ids", set())
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_running_reqs=0,
        num_waiting_reqs=0,
        **kwargs,
    )


def _legacy_request(rid):
    return OmniDiffusionRequest(
        request_id=rid,
        prompt=rid,
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
    )


def _job(rid):
    return {"request_id": rid, "kv_sender_info": None}


def _legacy_scheduler(cls=RequestScheduler, *, receive=True):
    scheduler = cls()
    scheduler.initialize(
        OmniDiffusionConfig.from_kwargs(
            model_class_name="TestDiffusionPipeline",
            omni_kv_config={"need_recv_cache": receive},
        )
    )
    scheduler.async_kv = True
    return scheduler


@pytest.mark.parametrize("cls", [RequestScheduler, StepScheduler])
def test_slow_and_ready_requests_are_isolated(cls):
    scheduler = _legacy_scheduler(cls)
    scheduler.add_request(_legacy_request("slow"))
    scheduler.add_request(_legacy_request("ready"))
    first = scheduler.schedule()
    assert first.kv_prefetch_job == _job("slow")
    assert first.num_scheduled_reqs == 0 and not first.is_empty
    second = scheduler.schedule()
    assert second.kv_prefetch_job == _job("ready") and second.kv_poll_only
    assert scheduler.schedule().kv_prefetch_job is None
    scheduler.finish_kv_transfer(BatchRunnerOutput([], kv_finished_req_ids=["ready"]))
    third = scheduler.schedule()
    assert third.scheduled_request_ids == ["ready"]
    assert scheduler.get_request_state("slow").status == DiffusionRequestStatus.WAITING_FOR_KV


@pytest.mark.parametrize("cls", [RequestScheduler, StepScheduler])
def test_per_sample_kv_request_bypasses_legacy_async_kv(cls):
    scheduler = _legacy_scheduler(cls)
    request = OmniDiffusionRequest(
        request_id="multi",
        prompt="multi",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=2,
            extra_args={PER_SAMPLE_KV_COUNT_KEY: 2},
        ),
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.scheduled_request_ids == ["multi"]
    assert output.kv_prefetch_job is None
    assert not output.kv_poll_only
    assert not output.has_nonblocking_kv_work


@pytest.mark.parametrize("required", [None, set(), {"A"}])
def test_only_empty_required_ids_select_nonblocking_steps(required):
    output = _step(kv_required_request_ids=required, kv_poll_only=True)
    assert output.has_nonblocking_kv_work == (required == set())
    assert not output.is_empty


def test_legacy_job_and_terminal_notification_execute_without_forward_or_poll_flag():
    for output in (_step(kv_prefetch_job=_job("A")), _step(finished_req_ids={"A"})):
        assert output.has_nonblocking_kv_work and not output.is_empty
    assert _step().is_empty


def test_abort_before_legacy_submission_cancels_without_loading():
    scheduler = _legacy_scheduler()
    scheduler.add_request(_legacy_request("A"))
    scheduler.finish_requests("A", DiffusionRequestStatus.FINISHED_ABORTED)
    scheduler.pop_request_state("A")
    output = scheduler.schedule()
    assert output.kv_prefetch_job is None and output.finished_req_ids == {"A"}
    with pytest.raises(ValueError, match="already active"):
        scheduler.add_request(_legacy_request("A"))
    scheduler.finish_kv_transfer(BatchRunnerOutput([], kv_cancelled_req_ids=["A"], kv_cleaned_req_ids=["A"]))
    assert not scheduler.has_requests()
    scheduler.add_request(_legacy_request("A"))
    assert scheduler.schedule().kv_prefetch_job == _job("A")


def test_native_waits_for_all_ranks_before_admission_and_speculation(monkeypatch):
    scheduler, connector = _scheduler(monkeypatch)
    scheduler.async_kv = True
    scheduler.add_request(_request("A"))
    scheduler.add_request(_request("B"))
    output = scheduler.schedule()
    assert output.num_scheduled_reqs == 0
    assert output.kv_transfer_request_ids == {"A/0", "A/1"}
    assert output.kv_prefetch_connector_metadata is None
    aggregator = _KVOutputAggregator(2)
    empty = BatchRunnerOutput([])
    ready = BatchRunnerOutput([], kv_connector_output=KVConnectorOutput(finished_recving={"A/0", "A/1"}))
    # Rank 0 completes in step 2, rank 1 only in step 4. Repeated snapshots
    # from the fast rank are not additional acknowledgements.
    for _ in (2, 3):
        scheduler.finish_kv_transfer(aggregator.aggregate([ready, empty], set()))
        assert scheduler.schedule().num_scheduled_reqs == 0
    scheduler.finish_kv_transfer(aggregator.aggregate([ready, ready], set()))
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["A"]
    assert output.kv_prefetch_connector_metadata == ("B/0", "B/1")
    assert connector.commits == ["A/0", "A/1", "B/0", "B/1"]


def test_native_slow_receive_does_not_block_request_without_remote_kv(monkeypatch):
    scheduler, connector = _scheduler(monkeypatch)
    scheduler.async_kv = True
    scheduler.add_request(_request("slow"))
    local = _request("local")
    local.kv_transfer_params = None
    scheduler.add_request(local)
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["local"]
    assert connector.commits == ["slow/0", "slow/1"]


def test_abort_retains_native_pages_until_receive_and_cleanup_acks(monkeypatch):
    scheduler, _ = _scheduler(monkeypatch)
    scheduler.async_kv = True
    scheduler.add_request(_request("A"))
    scheduler.schedule()
    scheduler.finish_requests("A", DiffusionRequestStatus.FINISHED_ABORTED)
    scheduler.pop_request_state("A")
    manager = scheduler._diffusion_kv_manager
    pending = scheduler.schedule()
    assert not pending.finished_req_ids and pending.kv_poll_only
    assert manager.has_request("A")
    scheduler.finish_kv_transfer(
        BatchRunnerOutput(
            [],
            kv_connector_output=KVConnectorOutput(finished_recving={"A/0", "A/1"}),
        )
    )
    cleanup_step = scheduler.schedule()
    assert cleanup_step.finished_req_ids == {"A"}
    assert cleanup_step.has_nonblocking_kv_work and not cleanup_step.is_empty
    assert not scheduler.schedule().finished_req_ids
    with pytest.raises(ValueError, match="already active"):
        scheduler.add_request(_request("A"))
    aggregator = _KVOutputAggregator(2)
    cleanup = BatchRunnerOutput([], kv_cleaned_req_ids=["A"])
    scheduler.finish_kv_transfer(aggregator.aggregate([cleanup, BatchRunnerOutput([])], set()))
    assert manager.has_request("A")
    scheduler.finish_kv_transfer(aggregator.aggregate([BatchRunnerOutput([]), cleanup], set()))
    assert not manager.has_request("A")
    # Sequence retirement must be queued before the same ID submits new loads.
    with pytest.raises(ValueError, match="already active"):
        scheduler.add_request(_request("A"))
    assert scheduler.schedule().kv_finished_request_ids == {"A/0", "A/1"}
    scheduler.add_request(_request("A"))
    assert scheduler.schedule().kv_transfer_request_ids == {"A/0", "A/1"}


def test_failure_is_available_before_other_ranks_and_generates_cancel():
    scheduler = _legacy_scheduler()
    scheduler.add_request(_legacy_request("bad"))
    scheduler.schedule()
    aggregator = _KVOutputAggregator(2)
    errors = aggregator.record_failures(BatchRunnerOutput([], kv_failed_req_ids={"bad": "timeout"}))
    assert errors == {"bad": "timeout"}
    scheduler.finish_kv_transfer(BatchRunnerOutput([], kv_failed_req_ids=errors))
    assert scheduler.get_request_state("bad").status == DiffusionRequestStatus.FINISHED_ERROR
    assert scheduler.schedule().finished_req_ids == {"bad"}
    scheduler.finish_kv_transfer(
        aggregator.aggregate(
            [
                BatchRunnerOutput([], kv_finished_req_ids=["bad"]),
                BatchRunnerOutput([], kv_finished_req_ids=["bad"]),
            ],
            set(),
        )
    )
    assert scheduler.get_request_state("bad").status == DiffusionRequestStatus.FINISHED_ERROR


def test_no_kv_does_not_enter_receive_or_broadcast(mocker):
    scheduler = _legacy_scheduler(receive=False)
    scheduler.add_request(_legacy_request("no-kv"))
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["no-kv"]
    assert output.kv_prefetch_job is None and not output.kv_poll_only and not output.has_nonblocking_kv_work
    manager = OmniKVTransferManager(OmniKVCacheConfig())
    manager.distribute_kv_cache = mocker.Mock(side_effect=AssertionError("unexpected broadcast"))
    assert not manager.receive_multi_kv_cache_distributed(_legacy_request("no-kv"))
    assert manager._connector is None


def test_dummy_load_and_cancel_never_initialize_connector():
    manager = _manager()
    assert manager.start_receive_kv_async(_job(DUMMY_DIFFUSION_REQUEST_ID), torch.device("cpu")) == []
    assert manager.cancel_receive_kv(DUMMY_DIFFUSION_REQUEST_ID) == []
    assert manager.poll_legacy_kv() == ([], {}, [DUMMY_DIFFUSION_REQUEST_ID])
    assert manager._connector is None
    manager.close()


@pytest.mark.parametrize(
    "pp_rank,cfg_size,cfg_rank,expected",
    [
        (0, 1, 0, ["pp"]),
        (1, 1, 0, ["pp"]),
        (0, 2, 0, ["cfg_send", "sp", "pp"]),
        (0, 2, 1, ["cfg_recv", "sp", "pp"]),
        (1, 2, 1, ["pp"]),
    ],
)
def test_sync_tp_fallback_distributes_cfg_sp_before_pp(pp_rank, cfg_size, cfg_rank, expected, mocker):
    manager = _manager()
    leader = pp_rank == 0 and cfg_rank == 0
    manager._topo_config = _TransferTopoConfig(
        role=ReceiveRole.LEADER if leader else ReceiveRole.FOLLOWER,
        tp_active=True,
        cfg_size=cfg_size,
        cfg_rank=cfg_rank,
        cfg_group="cfg",
        sp_size=cfg_size,
        sp_rank=0,
        sp_group="sp",
        world="world",
        pp_size=2,
        pp_rank=pp_rank,
        pp_group="pp",
    )
    payload = {"past_key_values": object()}
    calls = []
    manager._collect_request_kv_payload = mocker.Mock(return_value=payload)
    manager._build_cfg_rank_local_payloads = mocker.Mock(return_value=[payload, payload])
    manager._cfg_scatter = mocker.Mock(side_effect=lambda *args: calls.append("cfg_send"))

    def receive(*args):
        calls.append("cfg_recv")
        return payload

    def broadcast(group, data, device, src):
        calls.append(group)
        assert data is (payload if pp_rank == 0 else None)
        return payload

    manager._recv_kv_payload = receive
    manager._broadcast_kv_payload = broadcast
    assert manager.distribute_kv_cache(_legacy_request("A"), received=leader) is payload
    assert calls == expected
    manager.close()


def test_sync_pp0_readers_preserve_tp_receive_consensus():
    from tests.distributed.omni_connectors.test_kv_recv_tp_consensus import _FakeGroup, _req_with_kv, _topo

    group = _FakeGroup(0, 2, peer_signatures=[(False, ())])
    manager = _manager()
    manager._topo_config = replace(_topo(tp_group=group), role=ReceiveRole.LEADER, pp_size=2)
    request = _req_with_kv()
    assert manager._tp_local_receive_consensus(request, True) is False
    assert group.broadcasts == [False]
    assert not manager._collect_request_kv_payload(request)
    manager.close()


def test_empty_step_never_activates_model_or_gathers(monkeypatch, mocker):
    worker = object.__new__(DiffusionWorker)
    runner = mocker.Mock(spec=DiffusionModelRunner)
    worker.model_runner = runner
    runner.poll_kv_transfers.return_value = BatchRunnerOutput([])
    monkeypatch.setattr(torch.distributed, "all_gather_object", mocker.Mock(side_effect=AssertionError("gather")))
    result = worker.execute_stepwise(_step(kv_poll_only=True))
    assert result.runner_outputs == []
    runner.execute_stepwise.assert_not_called()


def test_worker_consumes_job_once_before_full_forward_fallback(mocker):
    worker = object.__new__(DiffusionWorker)
    runner = mocker.Mock(spec=DiffusionModelRunner)
    worker.model_runner = runner
    worker._step_lora_state = {}
    worker._activate_step_lora = mocker.Mock()
    worker._get_profiler = lambda: None
    runner.poll_kv_transfers.return_value = BatchRunnerOutput([], kv_cleaned_req_ids=["old"])
    runner.execute_stepwise.return_value = BatchRunnerOutput([RunnerOutput("A", finished=False)])
    output = _step(kv_prefetch_job=_job("B"), finished_req_ids={"old"})
    output.scheduled_cached_reqs.request_ids.append("A")
    result = worker.execute_stepwise(output)
    forwarded = runner.execute_stepwise.call_args.args[0]
    assert forwarded.kv_prefetch_job is None and not forwarded.finished_req_ids
    assert output.kv_prefetch_job == _job("B") and output.finished_req_ids == {"old"}
    assert result.kv_cleaned_req_ids == ["old"]
    assert result.get_request_output("A") is not None


def test_native_cleanup_notification_follows_receive_then_all_rank_cleanup(monkeypatch, mocker):
    scheduler, _ = _scheduler(monkeypatch)
    scheduler.async_kv = True
    scheduler.add_request(_request("A"))
    runners = []
    for _ in range(2):
        runner = object.__new__(DiffusionModelRunner)
        runner.od_config = scheduler.od_config
        runner.kv_transfer_manager = None
        runner._kv_connector = mocker.Mock()
        runner._kv_connector.post_forward.return_value = KVConnectorOutput()
        runner._kv_receive_progress = KVReceiveProgress()
        runner.state_cache = {}
        runner.diffusion_kv_backend = mocker.Mock()
        runners.append(runner)
    aggregator = _KVOutputAggregator(2)

    def poll(output):
        results = [runner.poll_kv_transfers(output) for runner in runners]
        scheduler.finish_kv_transfer(aggregator.aggregate(results, output.kv_finished_request_ids))

    poll(scheduler.schedule())
    scheduler.finish_requests("A", DiffusionRequestStatus.FINISHED_ABORTED)
    scheduler.pop_request_state("A")
    runners[0]._kv_connector.post_forward.return_value = KVConnectorOutput(finished_recving={"A/0", "A/1"})
    poll(scheduler.schedule())
    for runner in runners:
        runner.diffusion_kv_backend.remove_diffusion_kv_requests.assert_not_called()
    runners[1]._kv_connector.post_forward.return_value = KVConnectorOutput(finished_recving={"A/0", "A/1"})
    poll(scheduler.schedule())
    cleanup = scheduler.schedule()
    assert cleanup.finished_req_ids == {"A"}
    first = runners[0].poll_kv_transfers(cleanup)
    scheduler.finish_kv_transfer(aggregator.aggregate([first, BatchRunnerOutput([])], set()))
    assert scheduler._diffusion_kv_manager.has_request("A")
    second = runners[1].poll_kv_transfers(cleanup)
    scheduler.finish_kv_transfer(aggregator.aggregate([BatchRunnerOutput([]), second], set()))
    assert not scheduler._diffusion_kv_manager.has_request("A")
    for runner in runners:
        runner.diffusion_kv_backend.remove_diffusion_kv_requests.assert_called_once_with(["A"])
    poll(scheduler.schedule())
    assert all(not runner._kv_receive_progress.submitted for runner in runners)


def test_native_poll_does_not_wait_for_unready_transfer(monkeypatch, mocker):
    progress = KVReceiveProgress()
    connector = mocker.Mock()
    connector.post_forward.return_value = KVConnectorOutput()
    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_kv.kv_connector.time.sleep", mocker.Mock(side_effect=AssertionError("wait"))
    )
    result = progress.prepare(
        connector,
        _step(
            kv_connector_metadata=object(),
            kv_transfer_request_ids={"A"},
            kv_required_request_ids=set(),
        ),
        10,
    )
    assert not result.finished_recving
    connector.post_forward.assert_called_once()


def test_engine_poll_does_not_drain_pipeline(mocker):
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = _legacy_scheduler()
    engine.executor = mocker.Mock()
    engine._execute_control_operation_with_drain = mocker.Mock(side_effect=AssertionError("drain"))
    engine._poll_native_kv()
    engine.executor.collective_rpc.assert_not_called()


def test_sync_engine_does_not_resubmit_prepared_connector_work(mocker):
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = _legacy_scheduler()
    engine.scheduler.async_kv = False
    engine._prepare_kv_for_forward = mocker.Mock()
    engine._poll_native_kv = mocker.Mock()
    engine.execute_fn = mocker.Mock(return_value=BatchRunnerOutput([]))
    output = _step(kv_connector_metadata=object(), kv_transfer_request_ids={"B/0"}, kv_prefetch_job=_job("C"))
    output.scheduled_cached_reqs.request_ids.append("A")
    engine._execute_scheduled_step(output)
    engine._prepare_kv_for_forward.assert_called_once_with(output)
    forwarded = engine.execute_fn.call_args.args[0]
    assert not forwarded.has_nonblocking_kv_work
    assert forwarded.kv_connector_metadata is None and not forwarded.kv_transfer_request_ids
    assert forwarded.kv_prefetch_job == _job("C")
    assert output.kv_transfer_request_ids == {"B/0"}


def test_request_mode_prefetch_is_consumed_by_maintenance_once(mocker):
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = _legacy_scheduler()
    engine.execution_mode = DiffusionExecutionMode.REQUEST_BATCH
    engine.executor = mocker.Mock()
    engine.executor.execute_step.return_value = BatchRunnerOutput([], kv_finished_req_ids=["B"])
    engine.execute_fn = mocker.Mock(return_value=BatchRunnerOutput([RunnerOutput("A", finished=True)]))
    output = _step(kv_prefetch_job=_job("B"))
    output.scheduled_cached_reqs.request_ids.append("A")
    result = engine._execute_scheduled_step(output)
    maintenance = engine.executor.execute_step.call_args.args[0]
    assert maintenance.kv_prefetch_job == _job("B") and not maintenance.scheduled_request_ids
    forwarded = engine.execute_fn.call_args.args[0]
    assert forwarded.kv_prefetch_job is None
    assert result.kv_finished_req_ids == ["B"] and result.get_request_output("A").finished


def _manager():
    return OmniKVTransferManager(
        OmniKVCacheConfig(
            need_recv_cache=True,
            from_stage="ar",
            stage_id="dit",
            recv_timeout=0.03,
        )
    )


def test_background_slow_receive_does_not_block_ready_or_discard_prefetch(mocker):
    manager = _manager()
    manager._connector = mocker.Mock(spec=SharedMemoryConnector)
    slow_entered, release = threading.Event(), threading.Event()

    def receive(rid, *args, **kwargs):
        if rid == "slow":
            slow_entered.set()
            assert release.wait(2)
        return {"layer_blocks": {"key_cache": [torch.ones(2)], "value_cache": []}}, 8

    manager.receive_kv_cache_for_request = receive
    retained_prefetch: Future[object] = Future()
    manager._prefetch_futures["next"] = retained_prefetch
    try:
        manager.start_receive_kv_async(_job("slow"), torch.device("cpu"))
        assert slow_entered.wait(1)
        manager.start_receive_kv_async(_job("ready"), torch.device("cpu"))
        manager._receive_futures["ready"].result(timeout=1)
        finished, failed, _ = manager.poll_legacy_kv()
        assert finished == ["ready"] and not failed
        assert manager.consume_received_kv(_legacy_request("ready"))
        assert manager._prefetch_futures["next"] is retained_prefetch
        manager.cancel_receive_kv("slow")
        release.set()
        manager._receive_futures["slow"].result(timeout=1)
        finished, failed, cancelled = manager.poll_legacy_kv()
        assert not finished and not failed and cancelled == ["slow"]
        assert "slow" not in manager._ready_legacy_kv
    finally:
        release.set()
        manager.close()


@dataclass
class _Event:
    ready: bool = False

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True


def test_cancelled_h2d_keeps_source_until_copy_finishes(mocker):
    manager = _manager()
    manager._connector = mocker.Mock()
    event = _Event()
    source = torch.ones(2)
    payload = _LegacyKVPayload({}, event, [source])
    future: Future[_LegacyKVPayload] = Future()
    future.set_result(payload)
    manager._receive_futures["A"] = future
    manager._receive_cancels["A"] = threading.Event()
    assert manager.poll_legacy_kv() == ([], {}, [])
    manager.cancel_receive_kv("A")
    assert manager.poll_legacy_kv() == ([], {}, [])
    assert payload.sources[0] is source
    event.ready = True
    assert manager.poll_legacy_kv() == ([], {}, ["A"])
    assert not payload.sources and "A" not in manager._ready_legacy_kv
    manager.close()


@pytest.mark.parametrize("failed", [False, True])
def test_shm_two_readers_and_aggregated_cleanup(failed):
    key = f"kv-bubble-{uuid.uuid4().hex}"
    writer = SharedMemoryConnector({})
    readers = [SharedMemoryConnector({}, unlink_on_get=False) for _ in range(2)]
    try:
        assert writer.put("ar", "dit", key, {"value": 3})[0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            # A publication lock may make a simultaneous try-get return None;
            # both readers can retry without consuming the other reader's key.
            futures = [pool.submit(reader.get, "ar", "dit", key) for reader in readers]
            values = [future.result(timeout=1) for future in futures]
        for index, value in enumerate(values):
            if value is None:
                value = readers[index].get("ar", "dit", key)
            assert value[0] == {"value": 3}
        segment = shared_memory.SharedMemory(key)
        segment.close()
        aggregator = _KVOutputAggregator(2)
        reports = [BatchRunnerOutput([], kv_shm_keys={"A": [key]}, kv_finished_req_ids=["A"]) for _ in range(2)]
        if failed:
            reports[0].kv_failed_req_ids = {"A": "timeout"}
            assert aggregator.record_failures(reports[0]) == {"A": "timeout"}
        else:
            assert not aggregator.aggregate([reports[0], BatchRunnerOutput([])], set()).kv_finished_req_ids
            segment = shared_memory.SharedMemory(key)
            segment.close()
        result = aggregator.aggregate(reports, set())
        assert result.kv_finished_req_ids == ([] if failed else ["A"])
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(key)
        from pathlib import Path

        assert not Path(f"/dev/shm/shm_{key}_lockfile.lock").exists()
    finally:
        writer.cleanup(key)
        writer.close()
        for reader in readers:
            reader.close()


@pytest.fixture
def pump(monkeypatch, mocker):
    from tests.diffusion.test_step_pp_result_pump import _Queue

    queues = [_Queue(), _Queue()]
    broadcast = mocker.Mock()
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_init_broadcast_queue", lambda *_: broadcast)
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_launch_workers", lambda *_: ([], [0, 1]))
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_init_result_queue", lambda _, index: queues[index])
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_start_worker_monitor", lambda _: None)
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = OmniDiffusionConfig.from_kwargs(
        num_gpus=2,
        step_execution=True,
        parallel_config=DiffusionParallelConfig(pipeline_parallel_size=2),
    )
    executor._init_executor()
    try:
        yield executor, queues, broadcast
    finally:
        executor.shutdown()


def test_fast_rank_can_submit_next_step_before_slow_rank_ack(pump):
    executor, queues, broadcast = pump
    first = executor.submit_step(_step(kv_poll_only=True))
    first_roles = broadcast.enqueue.call_args.args[0]["reply_roles"]
    queues[0].queue.put(RpcReply(first_roles["entry"]))
    assert first.pp0_done.result(timeout=1) is None
    queues[0].queue.put(RpcReply(first_roles["result"], BatchRunnerOutput([]), kv_rank=0))
    assert not first.step_done.done()
    second = executor.submit_step(_step(kv_poll_only=True))
    second_roles = broadcast.enqueue.call_args.args[0]["reply_roles"]
    queues[0].queue.put(RpcReply(second_roles["entry"]))
    assert second.pp0_done.result(timeout=1) is None
    queues[0].queue.put(RpcReply(second_roles["result"], BatchRunnerOutput([]), kv_rank=0))
    queues[1].queue.put(RpcReply(first_roles["result"], BatchRunnerOutput([]), kv_rank=1))
    assert first.step_done.result(timeout=1).runner_outputs == []
    assert not second.step_done.done()
    queues[1].queue.put(RpcReply(second_roles["result"], BatchRunnerOutput([]), kv_rank=1))
    second.step_done.result(timeout=1)
    assert not executor._kv_step_reports


def test_single_rank_failure_callback_does_not_wait_for_slow_rank(pump):
    executor, queues, broadcast = pump
    notified = threading.Event()
    failures = {}

    def failed(errors):
        failures.update(errors)
        notified.set()

    executor.kv_failure_callback = failed
    futures = executor.submit_step(_step(kv_poll_only=True))
    roles = broadcast.enqueue.call_args.args[0]["reply_roles"]
    queues[0].queue.put(RpcReply(roles["result"], BatchRunnerOutput([], kv_failed_req_ids={"A": "timeout"}), kv_rank=0))
    assert notified.wait(1)
    assert failures == {"A": "timeout"}
    assert not futures.step_done.done()
    queues[1].queue.put(RpcReply(roles["result"], BatchRunnerOutput([]), kv_rank=1))
    assert futures.step_done.result(timeout=1).kv_failed_req_ids == failures


def test_missing_rank_ack_has_deadline(pump, monkeypatch):
    executor, _, _ = pump
    monkeypatch.setattr("vllm_omni.diffusion.executor.multiproc_executor._DLO_DP_WAVE_TIMEOUT_S", 0.02)
    futures = executor.submit_step(_step(kv_poll_only=True))
    with pytest.raises(TimeoutError, match="Missing diffusion KV step ACK"):
        futures.step_done.result(timeout=1)
    with pytest.raises(TimeoutError):
        futures.pp0_done.result(timeout=1)


def test_non_result_worker_returns_tensor_free_shell(monkeypatch, mocker):
    import vllm_omni.diffusion.distributed.parallel_state as parallel_state

    worker = object.__new__(WorkerProc)
    worker.gpu_id = 0
    worker.result_mq = object()
    worker._async_output_queue = None
    worker.worker = mocker.Mock()
    worker.worker.execute_method.return_value = BatchRunnerOutput(
        [RunnerOutput("A", result=mocker.sentinel.tensor_payload)],
        kv_finished_req_ids=["B"],
    )
    worker._is_stage_leader = lambda: True
    replies: list[RpcReply] = []
    worker._enqueue_result = replies.append
    monkeypatch.setattr(parallel_state, "get_pipeline_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_pipeline_parallel_world_size", lambda: 2)
    worker._execute_role_rpc(
        {
            "method": "execute_stepwise",
            "args": (_step(kv_poll_only=True),),
            "exec_all_ranks": True,
            "reply_roles": {"entry": "e", "result": "r"},
        }
    )
    assert len(replies) == 2
    assert replies[0].rpc_id == "e"
    assert replies[1].kv_rank == 0
    assert replies[1].result.runner_outputs == []
    assert replies[1].result.kv_finished_req_ids == ["B"]
    assert worker.worker.execute_method.call_args.args[0] == "execute_stepwise"


def test_worker_retains_pool_if_connector_shutdown_fails(mocker):
    worker = object.__new__(DiffusionWorker)
    runner = mocker.Mock(spec=DiffusionModelRunner)
    runner.kv_transfer_manager = None
    runner.offload_backend = None
    worker.model_runner = runner
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.shutdown_kv_connector", side_effect=RuntimeError("not stopped")
    )
    with pytest.raises(RuntimeError, match="not stopped"):
        worker.shutdown()
    assert worker.model_runner is runner
    assert not worker._shutdown_complete
