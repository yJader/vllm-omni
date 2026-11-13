# Async Diffusion Output

## Table of Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Architecture](#architecture)
4. [Model Coverage](#model-coverage)
5. [Configuration](#configuration)
6. [Related Files](#related-files)

## Overview

The async diffusion output feature moves the D2H (Device→Host) copy and SHM packing from the Worker main thread to a background daemon thread, so the GPU default stream can immediately start the next request's forward pass. This eliminates the GPU bubble caused by synchronous D2H/packing.

**Core benefit**: the previous request's output D2H/packing overlaps with the next request's forward pass.

Enabled automatically for request mode and eligible step-mode PP batches; see [Configuration](#configuration).

## Performance

### HunyuanImage-3.0 (TP4)

| Resolution | Async Off (QPS) | Async On (QPS) | Change |
| --- | --- | --- | --- |
| 1024×1024 | 0.4773 | 0.4802 | +0.60% |
| 768×768 | 0.8370 | 0.8533 | +1.95% |

**Analysis**: smaller resolution → larger QPS gain. D2H/packing is relatively fixed overhead; forward is variable (scales with resolution). Smaller images → shorter forward → D2H/packing occupies a larger fraction of total time → more benefit from overlapping it.

## Architecture

### Execution Timeline

```text
Before (synchronous D2H):
[forward req1] [D2H+SHM pack req1] [forward req2] [D2H+SHM pack req2]
                                 ^^^^^^^^^^^^^^^^ GPU bubble

After (async D2H):
[forward req1] [forward req2]
[D2H+SHM pack req1 (side stream)] [D2H+SHM pack req2 (side stream)]
                                 ^^^^^^^^^^^^^^^^ bubble eliminated
```

### Data Flow

```text
                    ┌─────────────────────────────────────────────────────┐
                    │                 Worker Process                      │
                    │                                                    │
  execute_model ──> │ WorkerBusyLoop ──> forward() ──> DiffusionOutput   │
                    │       │                              │ (GPU tensor)│
                    │       │                    ┌─────────┴──────────┐  │
                    │       │                    │  async output path │  │
                    │       │                    │                    │  │
                    │       │              compute_done ──> result_mq │  │
                    │       │                    │                    │  │
                    │       │                    ▼                    │  │
                    │       │            AsyncOutputThread            │  │
                    │       │         (side CUDA stream)             │  │
                    │       │         wait_event(gpu_event)          │  │
                    │       │         pack_diffusion_output_shm()    │  │
                    │       │         D2H + SHM write                │  │
                    │       │                    │                    │  │
                    │       │              output_ready ──> result_mq │  │
                    │       │                    │                    │  │
                    │       ▼                    ▼                    │  │
                    │  (dequeue next request)                        │  │
                    └─────────────────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────────────┐
                    │              Executor (main process)               │
                    │                                                    │
                    │  ResultPumpThread (sole reader of result_mq)       │
                    │       │                                            │
                    │       ├── AsyncDiffusionOutput(COMPUTE_DONE)       │
                    │       │   → resolve _rpc_futures[rpc_id]           │
                    │       │   → RunnerOutput(async_output_id=...)      │
                    │       │                                            │
                    │       ├── AsyncDiffusionOutput(OUTPUT_READY)       │
                    │       │   → resolve _output_futures[async_output_id]│
                    │       │   → or batch split via _batch_split_map   │
                    │       │                                            │
                    │       └── Non-async message                        │
                    │           → _sync_result_buffer (for other RPCs)   │
                    │                                                    │
                    │  wait_output_ready(async_output_id) → Future       │
                    └─────────────────────────────────────────────────────┘
```

### Key Components

1. **AsyncDiffusionOutput** (`data.py`): Protocol envelope for `result_mq`. `kind` field routes messages:
   - `COMPUTE_DONE` — forward finished, GPU can start next request
   - `OUTPUT_READY` — D2H/SHM packing finished, final output available
   - `RPC_RESULT` — ordinary RPC return (including error propagation)

2. **Worker AsyncOutputThread** (`diffusion_worker.py`): Background daemon thread that:
   - Waits for `gpu_event` (cross-stream sync ensuring forward wrote the tensor)
   - Runs `pack_diffusion_output_shm()` on a side CUDA stream
   - Enqueues `OUTPUT_READY` to `result_mq` when packing completes in request-mode
   - Publishes `RpcReply` for [step-mode PP](#step-mode-pp-output)

3. **ResultPumpThread** (`multiproc_executor.py`): Sole reader of `result_mq` when enabled. Dispatches messages:
   - `COMPUTE_DONE` / `RPC_RESULT` → resolves `_rpc_futures[rpc_id]`
   - `OUTPUT_READY` → resolves `_output_futures[async_output_id]` (with batch split via `_batch_split_map`)
   - `RpcReply` → resolves the corresponding step-PP role Future
   - Other messages → `_sync_result_buffer` for other RPCs

4. **collective_rpc() two-path dispatch** (`multiproc_executor.py`):
   - **Path 1**: `execute_model` / `execute_model_batch` → generates `rpc_id`, registers Future, waits for pump to deliver `compute_done`
   - **Path 2**: All other RPCs → `_sync_result_buffer` when the pump is enabled, otherwise `_result_mq`

5. **record_device_event** (`platforms/`): Cross-platform GPU event recording for side-stream synchronization. Implemented for CUDA, NPU, ROCm, XPU, MUSA; base class returns `None` (safe no-op).

6. **IPC side-stream D2H** (`ipc.py`): `pack_diffusion_output_shm()` and helpers accept `d2h_stream` parameter; use `pin_memory` + `copy_(non_blocking=True)` on the side stream instead of synchronous `.cpu()`.

### Batch Split

When `execute_model_batch` returns a `COMPUTE_DONE` with a single `async_output_id` for the entire batch, the executor splits it into per-request `async_output_id`s (format `{batch_id}/{request_id}`) via `_batch_split_map`. When `OUTPUT_READY` arrives, the pump extracts per-request results from the batch output and resolves each request's output future independently. If `OUTPUT_READY` arrives before `execute_batch` registers the split map, `execute_batch` directly adopts the early output.

### Step-mode PP Output

For batches eligible under [Configuration](#configuration), the last-stage leader records a device event and queues
its original `RpcReply`. The background thread waits for the event, packs the output, and publishes the reply;
packing failures return an error for the same RPC ID. The result pump resolves `step_done` only after packing completes,
while the worker can execute an already submitted batch.

`post_decode()`, stage-payload transfer, and PP send/receive drains stay on the execution thread. Before handing off
the output, the runner retires terminal states and invalidates their cached input batch. The background task retains
tensor storage until packing completes. Streaming and mixed finished/unfinished batches keep synchronous transport
because live request states may still own mutable output storage.

### Reliability, Lifecycle & Timeout Behavior

1. **Drain Before Memory Release (`drain_async_outputs` / `_ASYNC_OUTPUT_DRAIN_TIMEOUT_S`)**:
   When workers transition into sleep states (`handle_sleep_task`) or invoke memory-releasing methods, `WorkerProc.drain_async_outputs()` blocks up to `_ASYNC_OUTPUT_DRAIN_TIMEOUT_S` (10.0s) until all in-flight background D2H/SHM packing tasks finish. This acts as a synchronization barrier preventing worker sleep cycles from releasing device tensors while the side CUDA stream is still actively reading them. Draining typically completes within 10–50 ms under normal operation.

2. **Timeout Diagnostics (`_async_output_timeout()`)**:
   The engine waits for `wait_output_ready()` with a timeout of `_async_output_timeout()` — 600.0s by default, overridable per run with the `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` environment variable (a non-numeric or non-positive value is ignored with a warning and the default applies, since this resolves on the request path). If a timeout occurs, the executor logs detailed diagnostic bookkeeping (pending futures, cached outputs, and batch split maps) to identify whether the stall occurred in the worker, pump, or waiter. The packing itself takes milliseconds, but the wait is queued behind the step's GPU work, so its wall-clock duration tracks step time; the bound is generous because a hung engine is surfaced by the worker monitor and `check_health()`, not by this timeout.

3. **Queue Serialization & Atomic Future Resolution**:
   - `result_mq` writes are serialized with `_result_mq_lock` to protect the single-writer `MessageQueue` against concurrent writes from the worker main loop (`COMPUTE_DONE`) and background thread (`OUTPUT_READY`).
   - `ResultPumpThread` unpacks shared-memory payloads first, and then atomically resolves waiting futures or populates `_completed_outputs` under `_futures_lock`, avoiding race conditions with concurrent `wait_output_ready()` invocations.

4. **Storage-Aware IPC Packing**:
   `pack_diffusion_output_shm()` evaluates `max(view_bytes, storage_bytes)` against `_SHM_TENSOR_THRESHOLD` (1 MB) to ensure that per-request slices sharing a large batch storage are moved to POSIX shared memory rather than serialized through pickle over the MessageQueue inline buffer. When a D2H side stream is supplied, device tensors below that threshold also use SHM so MessageQueue serialization cannot trigger a copy on the default stream. Small CPU tensors keep their inline transport.

## Model Coverage

### Currently Supported (request-mode, `step_execution=False`)

All models running in request-mode (`step_execution=False`, the default) automatically use async output.

**Verified models**:

| Model | Type | `supports_request_batch` |
| --- | --- | --- |
| **HunyuanImage-3.0** | Image | `False` |
| **Qwen-Image** | Image | `True` |
| **LTX-2.3** | Video / Audio | `False` |

Other models with `step_execution=False` are also supported but not yet verified.

### Step-mode PP

This path has CPU coverage for RPC progress during packing, output equality, error delivery and draining;
GPU overlap and throughput require workload-specific validation.

## Configuration

No additional configuration is required.

| Execution mode | Background output packing | Result pump | Worker background thread |
| --- | --- | --- | --- |
| Request mode (`step_execution=False`) | All outputs | Started | Started |
| Step PP (`pipeline_parallel_size > 1`) | Non-streaming, all-terminal batches | Started for PP role replies | Started |
| Other step execution | Synchronous transport | Existing synchronous path | Not started |

## Related Files

- `vllm_omni/diffusion/data.py`: `AsyncDiffusionOutput`, `AsyncOutputKind`, `DiffusionOutput.async_output_id`
- `vllm_omni/diffusion/worker/diffusion_worker.py`: `WorkerProc._return_result()`, `_async_output_loop()`, `_generate_async_output_id()`
- `vllm_omni/diffusion/executor/multiproc_executor.py`: `ResultPumpThread`, `collective_rpc()` two-path dispatch, `wait_output_ready()`, `_batch_split_map`
- `vllm_omni/diffusion/diffusion_engine.py`: `step()` / `step_streaming()` / `add_req_and_wait_for_response()` async output waiting
- `vllm_omni/diffusion/sched/request_scheduler.py`: `update_from_output()` — `async_output_id` → `FINISHED_COMPLETED`
- `vllm_omni/diffusion/ipc.py`: `pack_diffusion_output_shm()` side-stream D2H path
- `vllm_omni/platforms/interface.py`: `OmniPlatform.record_device_event()` base class
- `vllm_omni/platforms/cuda/platform.py`: `CudaOmniPlatform.record_device_event()`
- `vllm_omni/platforms/npu/platform.py`: `NPUOmniPlatform.record_device_event()`
- `vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py`: `get_hunyuan_image3_post_process_func()`, postprocess removal
- `vllm_omni/diffusion/registry.py`: `_DIFFUSION_POST_PROCESS_FUNCS` postprocess registry
