# Sprint v1 — Walkthrough

## Summary

Sprint v1 fixed five correctness bugs in `hmaintask_combine_llm.py` (the Rel-LLM–style Griffin+LLM pipeline), then expanded well beyond the original PRD scope into a full experimental arc: cross-dataset transfer, task-type-aware decoding, a Griffin-style unified decoder, few-shot per-task fine-tuning, and analysis tooling. The pipeline is now experiment-ready with three decoder strategies (Single-Head, TaskTypeHeads, UnifiedDecoder) and working transfer runs on both `others-1↔others-2` and `commerce-1↔commerce-2`.

## Architecture Overview

```
                         ┌─────────────────────────────────────────┐
                         │         Relational DB (RelBench)         │
                         │   datasets/joint-v65 (multi-dataset)     │
                         └──────────────────┬───────────────────────┘
                                            │
                          construct_dataset (per task, with neighbors)
                                            │
                                            ▼
         ┌──────────────────── LoaderWrapperTaskLLM ───────────────────┐
         │   yields (subgraph, seed indices, y, prompts, neighbor IDs) │
         └──────────────────────────────┬──────────────────────────────┘
                                        │
                                        ▼
                          ┌────────────────────────────┐
                          │   GriffinMod MPNN (hmodel) │  [11M params]
                          └─────────────┬──────────────┘
                                        │ seed + neighbor embeddings  [B, 512]
                                        ▼
                  ┌────────────────────────────────────────┐
                  │   GriffinToLLMProjector (GELU MLP)     │  [2.6M params]
                  │   512 → 1024 → llm_dim                  │
                  └─────────────┬──────────────────────────┘
                                │ soft tokens [B, 1, llm_dim] + neighbor tokens [B, K, llm_dim]
                                ▼
                  ┌────────────────────────────────────────┐
                  │   Text Prompt (task_prompts.py)        │
                  │   system + user + entity row + question │
                  └─────────────┬──────────────────────────┘
                                │ token embeddings
                                ▼
              ┌─────────────────────────────────────────────────┐
              │   Frozen LLM (Qwen3-1.7B / Llama / Gemma)        │  [2B params]
              │   hidden_states: [B, seq_len, llm_dim]           │
              └────────────────────┬────────────────────────────┘
                                   │ _pool_llm_hidden (last / entity / attention)
                                   ▼
                    ┌─────────── Decoder (3 flavors) ─────────┐
                    │                                          │
                    │  A) OutputMLP (per-task ModuleDict)       │
                    │  B) TaskTypeHeads (reg / bin / multi)     │
                    │  C) UnifiedDecoder (back_proj + dec / y.T)│
                    └───────────────────┬──────────────────────┘
                                        │
                                        ▼
                                  prediction, loss
```

## Files Created/Modified

### `hmaintask_combine_llm.py` (primary pipeline, ~2400 lines)

**Purpose**: End-to-end Griffin→Projector→LLM→Decoder training & evaluation, covering pretraining, transfer, and per-task fine-tuning.

**Key components**:

- `GriffinToLLMProjector` — 512→bottleneck→llm_dim MLP, **GELU** (was Sigmoid — Task 3). Used to convert Griffin's graph embedding into a soft LLM token.
- `OutputMLP` — per-task regression/classification head. Wrapped in `nn.ModuleDict` keyed by task name so mixed multi-task runs auto-size `out_channels` from metadata (Task 4). Dispatch uses `isinstance(output_mlp, nn.ModuleDict)` checks.
- `TaskTypeHeads` — three shared heads by task type (regression / binary / multiclass). Includes a **dummy gradient trick** so DDP doesn't complain about unused params: all three heads run each step, the two unused contribute a `0.0 * sum(params)` term to the loss.
- `UnifiedDecoder` — Griffin paper §3.3 decoder. Back-projects LLM hidden state to `griffin_dim=512`, then branches:
  - Regression → `floatdec-512.pt` (pretrained regression decoder)
  - Classification → `h @ y.T` similarity against graph class embeddings (handles unseen-class transfer without learnable heads)
- `LayerPooling`, `AttentionPool`, `_pool_llm_hidden` — three pooling strategies for LLM hidden states: last token, [entity+last] concat, or learned attention.
- `_extract_llm_components` — builds per-sample neighbor token sets. **Fixed** (Task 1) so each sample gets its own 1-hop neighbors via `edge_index`+`mapping`, zero-padded to K.
- Per-task fine-tune loop (~line 1743–1830) — when `--finetune_samples > 0` and `--mode test`:
  1. Freeze LLM, Griffin, and optionally projector.
  2. For each eval task: reload checkpoint head state, fine-tune only on that task's N samples, test only that task.
  3. Supports `--finetune_projector` to also update projector at `lr * 0.1`.
- `_run_icl_evaluation` — now takes pre-loaded graph/task objects (Task 5) instead of re-reading from disk per call.
- Multi-GPU fixes — `avg_valid_metric` initialized before conditional (Task 2), `getattr(output_mlp, 'module', output_mlp)` unwrap when DDP-wrapped.

**CLI flags added this sprint** (beyond the original):
- `--task_type_heads` — use TaskTypeHeads decoder
- `--unified_decoder` — use UnifiedDecoder (pretrained dec + @y.T)
- `--shared_head` — single OutputMLP shared across all tasks
- `--eval_tasks` — different task set for validation/test (cross-dataset transfer)
- `--finetune_samples N`, `--finetune_epochs`, `--finetune_lr`, `--finetune_projector`
- `--pool_mode {last, entity, attention}`, `--pool_layers K`
- `--entity_after_question` / `--no_entity_after_question`
- `--debug`

### `task_prompts.py`

**Purpose**: Builds the text prompt fed to the LLM alongside graph tokens.

**Key change**: `build_rich_system_prompt` now appends 2-hop entity features so the LLM sees richer relational context (not just the seed row). Entity-after-question ordering is controlled by a CLI flag.

### `hmodel.py`, `hdataset.py`, `hloaderwrapper.py`

Unchanged (Sprint v1 was explicitly scoped to avoid modifying these).

### `tests/test_debug_assertions.py` (new)

16 unit tests without GPU/LLM dependency:
- Per-sample neighbor assignment produces distinct embeddings per seed
- Zero-padding when fewer than K neighbors
- `nn.ModuleDict`-based OutputMLP dispatch
- Projector uses GELU (not Sigmoid), allows negative outputs
- `_pool_llm_hidden` in `last`, `entity`, `attention` modes
- `LayerPooling` single-layer identity vs multi-layer weighted combination
- `AttentionPool` shape and masking
- `debug_check_neighbors` detects identical-neighbor collisions and prints diagnostics

### `tests/test_llm_head.py` (new)

13 tests exercising the `--head llm` (text-generation) path for binary, regression, and multiclass tasks — prompt generation, greedy/sampling decoding, logit extraction.

### Accelerate configs

- `hconfig.yaml` — 8-GPU template (original)
- `hconfig_single_gpu.yaml` — single-GPU template
- `hconfig_runpod_2gpu.yaml` (new) — 2-GPU config, port 29503

### Transfer & fine-tune scripts (new)

- `run_finetune_sweep.sh` — sweep N ∈ {0, 512, 1024, …, 4096} on the TaskTypeHeads checkpoint (o1→o2).
- `run_finetune_sweep_unified.sh` — same sweep against the UnifiedDecoder checkpoint.
- `run_finetune_sweep_commerce.sh` — parameterized by direction arg (`c1_to_c2` or `c2_to_c1`); sweeps N = 512..4096 with per-task FT. Designed to run as two parallel single-GPU jobs.
- `run_finetune_study_outcome.sh` — focused 3D grid (N × LR × epochs) for `rel-trial-study-outcome` to close the gap vs Rel-LLM's 0.72 AUROC.

### `parse_sweep_results.py` (new)

Log parser: scans `=== N=X ===` markers and following `test_metric/<task>[/<metric>]: <val>` lines; prints a task × N table plus best-N-per-task and an oracle-average row showing the ceiling if N were tuned per task.

## Data Flow

**Training (pretraining or transfer)**:
1. `hmaintask_combine_llm.py` loads graph + task definitions for `--tasks`.
2. For each batch: Griffin produces seed + neighbor embeddings → Projector converts to LLM-dim tokens → prompt assembled → frozen LLM runs forward → pool final hidden → chosen decoder produces prediction.
3. Loss backprops through Griffin, Projector, and Decoder (LLM frozen).
4. Validation on `--tasks` (or `--eval_tasks` if set) picks best checkpoint by average metric.

**Transfer (cross-dataset)**:
1. `--tasks commerce-1 --eval_tasks commerce-2` (or `others-1 ↔ others-2`).
2. Heads that exist in both task sets are reused; transfer-friendly decoders (TaskTypeHeads, UnifiedDecoder) generalize to unseen tasks without dimension mismatches.
3. Best checkpoint saved to `--savepath`.

**Per-task fine-tuning (test-time adaptation)**:
1. Load transfer checkpoint.
2. For each eval task: restore head/projector state to checkpoint's → sample N training examples of that task → fine-tune for `--finetune_epochs` → evaluate on that task's test split.
3. Log `test_metric/<task>: <value>` per task, `Average test metric: <avg>` at the end.

**Analysis**:
1. `parse_sweep_results.py logs/*.log` → table of task × N with best-N column.

## Experiments — Completed

| # | Direction | Decoder | Best Result | Notes |
|---|-----------|---------|-------------|-------|
| T1 | others-1 → others-2 | Single-Head | avg ≈ -1.5 | Baseline; task competition in shared head |
| T1 | others-1 → others-2 | TaskTypeHeads + 512-shot per-task FT | **avg -0.987** | airbnb-destination AUROC 0.798; best overall |
| T1 | others-1 → others-2 | UnifiedDecoder (single GPU, bf16) | avg -1.44 zero-shot | 2-GPU DDP failed — NCCL timeout from branching code paths |
| T1 | others-1 → others-2 | UnifiedDecoder + per-task FT sweep | **avg -0.89 @ N=2048**; oracle -0.8632 | Per-task FT sweep results below |
| T1 (study-outcome only) | TaskTypeHeads zero-shot | — | **AUROC 0.6307** | Better launchpad than UnifiedDecoder (0.54) for focused sweep |
| T3 | commerce-1 → commerce-2 | UnifiedDecoder (single GPU) | trained; sweep in progress | |
| T4 | commerce-2 → commerce-1 | UnifiedDecoder (single GPU) | trained; sweep in progress | |

### UnifiedDecoder per-task FT sweep on others-2

Fine-tuned `transfer-o1-to-o2-unified/best_checkpoint` per-task across N ∈ {0, 512, …, 4096}. Parsed with `parse_sweep_results.py logs/ft-unified-sweep.log`:

| task (metric) | N=0 | N=512 | N=1024 | N=1536 | N=2048 | N=4096 | best | @N |
|---|---|---|---|---|---|---|---|---|
| airbnb-destination (AUROC) | 0.5565 | 0.8426 | 0.8149 | **0.8438** | 0.8278 | 0.8357 | 0.8438 | 1536 |
| rel-trial-site-success (-MAE) | -1.0461 | -0.9773 | -0.9714 | -0.9657 | -0.9747 | **-0.9552** | -0.9552 | 4096 |
| rel-trial-study-adverse (-MAE) | -2.5146 | -2.6919 | -2.6122 | -2.5341 | -2.3716 | -2.5465 | **-2.2955** | 3072 |
| rel-trial-study-outcome (AUROC) | 0.4865 | 0.5306 | **0.5388** | 0.5016 | 0.4865 | 0.5080 | 0.5388 | 1024 |
| talkingdata-demo-pred (-logloss) | -4.9828 | -2.4849 | -2.4849 | -2.4849 | -2.4849 | -2.4849 | **-2.4841** | 3072 |
| telstra-severity (-logloss) | -0.9347 | -1.0044 | -0.9063 | -0.8715 | **-0.8269** | -0.8282 | -0.8269 | 2048 |
| **AVERAGE** | — | -0.9642 | -0.9368 | -0.9185 | **-0.8906** | -0.9119 | -0.8906 | 2048 |

**Oracle average (best N picked per-task)**: **-0.8632** — only ~3% absolute gain over the best single-N choice (N=2048), so per-task N tuning is not a high-value lever here.

**Observations**:
- `airbnb-destination` saturates at N=512 (0.84+) and barely moves afterward — sample-efficient.
- `talkingdata-demo-pred` collapses from -4.98 (zero-shot) to -2.48 at N=512 and stays there — likely stuck in a local optimum.
- `rel-trial-study-outcome` peaks at 0.5388, still **far below Rel-LLM's 0.72**. Crucially, this ceiling is **lower than the TaskTypeHeads zero-shot (0.6307)** — the UnifiedDecoder's `@y.T` path is the wrong starting point for this task.

### rel-trial-study-outcome — partial focused sweep

Running `run_finetune_study_outcome.sh` from the TaskTypeHeads T1 checkpoint. So far only the N=0 (zero-shot) baseline has been logged:

| N | AUROC |
|---|---|
| 0 (zero-shot, TTH checkpoint) | **0.6307** |
| 512..4096 × LR × epochs × projector | (pending — sweep paused/running) |

The TTH zero-shot is already +0.09 above the UnifiedDecoder's best on this task, confirming the TTH checkpoint is the right launchpad for closing the 0.72 gap.

## Experiments — Planned / Running

- **Commerce FT sweeps in progress**: `run_finetune_sweep_commerce.sh c1_to_c2` on GPU 2, `c2_to_c1` on GPU 3. Per-task FT across N = 512..4096; results pending.
- **rel-trial-study-outcome focused sweep**: 18-point grid (N × LR × epochs) + `--finetune_projector`, started from the TTH checkpoint (zero-shot baseline 0.6307 confirmed). Only N=0 logged so far — sweep needs to be restarted/monitored to close the 0.63 → 0.72 AUROC gap vs Rel-LLM.
- **Full UnifiedDecoder T1 training** (vs. fine-tuning from a TaskTypeHeads checkpoint): would let projector + back-projection co-adapt to the `@y.T` path end-to-end. Motivation reinforced by the observation that UnifiedDecoder FT from a TTH checkpoint caps study-outcome at 0.54 while TTH zero-shot already hits 0.63 — end-to-end UnifiedDecoder training may close that gap or at least clarify where UnifiedDecoder's ceiling actually is.
- **Diagnostic from-scratch training** on `rel-trial-study-outcome` alone — to separate transfer/few-shot overhead from architectural ceiling. If this also caps at ~0.65, the gap to 0.72 is architectural (LLM size, prompt content, or pooling) not few-shot budget.

## Test Coverage

- **Unit**: 29 tests total
  - `tests/test_debug_assertions.py` — 16 tests (sprint v1 bug fixes, pooling, projector)
  - `tests/test_llm_head.py` — 13 tests (text-generation head path, all task types)
- **Integration**: none (running full pipeline requires GPU + LLM weights; validated empirically via transfer runs)
- **E2E**: transfer runs themselves serve as end-to-end validation

## Debug Infrastructure

`--debug` flag enables `debug_check_neighbors()` which prints per-step diagnostics:
- Shape of neighbor_embeds and graph_embeds
- Diversity check (warns if different seeds produced identical neighbor embeddings — the original v1 bug)
- OutputMLP selection info (which key was picked from ModuleDict)
- Zero-padding statistics

## Known Limitations

- **UnifiedDecoder + DDP hangs**: `@y.T` classification uses a y matrix whose shape depends on per-task class count. When ranks in a DDP group draw different tasks in the same step, they take different code paths and launch mismatched NCCL collectives (one rank ALLREDUCE, the other BROADCAST), hanging until the 600 s timeout. Workaround today: run as independent single-GPU jobs on different GPUs. Proper fix: serialize per-task batches across ranks, or compute both regression and classification paths every step (like TaskTypeHeads' dummy-gradient trick).
- **Per-task fine-tune resets only the head + optionally projector**: Griffin MPNN stays at the transfer-time state. If a target task needs a different graph encoding, this limits the ceiling.
- **`parse_sweep_results.py` assumes higher-is-better uniformly**: valid because regression metrics are pre-negated in logs, but any new metric that isn't pre-negated would break the `max()` selection.
- **Per-task FT with `batchsize=1`**: the inner FT loader uses `batch_size=1` for simplicity of loss dispatch — slow for high N. Would benefit from real batching with per-task collate.
- **Sample-count floors**: `--downsample_num` can exceed a task's actual sample count (e.g., `rel-f1-driver-top3` has 2667 samples). Currently caller must know the floor; no auto-cap.
- **Commerce sweeps not yet analyzed**: scripts pushed and running; results pending.

## What's Next (v2 candidates)

1. **Close the study-outcome gap**: restart the focused sweep from the TTH checkpoint (TTH zero-shot 0.6307 > UnifiedDecoder best 0.5388). Confirm whether Rel-LLM's 0.72 AUROC is few-shot or from-scratch. If from-scratch, run the diagnostic single-task full-data training. If few-shot, the focused LR × epochs × projector sweep should narrow the gap.
2. **Full UnifiedDecoder pretraining on T1/T3** — compare against FT-from-TaskTypeHeads to isolate the value of end-to-end `@y.T` training.
3. **Fix UnifiedDecoder DDP**: implement a rank-synchronized batching scheme or always-compute-both-paths pattern so multi-GPU speedup is recoverable.
4. **Per-task LR tuning** inside the FT loop — the sweep oracle-vs-single-N gap will show whether this is worth a harness change.
5. **Prompt ablation** on `rel-trial-*`: verify the 2-hop enrichment is actually helping vs just adding tokens; test adverse-event and sponsor-track features explicitly.
6. **Unified results dashboard**: extend `parse_sweep_results.py` to dump CSV + produce a multi-experiment comparison (T1 TTH vs T1 Unified vs T3 Unified, etc.).
