# Sprint v1 — Tasks

## Status: Complete

### Bug Fixes

- [x] Task 1: Fix per-sample neighbor embedding assignment in `_extract_llm_components` (P0)
  - **Problem**: Lines 560-569 took first K neighbors from entire batch and replicated to every sample.
  - **Fix**: Uses `edge_index` + `mapping` to build per-seed neighbor lists. Each sample gets its own 1-hop neighbors, zero-padded to K.
  - Files: `hmaintask_combine_llm.py` (function `_extract_llm_components`)

- [x] Task 2: Fix `avg_valid_metric` uninitialized variable on non-main processes (P0)
  - **Fix**: Initialize `avg_valid_metric = 0.0` before conditional, use `gathered[0]` instead of `.mean()`.
  - Files: `hmaintask_combine_llm.py` (lines ~1486-1494)

- [x] Task 3: Replace Sigmoid with GELU in `GriffinToLLMProjector` (P1)
  - **Fix**: `nn.Sigmoid()` → `nn.GELU()` in projector Sequential block.
  - Files: `hmaintask_combine_llm.py` (class `GriffinToLLMProjector`)

- [x] Task 4: Auto-detect `output_mlp_dim` from task metadata for classification (P0)
  - **Fix**: Replaced single `OutputMLP` with `nn.ModuleDict` keyed by task name. Regression gets out_channels=1, classification gets num_class from metadata. `compute_loss`/`compute_output` select per-task MLP.
  - Files: `hmaintask_combine_llm.py` (main + compute_loss + compute_output)

- [x] Task 5: Pass graph/task objects to `_run_icl_evaluation` instead of reloading from disk (P1)
  - **Fix**: Added `graph` and `task_obj` parameters; callers pass already-loaded objects.
  - Files: `hmaintask_combine_llm.py` (function `_run_icl_evaluation` + 2 callers)

### Verification

- [x] Task 6: Add lightweight smoke-test assertions for neighbor correctness (P1)
  - **Fix**: Added `debug_check_neighbors()` function + `--debug` CLI flag. When enabled, prints neighbor shape, diversity, zero-padding stats, and OutputMLP selection info. 10 unit tests cover all fixes.
  - Completed: 2026-04-02
  - Files: `hmaintask_combine_llm.py`, `tests/test_debug_assertions.py`

- [x] Task 7: Update CHANGES.md with v1 sprint fixes (P2)
  - Files: `CHANGES.md`
