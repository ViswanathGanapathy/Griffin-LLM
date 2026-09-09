# SMPNN-Griffin Multi-Seed Collaborator Handoff

Quickstart for someone joining the project to run the multi-seed
validation of the 3 decisive cross-task wins.

## TL;DR — your job

Run 2 new seeds (43 and 44) of the 3 anchor backbones (vanilla-4,
SMPNN-6 default, SMPNN-6 α=1e-2) on relevant source families, then
evaluate them on the 3 decisive cross-task cells. The goal is to put
mean ± std error bars on the headline numbers (we currently have
single-seed results only).

Expected output: a CSV with each of the 3 decisive cells reported as
mean ± std over n=3 seeds (existing 42 + your 43 + your 44).

## Background — what the project has shown so far

The 24-cell cross-task transfer matrix (4 directions × 3 anchors × 2
ICL heads, all single seed) showed:

- **Vanilla Griffin has zero decisive cross-task wins.** All 3 of its
  nominal wins are within ±0.015 of the runner-up (within noise).
- **SMPNN variants hold all 3 decisive wins (Δ > 0.02):**
  - V5 (SMPNN-6 α=1e-2, "D3"): wins o1→o2 TabPFN by +0.026 and
    c2→c1 TabICL by +0.018
  - V2 (SMPNN-6 α=1e-6, "D1", paper default): wins c1→c2 TabICL by
    +0.050 (strongest single signal)

Your multi-seed work validates whether these wins survive seed noise.

See [SMPNN_GRIFFIN_DESIGN.md](SMPNN_GRIFFIN_DESIGN.md) §3 for the full
matrix and [PRESENTATION_TABLES.md](PRESENTATION_TABLES.md) for the
per-task numbers.

## Hardware requirements

- 1× A100 80GB GPU minimum. 3× A100 80GB ideal for parallel runs.
- ~200 GB disk for the dataset + new checkpoints.
- Conda environment ~25 GB.

## Environment setup

```bash
# Conda env named "griffin"
conda create -n griffin python=3.11 -y
conda activate griffin

# Core dependencies
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric accelerate transformers
pip install tabpfn==8.0.3
pip install tabicl  # NB: classifier + regressor v2 weights download on first use
pip install safetensors pandas seaborn matplotlib tensorboard
```

If you hit a TabPFN auth license issue, follow the prompts to register
a free token — TabPFN v3 requires it.

## Get the code

```bash
git clone https://github.com/ViswanathGanapathy/Griffin-LLM.git Griffin
cd Griffin
git checkout smpnn-ablations
```

(If the repo is private, you'll need to be added as a collaborator
first — see "Access" section at the bottom.)

## Get the data and existing seed-42 checkpoints

Two things you need that aren't in the git repo:

1. **`datasets/joint-v65/`** — the RelBench-v0.65 dataset bundle.
   This is ~5 GB and is *not* committed to git. Coordinate with the
   project lead to obtain it (typically via Google Drive, S3, or
   direct file transfer from their RunPod).

2. **Existing seed-42 checkpoints**, used to compute the n=3 seed
   averages. The multi-seed eval script reuses these:
   ```
   checkpoints/smpnn-depth-vanilla-4/best_checkpoint/
   checkpoints/smpnn-ablation-d3-alpha-1e-2/best_checkpoint/
   checkpoints/smpnn-xtask-c1-c1-vanilla-4/best_checkpoint/
   checkpoints/smpnn-xtask-c1-d1-smpnn-6/best_checkpoint/
   checkpoints/smpnn-xtask-c2-c1-vanilla-4/best_checkpoint/
   checkpoints/smpnn-xtask-c2-d3-alpha-1e-2/best_checkpoint/
   ```
   Each is ~50 MB. Project lead can rsync or zip these.

Confirm both are in place before running anything:

```bash
ls datasets/joint-v65 | head        # should show many .pt files
ls checkpoints/smpnn-depth-vanilla-4/best_checkpoint/  # should show model.safetensors
```

## Run the multi-seed backbone training

```bash
# Default SCOPE=decisive: trains 14 backbones (7 source-anchor combos × 2 seeds)
CUDA_VISIBLE_DEVICES=0 ./run_smpnn_multiseed_backbones.sh \
    2>&1 | tee logs/smpnn-multiseed-backbones-master.log

# Wall time: ~42-56 GPU-hours on 1× A100.
# Each cell tees to its own log at logs/smpnn-multiseed/<tag>.log
# Skip-if-checkpoint-exists; FORCE=1 to override.
```

### Parallel-GPU acceleration (if you have ≥2 GPUs)

```bash
# Split by seed across 2 GPUs (cuts wall time in half)
SEEDS=43 CUDA_VISIBLE_DEVICES=0 ./run_smpnn_multiseed_backbones.sh \
    2>&1 | tee logs/multiseed-s43.log &
SEEDS=44 CUDA_VISIBLE_DEVICES=1 ./run_smpnn_multiseed_backbones.sh \
    2>&1 | tee logs/multiseed-s44.log &
wait
```

3-GPU and source-family-split strategies are documented in the script
comments (look for "Strategy B/C" in the docstring at the top).

### Monitor progress

```bash
# How many backbones have completed?
ls -d checkpoints/smpnn-multiseed-s*/best_checkpoint 2>/dev/null | wc -l
# Should grow from 0 to 14 over time.

# Watch the live log
tail -f logs/smpnn-multiseed-backbones-master.log
```

## Run the multi-seed eval

After Phase 1 finishes (or as soon as the relevant backbones land):

```bash
CUDA_VISIBLE_DEVICES=0 ./run_smpnn_multiseed_eval.sh \
    2>&1 | tee logs/smpnn-multiseed-eval-master.log

# Wall time: ~1.5 GPU-hours for all 18 evals.
# Each cell tees to its own log at logs/smpnn-multiseed-eval/<tag>.log
# Skip-if-complete via per-cell log line count.
```

## Aggregate and send results back

```bash
# Collect everything into one CSV
python collect_smpnn_results.py \
    --logs logs/smpnn-multiseed-eval/*.log \
    --out smpnn_multiseed_results.csv

# The CSV has one row per (direction, backbone, head, seed) with per-task numbers
head smpnn_multiseed_results.csv
```

Send the CSV back to the project lead. They will compute mean ± std
per (direction, backbone, head) across the 3 seeds and decide whether
the SMPNN wins survive the multi-seed bar.

## Possible issues

| Issue | Cause / fix |
|---|---|
| OOM mid-training | Drop `--batchsize 256` to 128 in the script. SMPNN-6 should fit in 80 GB. |
| OOM during eval | Drop `--icl_max_context 30000` (TabPFN) to 20000. |
| `TabPFNValidationError: All features are constant` on avito tasks | Should NOT happen with the new SMPNN backbones — was specific to old `o1-tth-lora-v3`. Flag it if seen. |
| Slow data loading | Bump `--num_workers` in the DataLoader (script default is 16). |
| Stray process on shared GPU | `nvidia-smi` to ID; `kill -9 <PID>` to clear. |

## Access — how I get this repo

If the repo is **public**: just clone the URL above.

If the repo is **private**, you need to be added as a collaborator:

1. You give the project lead your GitHub username (`<your-username>`)
2. Project lead adds you via Settings → Collaborators → Add people
3. You'll get an email; accept the invitation
4. Then `git clone` works as above

For push access (you'll want this to commit your multi-seed results
or any fixes you find), the same collaborator add gives you push.

## Send back

When done, push your local results back to the branch:

```bash
git add smpnn_multiseed_results.csv logs/smpnn-multiseed/  logs/smpnn-multiseed-eval/
git commit -m "Multi-seed results for decisive cross-task cells (seeds 43, 44)"
git push github smpnn-ablations
```

Or just email the CSV if push access wasn't granted.
