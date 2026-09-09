# Running the LoG Rebuttal Ablations on RunPod

**Date:** 2026-09-07. Companion to `run_log_rebuttal_ablations.sh` /
`analyze_rebuttal_ablations.py` (repo root). Goal: fill the two Table-1
N/A cells (V6 on o2→o1 both heads; SMPNN-6 α=1e-6 on o1→o2 TabPFN),
seeds 42–46, identically to the paper's existing cells.

## 0. What moves where

| thing | size | how |
|---|---|---|
| code | — | `git clone` the fork, branch `smpnn-ablations` |
| dataset | **18 GB** subset (not the 90 GB corpus) | `make_o1o2_subset.py --tar` locally, then `runpodctl send` |
| existing checkpoints (optional, saves ~15–20 GPU-h) | ~0.2 GB each | copy any `smpnn-multiseed-s*-others-1-d1-smpnn-6` and V6-on-o2 checkpoints from the old training box |
| results back | ~MBs | `scp` the eval logs; analysis runs anywhere |

## 1. Rent the pod

- **GPU**: 1× A100 80 GB (SXM) or H100 PCIe. The paper used H100-class;
  A100 is fine and cheaper (~$1.6–2.2/h vs ~$2.7–3.5/h) — training is
  ~3–4 h/run on A100, ~2–3 h on H100.
- **Template**: RunPod PyTorch (e.g. `runpod/pytorch:2.x`-cuda12.x); any
  CUDA ≥ 12.1 image works, we pip-install everything anyway.
- **Volume**: 100 GB persistent volume mounted at `/workspace`
  (18 GB data + per-epoch checkpoints ≈ 3–4 GB/run × up to 10 runs).
- **Budget envelope**: evals only ≈ 2 GPU-h (~$5); worst case
  (all 10 trainings) ≈ 35–40 GPU-h (~$70–90 on A100).

Enable SSH (add your public key in RunPod settings) and note the pod's
SSH command from the console.

## 2. Ship the dataset (do this first — it's the long pole)

On the WSL2 box:

```bash
cd ~/Griffin
python make_o1o2_subset.py --out /tmp/joint-v65-o1o2 --tar
# -> /tmp/joint-v65-o1o2.tar (~18 GB; validated: Graph loads 65 types)

# Option A: runpodctl (simplest; relay transfer, resumable)
wget -qO runpodctl https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 && chmod +x runpodctl
./runpodctl send /tmp/joint-v65-o1o2.tar    # prints a one-time code
# on the pod:  runpodctl receive <code>

# Option B: rsync over the pod's SSH (resumable, shows progress)
rsync -avP -e "ssh -p <PORT>" /tmp/joint-v65-o1o2.tar root@<POD_IP>:/workspace/
```

If you have the old training box's checkpoints, send these too (each
saves one 3–4 h training):
`checkpoints/smpnn-multiseed-s{42..46}-others-1-d1-smpnn-6/`,
`checkpoints/smpnn-ablation-d1-smpnn-6/` (seed 42 legacy), and any
V6-on-others-2 checkpoints from the in-progress o2→o1 depth sweep.

## 3. Set up the pod

```bash
cd /workspace
git clone https://github.com/ViswanathGanapathy/Griffin-LLM.git Griffin
cd Griffin && git checkout smpnn-ablations

pip install datasets pqdm accelerate evaluate sentence_transformers \
            einops torchmetrics seaborn transformers scipy pyyaml
pip install tabpfn "tabicl>=2.1.1"   # TabPFN v3 + TabICL v2 checkpoint support

mkdir -p datasets logs
tar -xf /workspace/joint-v65-o1o2.tar -C datasets/
mv datasets/joint-v65-o1o2 datasets/joint-v65

# checkpoints from the old box, if transferred:
mkdir -p checkpoints && cp -r /workspace/old-checkpoints/* checkpoints/ 2>/dev/null

accelerate config default          # single-GPU, no distributed
python - <<'EOF'                   # smoke: loader + one batch of graph access
from hdataset import Graph, Task
g = Graph("datasets/joint-v65"); Task("datasets/joint-v65")
print("OK:", len(g.nodes), "node types")
EOF
```

First eval will download the TabPFN v3 and TabICL v2 weights from
HuggingFace automatically (pod has internet; ~1–2 GB, cached).

## 4. Run

Always inside `tmux` (pod SSH sessions drop):

```bash
tmux new -s ablations
cd /workspace/Griffin
./run_log_rebuttal_ablations.sh 2>&1 | tee logs/log-rebuttal.log
```

Useful variants:

```bash
PHASE=train ./run_log_rebuttal_ablations.sh          # trainings first
PHASE=eval  ./run_log_rebuttal_ablations.sh          # then evals
ABLATION=1  ./run_log_rebuttal_ablations.sh          # V6 o2->o1 only
SEEDS="42 43 44" ... & SEEDS="45 46" ...             # split across 2 pods
FORCE=1 ...                                          # re-run finished cells
```

The script skips anything whose checkpoint/eval-log already exists, so
it is safe to re-run after any interruption. Watch progress with
`tail -f logs/smpnn-multiseed/*.log logs/log-rebuttal-eval/*.log`.

Completion check (do not trust silence): each eval log must contain
6 `test_metric/` lines; `grep -c "test_metric/" logs/log-rebuttal-eval/*.log`.

## 5. Numbers + teardown

```bash
python analyze_rebuttal_ablations.py                 # on the pod, or:
scp -P <PORT> root@<POD_IP>:/workspace/Griffin/logs/log-rebuttal-eval/*.log logs/log-rebuttal-eval/
python analyze_rebuttal_ablations.py                 # locally (needs scipy only)
```

The analyzer prints per-seed values, mean ± σ, and Welch/paired tests
against the paper's published baselines (it reproduces all 8 published
test statistics exactly — validated). Copy the checkpoints you want to
keep off the volume (`runpodctl send` from the pod), then **terminate
the pod**; keep or release the volume depending on whether reruns are
expected.

## Gotchas

- The subset corpus **cannot run commerce directions** — it contains
  only others-1/others-2. Ship the full corpus if scope grows.
- `--tabpfn_version v3` requires a recent `tabpfn` pip package; if the
  eval errors on the flag, `pip install -U tabpfn`.
- Per-epoch checkpoints accumulate (~3–4 GB/training). If the volume
  fills, delete `checkpoints/smpnn-multiseed-*/checkpoint-*` dirs,
  keeping `best_checkpoint/`.
- Head versions are pinned in the script (TabPFN v3 / TabICL v2 ctx
  10000) to match the paper's completed cells — don't "upgrade" them.
