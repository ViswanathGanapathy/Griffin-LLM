"""
tabular_heads.py — TabPFN v2 and TabICL v2 wrappers for Griffin.

These heads use in-context learning (ICL): Griffin MPNN embeddings are
extracted first, then fed as "tabular features" to the ICL model which
sees labeled train examples as context and predicts on test examples
in a single forward pass — no gradient-based training of the head.

Architecture:

  Griffin MPNN ──→ [B, hiddim] seed embeddings
                        │
                  ICLProjection (trainable)
                        │
                  [B, icl_dim] projected embeddings
                        │
                ┌───────┴────────┐
                ▼                ▼
          TabPFN v2         TabICL v2
       (ICL Transformer)  (3-stage ICL)
                │                │
                ▼                ▼
          predictions       predictions

Integration modes:
  1. Frozen head (default): Griffin embeddings → Projection → ICL model.
     Griffin can still be fine-tuned with a proxy MSE/CE loss on a
     learnable linear probe, while the ICL model is used for final eval.
  2. TabPFN fine-tuned: Uses FinetunedTabPFNClassifier for end-to-end
     fine-tuning of TabPFN on the extracted embeddings.

ICL context construction:
  - TabPFN has a context limit (~10K samples). For large datasets, we
    subsample the training set using stratified sampling (classification)
    or random sampling (regression) to fit within the context window.
  - The max context size is configurable via `max_context_size`.

Requirements:
  pip install tabpfn     # for TabPFN v2
  pip install tabicl     # for TabICL v2
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

# Default context limits for ICL models
TABPFN_MAX_CONTEXT = 10_000   # TabPFN v2 practical limit
TABICL_MAX_CONTEXT = 48_000   # TabICL v2 practical limit


# ═══════════════════════════════════════════════════════════════════════════════
# Linear Probe — trainable proxy head for Griffin fine-tuning
# ═══════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """Simple linear probe used to fine-tune Griffin when the final
    prediction head (TabPFN/TabICL) is non-differentiable.

    During training: Griffin → LinearProbe → MSE/CE loss (gradient flows to Griffin).
    During eval: Griffin → TabPFN/TabICL → predictions (LinearProbe unused).
    """

    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim, elementwise_affine=False),
            nn.Linear(in_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# ICL Projection — trainable layer between Griffin and TabPFN/TabICL
# ═══════════════════════════════════════════════════════════════════════════════

class ICLProjection(nn.Module):
    """Trainable projection from Griffin embedding space to ICL input space.

    Compresses Griffin's 512-d embeddings to a lower dimension better suited
    for TabPFN/TabICL's feature encoder. This is trained jointly with Griffin
    (or alone if Griffin is frozen) using the linear probe loss.

    Architecture: Linear → LayerNorm → GELU → Dropout → Linear

    Training flow:
      Griffin → ICLProjection → LinearProbe → MSE/CE loss
                     ↑                          ↑
              gradients flow            gradients flow
              (trainable)               (trainable)

    Inference flow:
      Griffin → ICLProjection → numpy → TabPFN/TabICL.fit() → predict()
    """

    def __init__(self, in_dim: int = 512, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


# ═══════════════════════════════════════════════════════════════════════════════
# ICL Context Subsampling
# ═══════════════════════════════════════════════════════════════════════════════

def subsample_context(X_train: np.ndarray, y_train: np.ndarray,
                      max_context: int, seed: int = 42) -> tuple:
    """Subsample training data to fit within ICL context window.

    For classification: uses stratified sampling to preserve class balance.
    For regression: uses random sampling.

    Args:
        X_train:     [N, D] feature array
        y_train:     [N] label array
        max_context: maximum number of context examples
        seed:        random seed for reproducibility

    Returns:
        X_sub: [min(N, max_context), D]
        y_sub: [min(N, max_context)]
    """
    N = X_train.shape[0]
    if N <= max_context:
        return X_train, y_train

    rng = np.random.RandomState(seed)
    logger.info(f"[ICL] Subsampling context: {N} → {max_context} samples")

    # Check if classification (integer labels) or regression (float labels)
    is_classification = (
        y_train.dtype in (np.int32, np.int64, np.int_)
        or (np.issubdtype(y_train.dtype, np.floating)
            and np.all(y_train == y_train.astype(int)))
    )

    if is_classification:
        # Stratified sampling: equal representation per class
        classes, counts = np.unique(y_train, return_counts=True)
        per_class = max(1, max_context // len(classes))
        indices = []
        for cls in classes:
            cls_idx = np.where(y_train == cls)[0]
            n_take = min(per_class, len(cls_idx))
            indices.append(rng.choice(cls_idx, size=n_take, replace=False))
        indices = np.concatenate(indices)
        # If we have room left, fill with random samples from remainder
        if len(indices) < max_context:
            remaining = np.setdiff1d(np.arange(N), indices)
            extra = min(max_context - len(indices), len(remaining))
            if extra > 0:
                indices = np.concatenate([
                    indices, rng.choice(remaining, size=extra, replace=False),
                ])
        indices = indices[:max_context]
    else:
        # Random sampling for regression
        indices = rng.choice(N, size=max_context, replace=False)

    return X_train[indices], y_train[indices]


# ═══════════════════════════════════════════════════════════════════════════════
# TabPFN v2 Head
# ═══════════════════════════════════════════════════════════════════════════════

class TabPFNHead:
    """Wraps TabPFN v2 as a prediction head over Griffin embeddings.

    TabPFN uses in-context learning: it sees (X_train, y_train) as context
    and predicts on X_test in a single forward pass through a pretrained
    transformer — no weight updates during inference.

    Context construction:
      - `.fit(X_train, y_train)` stores training data as ICL context.
      - If N_train > max_context_size, training data is subsampled using
        stratified sampling (classification) or random sampling (regression).
      - `.predict(X_test)` concatenates [context; X_test] internally and
        runs a single forward pass. Test rows attend to context rows via
        causal masking — this IS the in-context learning mechanism.

    Args:
        task_type:        "regression" or "retrieval" (classification)
        device:           torch device string
        n_estimators:     number of ensemble members (feature subsets)
        max_context_size: max ICL context examples (subsampled if exceeded)
        model_version:    "v2.5" or "v2.6" or "default" (package default)
        finetune:         if True, use FinetunedTabPFNClassifier/Regressor
        finetune_epochs:  epochs for fine-tuning (if finetune=True)
    """

    def __init__(
        self,
        task_type: str = "regression",
        device: str = "cuda",
        n_estimators: int = 8,
        max_context_size: int = TABPFN_MAX_CONTEXT,
        model_version: str = "v2.5",
        finetune: bool = False,
        finetune_epochs: int = 30,
        finetune_lr: float = 2e-5,
    ):
        self.task_type = task_type
        self.device = device
        self.n_estimators = n_estimators
        self.max_context_size = max_context_size
        self.model_version = model_version
        self.finetune = finetune
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.model = None
        self._fitted = False

    def _resolve_model_version(self):
        """Map version string to TabPFN ModelVersion enum."""
        if self.model_version == "default":
            return None  # let TabPFN pick its default
        from tabpfn.constants import ModelVersion
        version_map = {
            "v2.5": ModelVersion.V2_5,
            "v2.6": ModelVersion.V2_6,
        }
        if self.model_version not in version_map:
            raise ValueError(
                f"Unknown TabPFN version '{self.model_version}'. "
                f"Choose from: {list(version_map.keys())} or 'default'"
            )
        return version_map[self.model_version]

    def _create_model(self):
        """Lazily create the TabPFN model (import is heavy).

        Version selection:
          - Frozen models use `TabPFNClassifier.create_default_for_version()`
            which sets model_path and default params for the chosen version.
          - FinetunedTabPFN* hardcodes v2.5 internally, so --tabpfn_version
            only affects the frozen path. A warning is logged if a non-v2.5
            version is requested with --tabpfn_finetune.
        """
        model_version = self._resolve_model_version()

        if self.finetune:
            # FinetunedTabPFNClassifier/Regressor internally hardcodes v2.5.
            # extra_classifier_kwargs can override TabPFNClassifier settings
            # but not the model version.
            if model_version is not None:
                from tabpfn.constants import ModelVersion
                if model_version != ModelVersion.V2_5:
                    logger.warning(
                        f"[TabPFN] --tabpfn_version={self.model_version} requested "
                        f"but FinetunedTabPFN* hardcodes v2.5. Using v2.5."
                    )
            from tabpfn.finetuning import (
                FinetunedTabPFNClassifier,
                FinetunedTabPFNRegressor,
            )
            if self.task_type == "regression":
                self.model = FinetunedTabPFNRegressor(
                    device=self.device,
                    epochs=self.finetune_epochs,
                    learning_rate=self.finetune_lr,
                    early_stopping=True,
                    early_stopping_patience=8,
                )
            else:
                self.model = FinetunedTabPFNClassifier(
                    device=self.device,
                    epochs=self.finetune_epochs,
                    learning_rate=self.finetune_lr,
                    early_stopping=True,
                    early_stopping_patience=8,
                )
        else:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            if model_version is not None:
                # Use create_default_for_version() — the correct API.
                # Overrides are passed as **kwargs (device, n_estimators).
                if self.task_type == "regression":
                    self.model = TabPFNRegressor.create_default_for_version(
                        model_version,
                        device=self.device,
                        n_estimators=self.n_estimators,
                    )
                else:
                    self.model = TabPFNClassifier.create_default_for_version(
                        model_version,
                        device=self.device,
                        n_estimators=self.n_estimators,
                    )
            else:
                # "default" — use package default (currently v2.6)
                if self.task_type == "regression":
                    self.model = TabPFNRegressor(
                        device=self.device,
                        n_estimators=self.n_estimators,
                    )
                else:
                    self.model = TabPFNClassifier(
                        device=self.device,
                        n_estimators=self.n_estimators,
                    )
        logger.info(f"[TabPFN] Created {'finetuned ' if self.finetune else ''}"
                     f"{'regressor' if self.task_type == 'regression' else 'classifier'}"
                     f" (version={self.model_version})")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit TabPFN on training embeddings + labels.

        If N_train exceeds max_context_size, the training data is subsampled
        to fit within TabPFN's context window. For classification, stratified
        sampling preserves class balance. For regression, random sampling.

        Args:
            X_train: [N_train, embed_dim] Griffin embeddings (numpy)
            y_train: [N_train] labels (numpy)
        """
        if self.model is None:
            self._create_model()
        # Subsample if training set exceeds context limit
        X_ctx, y_ctx = subsample_context(
            X_train, y_train, self.max_context_size,
        )
        logger.info(f"[TabPFN] Fitting on {X_ctx.shape[0]} context samples "
                     f"(from {X_train.shape[0]} total), "
                     f"{X_ctx.shape[1]} features")
        self.model.fit(X_ctx, y_ctx)
        self._fitted = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict on test embeddings.

        TabPFN internally concatenates [stored_context; X_test] and runs
        a single forward pass. Test rows attend to context rows via causal
        masking — the context examples serve as in-context demonstrations.

        Returns:
            For regression: [N_test] predictions
            For classification: [N_test, num_classes] probabilities
        """
        assert self._fitted, "Must call fit() before predict()"
        if self.task_type == "regression":
            return self.model.predict(X_test)
        else:
            return self.model.predict_proba(X_test)

    def predict_tensor(self, X_test: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Predict and return as torch.Tensor."""
        preds = self.predict(X_test)
        return torch.tensor(preds, dtype=torch.float32, device=device)


# ═══════════════════════════════════════════════════════════════════════════════
# TabICL v2 Head
# ═══════════════════════════════════════════════════════════════════════════════

class TabICLHead:
    """Wraps TabICL v2 as a prediction head over Griffin embeddings.

    TabICL v2 uses a three-stage architecture:
      1. Column Embedding — projects each feature dim to 128-d
      2. Row Interaction — transformer with RoPE, produces 512-d row vectors
      3. ICL Learning — 12-layer transformer, train rows (with labels) serve
         as context for test row predictions

    Context construction follows the same pattern as TabPFNHead:
    subsampled training data is stored as ICL context.

    Args:
        task_type:        "regression" or "retrieval" (classification)
        device:           torch device string or None (auto)
        n_estimators:     number of ensemble members (feature shuffles)
        max_context_size: max ICL context examples
        checkpoint_version: TabICL checkpoint to load. Examples:
                            - None / "default": library auto-selects
                            - "v2": map to known v2 checkpoint name
                            - explicit ".ckpt" filename: pass through verbatim
                            See tabicl release notes for available checkpoints.
        model_path:       absolute path to a local TabICL .ckpt file. If set,
                          overrides checkpoint_version (and avoids download).
    """

    # Known TabICL checkpoint version aliases. Update as the library publishes
    # new model versions on HuggingFace. v2 was the latest as of mid-2025.
    _CHECKPOINT_VERSION_MAP = {
        "default": None,                                       # let tabicl pick
        "v1": "tabicl-classifier-v1-20250208.ckpt",
        "v1.1": "tabicl-classifier-v1.1-20250506.ckpt",
        "v2": "tabicl-classifier-v2-20260212.ckpt",            # latest
    }

    def __init__(
        self,
        task_type: str = "regression",
        device: Optional[str] = None,
        n_estimators: int = 8,
        max_context_size: int = TABICL_MAX_CONTEXT,
        checkpoint_version: str = "v2",
        model_path: Optional[str] = None,
    ):
        self.task_type = task_type
        self.device = device
        self.n_estimators = n_estimators
        self.max_context_size = max_context_size
        self.checkpoint_version = checkpoint_version
        self.model_path = model_path
        self.model = None
        self._fitted = False

    def _resolve_checkpoint_kwargs(self) -> dict:
        """Build kwargs to pass to TabICLClassifier/Regressor for version
        selection.

        Returns a dict that may contain `model_path` and/or `checkpoint_version`
        depending on what's set. If both are unset (default), returns empty
        and lets the library pick.

        We probe the underlying class for accepted kwargs and only pass what
        it supports — different tabicl releases expose different version knobs.

        Note: the version aliases in _CHECKPOINT_VERSION_MAP are classifier
        checkpoints. For regression we let tabicl pick its own default
        regressor weights instead of forcing the classifier file in (which
        loads but then trips predict_stats's max_classes==0 assertion).
        """
        kwargs = {}
        if self.model_path is not None:
            kwargs["model_path"] = self.model_path
            return kwargs

        if self.task_type == "regression":
            return kwargs  # let library pick its regressor default

        # Resolve checkpoint_version alias to a real checkpoint filename
        ckpt = self._CHECKPOINT_VERSION_MAP.get(
            self.checkpoint_version, self.checkpoint_version
        )
        if ckpt is None:
            return kwargs  # let library pick default
        kwargs["checkpoint_version"] = ckpt
        return kwargs

    def _create_model(self):
        """Lazily create the TabICL model."""
        from tabicl import TabICLClassifier, TabICLRegressor
        cls = TabICLRegressor if self.task_type == "regression" else TabICLClassifier

        # tabicl's mem_get_info requires an indexed device (e.g. cuda:0);
        # accelerator.device often serialises to bare "cuda".
        device = self.device
        if isinstance(device, str) and device == "cuda":
            device = f"cuda:{torch.cuda.current_device()}"

        # Build base kwargs always supported
        ctor_kwargs = {
            "n_estimators": self.n_estimators,
            "device": device,
        }

        # Add version-related kwargs, filtering to only those the installed
        # tabicl release accepts (forward/backward-compat across versions).
        import inspect
        accepted = set(inspect.signature(cls.__init__).parameters.keys())
        for k, v in self._resolve_checkpoint_kwargs().items():
            if k in accepted:
                ctor_kwargs[k] = v
            else:
                logger.warning(
                    f"[TabICL] installed tabicl does not accept '{k}'; "
                    f"falling back to library default checkpoint. "
                    f"Upgrade tabicl to use {self.checkpoint_version}."
                )

        self.model = cls(**ctor_kwargs)
        head_type = "regressor" if self.task_type == "regression" else "classifier"
        version_msg = (
            f" model_path={self.model_path}" if self.model_path
            else f" checkpoint_version={self.checkpoint_version}"
        )
        logger.info(f"[TabICL] Created {head_type}{version_msg}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit TabICL on training embeddings + labels.

        Subsamples if N_train > max_context_size.

        Args:
            X_train: [N_train, embed_dim] Griffin embeddings (numpy)
            y_train: [N_train] labels (numpy)
        """
        if self.model is None:
            self._create_model()
        X_ctx, y_ctx = subsample_context(
            X_train, y_train, self.max_context_size,
        )
        logger.info(f"[TabICL] Fitting on {X_ctx.shape[0]} context samples "
                     f"(from {X_train.shape[0]} total), "
                     f"{X_ctx.shape[1]} features")
        self.model.fit(X_ctx, y_ctx)
        self._fitted = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict on test embeddings.

        Returns:
            For regression: [N_test] predictions
            For classification: [N_test, num_classes] probabilities
        """
        assert self._fitted, "Must call fit() before predict()"
        if self.task_type == "regression":
            return self.model.predict(X_test)
        else:
            return self.model.predict_proba(X_test)

    def predict_tensor(self, X_test: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Predict and return as torch.Tensor."""
        preds = self.predict(X_test)
        return torch.tensor(preds, dtype=torch.float32, device=device)


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding Extraction Utility
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, dataset, accelerator, icl_projection=None):
    """Extract Griffin MPNN embeddings for all samples in a dataset.

    Runs the model in eval mode and collects seed-node embeddings
    along with their labels, class embeddings (y), and task metadata.
    If an ICLProjection is provided, embeddings are projected before
    returning (so the ICL head receives projected features).

    Args:
        model:          GriffinMod (accelerator-wrapped)
        dataset:        LoaderWrapperTask or LoaderWrapperTaskLLM
        accelerator:    Accelerator instance
        icl_projection: Optional ICLProjection module (accelerator-wrapped)

    Returns:
        all_embeddings: [N, proj_dim or hiddim] numpy array
        all_labels:     [N] numpy array
        all_y:          class embeddings tensor or None (first batch's y)
        task_name:      str or None (if LoaderWrapperTaskLLM)
    """
    model.eval()
    if icl_projection is not None:
        icl_projection.eval()
    dataset.rebuild_indice(accelerator)
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=1,
        collate_fn=lambda xlist: xlist[0],
        num_workers=4, persistent_workers=False,
    )
    loader = accelerator.prepare(loader)

    embeddings_list = []
    labels_list = []
    first_y = None
    task_name = None

    for data in loader:
        # Handle both LoaderWrapperTask and LoaderWrapperTaskLLM formats
        if len(data) == 12:
            # LoaderWrapperTaskLLM: extra (taskname, rootnodetype, neighbor_mask)
            (node, mask, taskfeat, edge_index, edge_attr_type, edge_attr,
             label, y, mapping, tn, rnt, nb_mask) = data
            if task_name is None:
                task_name = tn
        else:
            # LoaderWrapperTask: 9 elements
            (node, mask, taskfeat, edge_index, edge_attr_type, edge_attr,
             label, y, mapping) = data

        all_embs = model(node, mask, taskfeat, edge_index, edge_attr_type, edge_attr)
        seed_emb = all_embs[mapping]  # [B, hiddim]

        # Apply projection if available
        if icl_projection is not None:
            seed_emb = icl_projection(seed_emb)  # [B, proj_dim]

        # Gather across processes
        seed_emb_gathered = accelerator.gather(seed_emb.unsqueeze(0)).flatten(0, 1)
        label_gathered = accelerator.gather(label.unsqueeze(0)).flatten(0, 1)

        if accelerator.is_main_process:
            embeddings_list.append(seed_emb_gathered.cpu())
            labels_list.append(label_gathered.cpu())
            if first_y is None and y is not None:
                first_y = y.cpu()

    if accelerator.is_main_process:
        all_embeddings = torch.cat(embeddings_list, dim=0).numpy()
        all_labels = torch.cat(labels_list, dim=0).numpy()
        return all_embeddings, all_labels, first_y, task_name
    else:
        return None, None, None, None


def eval_with_icl_head(icl_head, train_embs, train_labels,
                       test_embs, test_labels, metric_name):
    """Fit an ICL head on train embeddings and evaluate on test embeddings.

    Args:
        icl_head:     TabPFNHead or TabICLHead instance
        train_embs:   [N_train, D] numpy array
        train_labels: [N_train] numpy array
        test_embs:    [N_test, D] numpy array
        test_labels:  [N_test] numpy array
        metric_name:  metric string for compute_metric

    Returns:
        metric score (float)
    """
    from metric import compute_metric

    icl_head.fit(train_embs, train_labels)
    preds = icl_head.predict(test_embs)

    # Convert to tensors matching compute_metric expectations
    preds_t = torch.tensor(preds, dtype=torch.float32)
    labels_t = torch.tensor(test_labels)

    if preds_t.ndim == 1:
        preds_t = preds_t.unsqueeze(1)  # [N, 1] for regression

    return compute_metric(preds_t, labels_t, metric_name)
