"""Probe TabPFN's installed API to find v3 + KV cache knobs.

Prints constructor + fit/predict signatures for TabPFNClassifier,
TabPFNRegressor, and (if installed) the finetuned variants. Highlights
any param matching KV-cache / inference-config / v3 keywords.

Usage:
    python probe_tabpfn.py
"""
import inspect


def show(cls, label):
    print(f"\n=== {label} ===")
    print(f"class: {cls.__module__}.{cls.__name__}")
    for method_name in ("__init__", "fit", "predict", "predict_proba"):
        if not hasattr(cls, method_name):
            continue
        sig = inspect.signature(getattr(cls, method_name))
        print(f"{method_name} params:")
        for name, p in sig.parameters.items():
            default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
            print(f"  {name}{default}")

    keywords = [
        "cache", "kv", "k_v",
        "version", "v3", "model_name", "model_path",
        "inference", "max_context", "context_size", "max_data",
        "compile", "fast", "memory", "precision",
        "device", "gpu",
    ]
    all_params = set()
    for method_name in ("__init__", "fit", "predict", "predict_proba"):
        if hasattr(cls, method_name):
            all_params |= set(inspect.signature(getattr(cls, method_name)).parameters)
    hits = [p for p in sorted(all_params)
            if any(k in p.lower() for k in keywords)]
    if hits:
        print(f"interesting params: {hits}")


if __name__ == "__main__":
    try:
        import tabpfn
        print(f"tabpfn version: {getattr(tabpfn, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"tabpfn not installed: {e}")
        raise SystemExit(1)

    from tabpfn import TabPFNClassifier, TabPFNRegressor
    show(TabPFNClassifier, "TabPFNClassifier")
    show(TabPFNRegressor, "TabPFNRegressor")

    # v3 may live in a submodule or expose a ModelVersion enum
    try:
        from tabpfn.constants import ModelVersion
        members = [m for m in dir(ModelVersion) if not m.startswith("_")]
        print(f"\n=== ModelVersion members ===")
        for m in members:
            try:
                print(f"  {m} = {getattr(ModelVersion, m)!r}")
            except Exception as e:
                print(f"  {m} (error: {e})")
    except Exception as e:
        print(f"\nModelVersion enum not found: {e}")

    # KV cache may be a separate class
    print("\n=== Looking for KV-cache / cached estimator helpers ===")
    import importlib, pkgutil
    try:
        pkg = importlib.import_module("tabpfn")
        for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__):
            print(f"  tabpfn submodule: {name}")
    except Exception as e:
        print(f"  submodule walk failed: {e}")

    try:
        from tabpfn.finetuning import (
            FinetunedTabPFNClassifier,
            FinetunedTabPFNRegressor,
        )
        show(FinetunedTabPFNClassifier, "FinetunedTabPFNClassifier")
        show(FinetunedTabPFNRegressor, "FinetunedTabPFNRegressor")
    except ImportError as e:
        print(f"\nFinetunedTabPFN* not available: {e}")
