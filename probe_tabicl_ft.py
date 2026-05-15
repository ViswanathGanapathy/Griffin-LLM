"""Print FinetunedTabICL{Classifier,Regressor} constructor + fit signatures.
Run this on the RunPod environment to find out what knobs tabicl exposes for
mixed precision, batching, and multi-GPU.

Usage:
    python probe_tabicl_ft.py
"""
import inspect


def show(cls, label):
    print(f"\n=== {label} ===")
    print(f"class: {cls.__module__}.{cls.__name__}")
    init_sig = inspect.signature(cls.__init__)
    print("__init__ params:")
    for name, p in init_sig.parameters.items():
        default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
        print(f"  {name}{default}")
    fit_sig = inspect.signature(cls.fit)
    print("fit params:")
    for name, p in fit_sig.parameters.items():
        default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
        print(f"  {name}{default}")

    # Highlight anything that looks multi-GPU / memory-related
    keywords = [
        "device", "gpu", "multi", "ddp", "distributed", "rank", "world",
        "mixed", "fp16", "bf16", "amp", "precision",
        "batch", "accum", "checkpoint", "grad_checkpoint",
        "compile", "jit",
    ]
    all_params = set(init_sig.parameters) | set(fit_sig.parameters)
    hits = [p for p in sorted(all_params)
            if any(k in p.lower() for k in keywords)]
    if hits:
        print(f"interesting params: {hits}")
    else:
        print("interesting params: (none matched device/precision/batch keywords)")


if __name__ == "__main__":
    from tabicl import FinetunedTabICLClassifier, FinetunedTabICLRegressor
    show(FinetunedTabICLClassifier, "FinetunedTabICLClassifier")
    show(FinetunedTabICLRegressor, "FinetunedTabICLRegressor")
