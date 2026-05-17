"""Probe tabpfn.inference_config to find the InferenceConfig class fields
and the valid values for fit_mode / memory_saving_mode.

Usage:
    python probe_tabpfn_inference.py
"""
import inspect


def list_classes(mod, label):
    print(f"\n=== {label}: classes ===")
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if inspect.isclass(obj):
            print(f"  {name}")
            try:
                sig = inspect.signature(obj.__init__)
                for pname, p in sig.parameters.items():
                    if pname == "self":
                        continue
                    default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
                    print(f"    {pname}{default}")
            except (TypeError, ValueError) as e:
                print(f"    (no inspectable signature: {e})")
            # If it's a dataclass / has __dataclass_fields__
            if hasattr(obj, "__dataclass_fields__"):
                print(f"    dataclass fields: {list(obj.__dataclass_fields__.keys())}")


if __name__ == "__main__":
    print("Inspecting tabpfn.inference_config ...")
    from tabpfn import inference_config
    print(f"submodule: {inference_config.__name__}")
    print(f"top-level names: "
          f"{[n for n in dir(inference_config) if not n.startswith('_')]}")
    list_classes(inference_config, "tabpfn.inference_config")

    # Look for valid fit_mode and memory_saving_mode values.
    # In TabPFN 8.x these are likely Literal types or string enums living
    # somewhere in tabpfn.classifier or tabpfn.base.
    print("\n=== Searching for fit_mode / memory_saving_mode literals ===")
    for modname in ("tabpfn.classifier", "tabpfn.base",
                    "tabpfn.inference", "tabpfn.constants"):
        try:
            mod = __import__(modname, fromlist=["*"])
        except ImportError:
            continue
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if "FitMode" in name or "MemorySaving" in name:
                print(f"  found {modname}.{name}: {obj}")
                if hasattr(obj, "__members__"):
                    for m_name, m_val in obj.__members__.items():
                        print(f"    {m_name} = {m_val!r}")

    # Also check the source-level Literal annotations on TabPFNClassifier
    print("\n=== TabPFNClassifier source-level annotations ===")
    try:
        from tabpfn.classifier import TabPFNClassifier
        ann = getattr(TabPFNClassifier.__init__, "__annotations__", {})
        for k in ("fit_mode", "memory_saving_mode", "inference_precision",
                  "inference_config"):
            if k in ann:
                print(f"  {k}: {ann[k]}")
    except ImportError as e:
        print(f"  import failed: {e}")
