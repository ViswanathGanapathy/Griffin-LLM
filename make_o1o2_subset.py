"""Build a pruned joint-v65 subset for the LoG rebuttal ablations.

The two ablations (V6 on o2->o1, alpha=1e-6 on o1->o2) touch only the
others-1 and others-2 families, but Graph.__init__ loads EVERY node type
listed in metanode.yaml — so shipping a subset to a cloud box requires
pruned metadata, not just fewer directories. This script stages a
self-consistent corpus (~18 GB instead of ~90 GB):

  node/, edge/    only the 7 others-1/others-2 databases' types
  metanode.yaml,  pruned to those types
  metaadj.yaml
  task/,          kept in full (only ~325 MB; Task.__init__ loads all),
  metatask.yaml   with entries whose target types are absent left in place
                  (they are only dereferenced for requested tasks)
  *.pt embeddings copied as-is (superset keys are harmless)

Usage:
    python make_o1o2_subset.py [--src datasets/joint-v65] \
        [--out /tmp/joint-v65-o1o2] [--tar]

With --tar, also writes <out>.tar (uncompressed: the payload is float
embeddings, which do not compress; tar streams links dereferenced).
"""
import argparse
import os
import os.path as osp
import shutil
import subprocess

import yaml

PREFIXES = ("rel-f1-", "stackexchange-", "virus-",           # others-1
            "airbnb-", "rel-trial-", "talkingdata-", "telstra-")  # others-2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/joint-v65")
    ap.add_argument("--out", default="/tmp/joint-v65-o1o2")
    ap.add_argument("--tar", action="store_true")
    args = ap.parse_args()

    with open(osp.join(args.src, "metanode.yaml")) as f:
        metanode = yaml.safe_load(f)
    with open(osp.join(args.src, "metaadj.yaml")) as f:
        metaadj = yaml.safe_load(f)

    keep = [nt for nt in metanode if nt.startswith(PREFIXES)]
    assert len(keep) == 65, f"expected 65 o1+o2 node types, got {len(keep)}"
    print(f"keeping {len(keep)}/{len(metanode)} node types")

    os.makedirs(args.out, exist_ok=True)
    with open(osp.join(args.out, "metanode.yaml"), "w") as f:
        yaml.dump({nt: metanode[nt] for nt in keep}, f)
    with open(osp.join(args.out, "metaadj.yaml"), "w") as f:
        yaml.dump({nt: metaadj[nt] for nt in keep}, f)

    # symlink the heavy directories; tar -h dereferences them later
    for sub in ("node", "edge"):
        os.makedirs(osp.join(args.out, sub), exist_ok=True)
        for nt in keep:
            src_d = osp.join(args.src, sub, nt)
            dst_d = osp.join(args.out, sub, nt)
            if osp.exists(src_d) and not osp.lexists(dst_d):
                os.symlink(osp.abspath(src_d), dst_d)
    for entry in ("task", "metatask.yaml", "featnameemb.pt",
                  "edgenameemb.pt", "tasknameemb.pt"):
        src_e = osp.join(args.src, entry)
        dst_e = osp.join(args.out, entry)
        if osp.lexists(dst_e):
            continue
        if osp.isdir(src_e):
            os.symlink(osp.abspath(src_e), dst_e)
        else:
            shutil.copy2(src_e, dst_e)

    # sanity: the loader must accept the pruned corpus
    import sys
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)) or ".")
    from hdataset import Graph, Task
    g = Graph(args.out)
    assert len(g.nodes) == len(keep)
    Task(args.out)
    print(f"subset OK: Graph loads {len(g.nodes)} types from {args.out}")

    du = subprocess.run(["du", "-shL", args.out], capture_output=True,
                        text=True).stdout.split()[0]
    print(f"subset size (dereferenced): {du}")

    if args.tar:
        tarpath = args.out.rstrip("/") + ".tar"
        base = osp.basename(args.out.rstrip("/"))
        print(f"writing {tarpath} (this streams ~{du}) ...")
        subprocess.run(["tar", "-chf", tarpath, "-C",
                        osp.dirname(osp.abspath(args.out)), base], check=True)
        print(f"done: {tarpath}")
        print(f"on the pod:  tar -xf {osp.basename(tarpath)} && "
              f"mv {base} datasets/joint-v65")


if __name__ == "__main__":
    main()
