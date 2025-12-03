import json
from pathlib import Path
from collections import Counter

import numpy as np


def filter_by_min_count(root_dir, min_count=5, out_dir=None):
    root = Path(root_dir)
    assert root.exists(), f"Root dir not found: {root}"

    X = np.load(root / "X.npy")
    y = np.load(root / "y.npy")
    a_ids = np.load(root / "cdr3a_ids.npy")
    b_ids = np.load(root / "cdr3b_ids.npy")
    meta = json.load(open(root / "meta.json", "r"))

    assert X.shape[0] == y.shape[0] == a_ids.shape[0] == b_ids.shape[0], "sample_count mismatch"

    N = X.shape[0]
    ctr = Counter(y.tolist())
    keep_classes = sorted([c for c, n in ctr.items() if n >= min_count])
    keep_set = set(keep_classes)

    if len(keep_classes) == 0:
        raise ValueError(f"No classes have >= {min_count} samples, nothing to keep.")

    mask = np.array([int(lbl) in keep_set for lbl in y], dtype=bool)
    X_new   = X[mask]
    y_old   = y[mask]
    a_new   = a_ids[mask]
    b_new   = b_ids[mask]

    old2new = {old: i for i, old in enumerate(keep_classes)}
    y_new = np.array([old2new[int(lbl)] for lbl in y_old], dtype=np.int64)

    li = meta.get("label_info", {})
    labels_vocab = li.get("labels_vocab", None)

    if labels_vocab is not None:
        new_labels_vocab = [labels_vocab[old] for old in keep_classes]
        li["labels_vocab"] = new_labels_vocab
    li["num_classes"] = len(keep_classes)
    li["min_count_filter"] = min_count
    li["keep_classes_old_ids"] = keep_classes

    if "label_info_orig" not in meta:
        meta["label_info_orig"] = {
            "num_classes": meta.get("label_info", {}).get("num_classes", None),
            "labels_vocab_len": len(labels_vocab) if labels_vocab is not None else None,
        }

    meta["label_info"] = li
    meta["X_shape"] = [int(X_new.shape[0]), int(X_new.shape[1])]

    if out_dir is None:
        out_dir = root.parent / f"{root.name}_min{min_count}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", X_new)
    np.save(out_dir / "y.npy", y_new)
    np.save(out_dir / "cdr3a_ids.npy", a_new)
    np.save(out_dir / "cdr3b_ids.npy", b_new)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Filter by min_count =", min_count, "===")
    print(f"- Original samples : {N}")
    print(f"- New samples      : {X_new.shape[0]} ({X_new.shape[0]/N*100:.1f}%)")
    print(f"- Original classes : {len(ctr)}")
    print(f"- New classes      : {len(keep_classes)}")
    print(f"- Output dir       : {out_dir}")
    print("  (y labels have been remapped to 0..C_new-1)")
    print("==============================\n")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True)
    p.add_argument("--min_count", type=int, default=10)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    filter_by_min_count(args.dir, min_count=args.min_count, out_dir=args.out_dir)
