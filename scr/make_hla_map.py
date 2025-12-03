#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d", "--data-root", type=str, required=True,
        help="folder that contains X.npy, y.npy, meta.json"
    )
    ap.add_argument(
        "--backup", action="store_true",
        help="save a backup meta.before_hla_allowed.json"
    )
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    print(f"[INFO] Using data root: {root}")

    X_path = root / "X.npy"
    y_path = root / "y.npy"
    meta_path = root / "meta.json"

    assert X_path.exists(), f"X not found: {X_path}"
    assert y_path.exists(), f"y not found: {y_path}"
    assert meta_path.exists(), f"meta.json not found: {meta_path}"

    print("[INFO] Loading X, y, meta.json ...")
    X = np.load(X_path)
    y = np.load(y_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    labels_vocab = meta["label_info"]["labels_vocab"]
    num_labels = len(labels_vocab)


    hla_dim = int(meta.get("hla_vocab_size", X.shape[1]))

    print(f"[INFO] X shape      : {X.shape}")
    print(f"[INFO] num_labels   : {num_labels}")
    print(f"[INFO] hla_dim      : {hla_dim}")

    assert X.shape[1] >= hla_dim, "X 特征维度比 hla_dim 还小，说明 hla_dim 填错了"
    y = y.astype(int)
    assert y.min() >= 0 and y.max() < num_labels, "标签索引超出 labels_vocab 范围"

    x_hla = X[:, -hla_dim:]

    allowed = np.zeros((num_labels, hla_dim), dtype=np.float32)

    for i, label in enumerate(y):
        row = x_hla[i]
        idx = np.where(row > 0)[0]  
        if idx.size == 0:
            continue 
        allowed[label, idx] = 1.0

    empty_rows = np.where(allowed.sum(axis=1) == 0)[0]
    if empty_rows.size > 0:
        print("[WARN] These labels never saw any HLA in data, "
              "they will NOT be constrained:", empty_rows.tolist())
        allowed[empty_rows, :] = 1.0

    print(f"[INFO] Built hla_allowed_map with shape {allowed.shape}")

    meta["hla_allowed_map"] = allowed.tolist()

    if args.backup and meta_path.exists():
        backup_path = meta_path.with_suffix(".before_hla_allowed.json")
        backup_path.write_text(meta_path.read_text())
        print(f"[INFO] Backed up original meta to: {backup_path}")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Updated {meta_path} with 'hla_allowed_map'")


if __name__ == "__main__":
    main()
