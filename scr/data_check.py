import json
from pathlib import Path
from collections import Counter

import numpy as np


def human(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def load_json(p: Path):
    return json.load(open(p, "r"))


def load_npy(p: Path):
    return np.load(p, allow_pickle=False)


def check_featurized_dir(
    dir_path,
    rare_thresholds=(1, 5, 10),
    topk_for_coverage=(5, 10, 20, 50),
    return_stats: bool = False,
):
    """
    >>> from data_check import check_featurized_dir
    >>> stats = check_featurized_dir("/users/.../featurized", return_stats=True)
    """

    d = Path(dir_path)
    meta_p = d / "meta.json"
    X_p = d / "X.npy"
    y_p = d / "y.npy"
    aids_p = d / "cdr3a_ids.npy"
    bids_p = d / "cdr3b_ids.npy"

    print(f"\n=== Featurized directory: {d} ===")
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {d}")

    meta = load_json(meta_p) if meta_p.exists() else {}
    print("\n[meta.json]")
    if meta:
        print(f"- X_shape (recorded): {tuple(meta.get('X_shape', []))}")
        print(
            f"- alpha_vocab_size: {meta.get('alpha_vocab_size')}, "
            f"beta_vocab_size: {meta.get('beta_vocab_size')}, "
            f"hla_vocab_size: {meta.get('hla_vocab_size')}"
        )
        if "count_features" in meta:
            print(f"- count_features: {meta.get('count_features')}")
        if "label_info" in meta and meta["label_info"]:
            li = meta["label_info"]
            print(
                f"- label_info: column='{li.get('label_col')}', "
                f"#classes={len(li.get('labels_vocab', []))}"
            )
        if "cdr3_max_tokens_alpha" in meta or "cdr3_max_tokens_beta" in meta:
            print(
                f"- cdr3_max_tokens_alpha: {meta.get('cdr3_max_tokens_alpha')}, "
                f"cdr3_max_tokens_beta: {meta.get('cdr3_max_tokens_beta')}"
            )
        if "hla_col" in meta:
            print(f"- hla_col: {meta.get('hla_col')}")
    else:
        print("- (missing meta.json)")


    def safe_load(p: Path):
        return load_npy(p) if p.exists() else None

    X = safe_load(X_p)
    y = safe_load(y_p)
    a_ids = safe_load(aids_p)
    b_ids = safe_load(bids_p)

    print("\n[Arrays]")
    if X is not None:
        nnz = int(np.count_nonzero(X))
        print(
            f"- X.npy: shape={X.shape}, dtype={X.dtype}, "
            f"nnz≈{human(nnz)} / {human(X.size)}"
        )
    else:
        print("- X.npy: (missing)")
    if y is not None:
        print(
            f"- y.npy: shape={y.shape}, dtype={y.dtype}, "
            f"#unique_labels={len(np.unique(y))}"
        )
    else:
        print("- y.npy: (missing)")
    if a_ids is not None:
        print(f"- cdr3a_ids.npy: shape={a_ids.shape}, dtype={a_ids.dtype}")
    else:
        print("- cdr3a_ids.npy: (missing)")
    if b_ids is not None:
        print(f"- cdr3b_ids.npy: shape={b_ids.shape}, dtype={b_ids.dtype}")
    else:
        print("- cdr3b_ids.npy: (missing)")

    print("\n[Consistency]")
    sizes = []
    if X is not None:
        sizes.append(("X", X.shape[0]))
    if y is not None:
        sizes.append(("y", y.shape[0]))
    if a_ids is not None:
        sizes.append(("cdr3a_ids", a_ids.shape[0]))
    if b_ids is not None:
        sizes.append(("cdr3b_ids", b_ids.shape[0]))
    if sizes:
        ok = len({n for _, n in sizes}) == 1
        print(f"- sample_count alignment: {'OK' if ok else 'MISMATCH'} -> {sizes}")
    else:
        print("- No arrays to compare.")

    stats = {
        "dir": str(d),
        "n_samples": int(X.shape[0]) if X is not None else None,
        "n_features": int(X.shape[1]) if X is not None else None,
    }


    if y is not None and meta.get("label_info", {}).get("labels_vocab"):
        vocab = meta["label_info"]["labels_vocab"]
        num_classes = len(vocab)
        ctr = Counter(y.tolist())
        counts = np.array(list(ctr.values()), dtype=np.int64)

        print("\n[Labels]")
        print(f"- #classes (in vocab): {num_classes}")
        print(f"- #classes (actually used in y): {len(ctr)}")

        count_min = int(counts.min())
        count_med = float(np.median(counts))
        count_mean = float(counts.mean())
        count_max = int(counts.max())
        count_std = float(counts.std())
        print(
            f"- count stats per class: "
            f"min={count_min}, median={count_med:.1f}, "
            f"mean={count_mean:.1f}, std={count_std:.1f}, max={count_max}"
        )

        singles = [k for k, c in ctr.items() if c == 1]
        print(f"- singletons (count == 1): {len(singles)}")

        total_samples = len(y)
        sorted_counts = np.array(sorted(counts, reverse=True))

        print("- Top-K coverage:")
        cov_stats = {}
        for k in topk_for_coverage:
            if k <= 0:
                continue
            k_eff = min(k, len(sorted_counts))
            covered = int(sorted_counts[:k_eff].sum())
            frac = covered / total_samples if total_samples > 0 else 0.0
            print(f"  Top-{k_eff:3d}: {covered:7d} samples ({frac*100:5.1f}%)")
            cov_stats[int(k_eff)] = {
                "covered_samples": covered,
                "coverage_frac": frac,
            }


        print("- Rare-class impact (if we drop classes with count < T):")
        rare_stats = {}
        for t in rare_thresholds:
            mask_rare = counts < t
            n_rare_cls = int(mask_rare.sum())
            rare_samples = int(counts[mask_rare].sum())
            kept_samples = int(total_samples - rare_samples)
            kept_classes = int(len(counts) - n_rare_cls)
            frac_samples_kept = (
                kept_samples / total_samples if total_samples > 0 else 0.0
            )
            print(
                f"  T={t:3d}: "
                f"rare_classes={n_rare_cls:4d}, "
                f"rare_samples={rare_samples:7d}, "
                f"keep_samples={kept_samples:7d} ({frac_samples_kept*100:5.1f}%), "
                f"keep_classes={kept_classes:4d}"
            )
            rare_stats[int(t)] = {
                "rare_classes": n_rare_cls,
                "rare_samples": rare_samples,
                "kept_samples": kept_samples,
                "kept_classes": kept_classes,
                "kept_fraction": frac_samples_kept,
            }

        print(f"- Top-{min(10, num_classes)} most frequent classes:")
        for lab_id, cnt in ctr.most_common(min(10, num_classes)):
            name = vocab[lab_id] if lab_id < num_classes else "<out_of_range>"
            print(f"  {lab_id:3d}: {cnt:6d}  | {name}")

        stats["labels"] = {
            "num_classes_vocab": num_classes,
            "num_classes_used": len(ctr),
            "count_min": count_min,
            "count_median": count_med,
            "count_mean": count_mean,
            "count_max": count_max,
            "count_std": count_std,
            "num_singletons": len(singles),
            "topk_coverage": cov_stats,
            "rare_class_stats": rare_stats,
        }


    # Block layout & HLA-only view
    if X is not None and meta:
        aV = int(meta.get("alpha_vocab_size", 0) or 0)
        bV = int(meta.get("beta_vocab_size", 0) or 0)
        hV = int(meta.get("hla_vocab_size", 0) or 0)
        D = X.shape[1]
        rest = D - (aV + bV + hV)

        print("\n[Block layout]")
        print(
            f"- Expected slices: "
            f"alphaV[0:{aV}] | betaV[{aV}:{aV + bV}] | "
            f"HLA[{aV + bV}:{aV + bV + hV}] | rest({rest} dims)"
        )

        def nz_rate(block):
            if block.size == 0:
                return 0.0
            return float(np.count_nonzero(block)) / float(block.size)

        a_blk = X[:, 0:aV] if aV > 0 else np.zeros((X.shape[0], 0))
        b_blk = X[:, aV : aV + bV] if bV > 0 else np.zeros((X.shape[0], 0))
        h_blk = (
            X[:, aV + bV : aV + bV + hV] if hV > 0 else np.zeros((X.shape[0], 0))
        )
        r_blk = X[:, aV + bV + hV :] if rest > 0 else np.zeros((X.shape[0], 0))

        nz_alpha = nz_rate(a_blk)
        nz_beta = nz_rate(b_blk)
        nz_hla = nz_rate(h_blk)
        nz_rest = nz_rate(r_blk)

        print(
            f"- NZ-rate: alphaV={nz_alpha:.4f}, betaV={nz_beta:.4f}, "
            f"HLA={nz_hla:.4f}, rest={nz_rest:.4f}"
        )

        stats["blocks"] = {
            "alpha_nz_rate": nz_alpha,
            "beta_nz_rate": nz_beta,
            "hla_nz_rate": nz_hla,
            "rest_nz_rate": nz_rest,
        }

        if hV > 0:
            h_count = (h_blk > 0).sum(axis=1)
            nonempty = int((h_count > 0).sum())
            mean_alleles = float(h_count.mean())
            print(
                f"- HLA coverage: {nonempty}/{X.shape[0]} "
                f"({nonempty / X.shape[0] * 100:.1f}%) have ≥1 HLA; "
                f"mean alleles/sample = {mean_alleles:.2f}"
            )
            stats["blocks"]["hla_nonempty_samples"] = nonempty
            stats["blocks"]["hla_mean_alleles_per_sample"] = mean_alleles

        eff_dim = hV
        print(
            f"- If using ONLY HLA from X: effective tabular dim = {eff_dim} "
            f"(alphaV/betaV/rest dropped)"
        )
        stats["blocks"]["effective_dim_hla_only"] = eff_dim

    # Vocab samples
    print("\n[Vocab samples]")
    for k in ["hla_vocab", "alpha_vocab", "beta_vocab"]:
        v = meta.get(k, [])
        if isinstance(v, list) and v:
            print(f"- {k}: size={len(v)}, head={v[:10]}")
        else:
            print(f"- {k}: (missing/empty)")

    # CDR3 coverage & lengths
    if a_ids is not None and "cdr3_token_vocab" in meta:
        tv = meta["cdr3_token_vocab"]
        pad_id = tv.get("<PAD>", 0)
        bos_id = tv.get("<BOS>", 1)
        eos_id = tv.get("<EOS>", 2)

        def token_len(arr):
            mask = (arr != pad_id) & (arr != bos_id) & (arr != eos_id)
            return mask.sum(axis=-1)  

        a_len = token_len(a_ids)
        b_len = token_len(b_ids) if b_ids is not None else None

        def slot_stats(name, L):
            any_tokens = int((L.sum(axis=1) > 0).sum())
            empty_both = int((L.sum(axis=1) == 0).sum())
            print(
                f"- {name}: non-empty={any_tokens}/{L.shape[0]} "
                f"({any_tokens / L.shape[0] * 100:.1f}%), empty-both={empty_both}"
            )
            per_slot_stats = []
            for s in range(L.shape[1]):
                ls = L[:, s]
                nz = int((ls > 0).sum())
                if nz > 0:
                    mean_len = float(ls[ls > 0].mean())
                    p95 = float(np.percentile(ls[ls > 0], 95))
                    max_len = int(ls.max())
                    print(
                        f"  slot{s}: mean_len={mean_len:.2f}, "
                        f"p95={p95:.1f}, max={max_len}, non-empty={nz}"
                    )
                    per_slot_stats.append(
                        {
                            "slot": int(s),
                            "mean_len": mean_len,
                            "p95_len": p95,
                            "max_len": max_len,
                            "non_empty": nz,
                        }
                    )
                else:
                    print(f"  slot{s}: all empty")
                    per_slot_stats.append(
                        {
                            "slot": int(s),
                            "mean_len": 0.0,
                            "p95_len": 0.0,
                            "max_len": 0,
                            "non_empty": 0,
                        }
                    )
            return {
                "non_empty_samples": any_tokens,
                "empty_samples": empty_both,
                "slots": per_slot_stats,
            }

        print("\n[CDR3 coverage]")
        stats["cdr3"] = {}
        stats["cdr3"]["alpha"] = slot_stats("CDR3α", a_len)
        if b_len is not None:
            stats["cdr3"]["beta"] = slot_stats("CDR3β", b_len)


    # X health
    if X is not None:
        print("\n[X health]")
        has_nan = bool(np.isnan(X).any())
        has_inf = bool(np.isinf(X).any())
        print(f"- NaN present: {has_nan}, Inf present: {has_inf}")
        if has_nan or has_inf:
            bad_rows = np.unique(np.argwhere(~np.isfinite(X))[:, 0])
            print(f"- Bad rows indices (first 20): {bad_rows[:20]}")
            stats["X_health"] = {
                "has_nan": has_nan,
                "has_inf": has_inf,
                "bad_rows_sample": bad_rows[:20].tolist(),
            }
        else:
            stats["X_health"] = {"has_nan": False, "has_inf": False}

    print("\nDone.\n")

    if return_stats:
        return stats
