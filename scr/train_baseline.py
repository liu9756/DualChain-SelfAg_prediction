# train_baseline.py
import argparse, json, os, math, joblib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, Dict, Any, List

from sklearn.model_selection import GroupKFold, StratifiedKFold, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.metrics import ndcg_score

from utils import set_seed, expected_calibration_error, mrr_score, load_data

# Main training pipeline
def run(args):
    set_seed(args.seed)
    X_path = Path(args.X)
    y_path = Path(args.y)
    meta_path = Path(args.meta)
    features_csv = Path(args.features_csv) if args.features_csv else None

    X, y, groups, labels_vocab, meta = load_data(X_path, y_path, meta_path, features_csv)

    if y is None:
        raise ValueError("y.npy was not found — please provide labels (e.g., example_epitope) before running baseline.")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if args.cv == "groupkfold":
        splitter = GroupKFold(n_splits=args.n_splits)
        split_iter = splitter.split(X, y, groups=groups)
    elif args.cv == "logo":
        splitter = LeaveOneGroupOut()
        split_iter = splitter.split(X, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(X, y)

    accs, f1m, top5s, top10s, mrrs, ndcg5s, eces = [], [], [], [], [], [], []

    fold = 0
    for tr_idx, te_idx in split_iter:
        fold += 1
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        model_kind = args.model.lower()
        if model_kind == "lr":
            clf = LogisticRegression(max_iter=800, n_jobs=args.n_jobs, C=args.C, class_weight=args.class_weight)
            clf.fit(X_tr, y_tr)
            prob_te = clf.predict_proba(X_te)
            train_classes = np.array(clf.classes_, dtype=int)  # global IDs seen in this fold

        elif model_kind == "svm":
            base = LinearSVC(C=args.C, class_weight=args.class_weight, dual=False)
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            clf.fit(X_tr, y_tr)
            prob_te = clf.predict_proba(X_te)
            train_classes = np.array(clf.classes_, dtype=int)

        elif model_kind == "xgb":
            from xgboost import XGBClassifier
            train_classes = np.unique(y_tr)                       
            fold_map = {c: i for i, c in enumerate(train_classes)} 
            y_tr_local = np.array([fold_map[c] for c in y_tr], dtype=int)

            clf = XGBClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.lr,
                subsample=args.subsample,
                colsample_bytree=args.colsample,
                reg_lambda=1.0,
                eval_metric="mlogloss",
                n_jobs=args.n_jobs,
                random_state=args.seed
            )
            clf.fit(X_tr, y_tr_local)              
            prob_te = clf.predict_proba(X_te)     

        else:
            raise ValueError(f"Unknown model: {args.model}")

        mask_seen = np.isin(y_te, train_classes)     
        coverage = float(mask_seen.mean())

        if mask_seen.sum() == 0:
            print(f"[Fold {fold}] coverage=0.00% — no overlapping classes; skipping metrics.")
            continue

        y_te_seen = y_te[mask_seen]
        prob_seen = prob_te[mask_seen]

        class_to_col = {c: i for i, c in enumerate(train_classes)}
        y_idx_seen = np.array([class_to_col[c] for c in y_te_seen], dtype=int)

        y_pred_idx = prob_seen.argmax(axis=1)
        y_pred_seen = train_classes[y_pred_idx]  
        accs.append(accuracy_score(y_te_seen, y_pred_seen))
        f1m.append(f1_score(y_te_seen, y_pred_seen, average="macro"))

        k5 = min(5, prob_seen.shape[1])
        k10 = min(10, prob_seen.shape[1])
        top5s.append(top_k_accuracy_score(y_idx_seen, prob_seen, k=k5, labels=np.arange(prob_seen.shape[1])))
        top10s.append(top_k_accuracy_score(y_idx_seen, prob_seen, k=k10, labels=np.arange(prob_seen.shape[1])))
        mrrs.append(mrr_score(y_idx_seen, prob_seen))

        Y_true_multi = np.zeros_like(prob_seen)
        Y_true_multi[np.arange(len(y_idx_seen)), y_idx_seen] = 1.0
        ndcg5s.append(ndcg_score(Y_true_multi, prob_seen, k=k5))

        eces.append(expected_calibration_error(prob_seen, y_idx_seen, n_bins=15))

        print(f"[Fold {fold}] coverage={coverage:.2%} | "
              f"ACC={accs[-1]:.3f}  F1m={f1m[-1]:.3f}  Top5={top5s[-1]:.3f}  "
              f"MRR={mrrs[-1]:.3f}  nDCG@5={ndcg5s[-1]:.3f}  ECE={eces[-1]:.3f}")

    def agg(x): return (float(np.mean(x)), float(np.std(x)))

    summary = {
        "model": args.model,
        "cv": args.cv,
        "n_splits": args.n_splits,
        "metrics": {
            "ACC": agg(accs),
            "F1_macro": agg(f1m),
            "Top5": agg(top5s),
            "Top10": agg(top10s),
            "MRR": agg(mrrs),
            "nDCG@5": agg(ndcg5s),
            "ECE": agg(eces),
        }
    }

    print("\n=== CV Summary ===")
    for k, (m, s) in summary["metrics"].items():
        print(f"{k}: {m:.3f} ± {s:.3f}")

    if args.model.lower() == "lr":
        final = LogisticRegression(max_iter=800, n_jobs=args.n_jobs, C=args.C, class_weight=args.class_weight)
        final.fit(X, y)
        final_classes = list(map(int, final.classes_))
        with open(outdir / "final_classes.json", "w") as f:
            json.dump(final_classes, f, indent=2)

    elif args.model.lower() == "svm":
        base = LinearSVC(C=args.C, class_weight=args.class_weight, dual=False)
        final = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        final.fit(X, y)
        final_classes = sorted(list(map(int, np.unique(y))))
        with open(outdir / "final_classes.json", "w") as f:
            json.dump(final_classes, f, indent=2)

    else:
        from xgboost import XGBClassifier
        final_classes = sorted(list(map(int, np.unique(y))))
        full_map = {c: i for i, c in enumerate(final_classes)}
        y_full_local = np.array([full_map[c] for c in y], dtype=int)

        final = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.lr,
            subsample=args.subsample,
            colsample_bytree=args.colsample,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            n_jobs=args.n_jobs,
            random_state=args.seed
        )
        final.fit(X, y_full_local)

        with open(outdir / "final_classes.json", "w") as f:
            json.dump(final_classes, f, indent=2)

    joblib.dump(final, outdir / "baseline_model.joblib")
    with open(outdir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if labels_vocab:
        with open(outdir / "labels_vocab.json", "w") as f:
            json.dump(labels_vocab, f, indent=2)

    print(f"\nSaved model to: {outdir / 'baseline_model.joblib'}")
    print(f"Saved CV summary to: {outdir / 'cv_summary.json'}")
    print(f"Saved final classes mapping to: {outdir / 'final_classes.json'}")
    if labels_vocab:
        print(f"Saved labels vocab to: {outdir / 'labels_vocab.json'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--X", type=str, required=True, help="Path to X.npy")
    p.add_argument("--y", type=str, required=True, help="Path to y.npy")
    p.add_argument("--meta", type=str, required=True, help="Path to meta.json")
    p.add_argument("--features_csv", type=str, required=True, help="Path to features_for_model.csv (for groups)")
    p.add_argument("--outdir", type=str, default="out_baseline", help="Output directory")
    p.add_argument("--cv", type=str, default="groupkfold", choices=["groupkfold","logo","stratified"])
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--model", type=str, default="lr", choices=["lr","svm","xgb"])
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--class_weight", type=str, default=None, choices=[None,"balanced"])
    p.add_argument("--n_estimators", type=int, default=400)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample", type=float, default=0.9)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(args)
