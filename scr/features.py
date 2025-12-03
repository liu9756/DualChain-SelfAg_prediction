import pandas as pd
import numpy as np
import json, re
from pathlib import Path

FEATURES_IN = Path("/users/PAS2177/liu9756/DualChain-SelfAg_prediction/data/processed/features_for_model.csv")
OUTDIR = Path("/users/PAS2177/liu9756/DualChain-SelfAg_prediction/data/featurized_300_ft")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURES_IN)

def norm_str(x):
    if pd.isna(x): return ""
    return str(x).strip()

# HLA class-II token
HLA_TOKEN_RE = re.compile(r'((?:HLA-)?(?:DRB[1-5]|DQA1|DQB1|DRA)\*[\daA-Z]+(?::[\daA-Z]+){1,3})')
def parse_hla_tokens(text: str):
    if not isinstance(text, str) or not text.strip(): return []
    ms = HLA_TOKEN_RE.findall(text)
    out = []
    for m in ms:
        m = m.strip().upper()
        if not m.startswith("HLA-"): m = "HLA-" + m
        out.append(m)
    return sorted(set(out))

# pair HLA alpha beta
def build_pairs_from_tokens(tokens):
    tset = set(tokens)
    dqa = sorted([t for t in tset if t.startswith("HLA-DQA1")])
    dqb = sorted([t for t in tset if t.startswith("HLA-DQB1")])
    dra = sorted([t for t in tset if t.startswith("HLA-DRA")])
    drb = sorted([t for t in tset if t.startswith("HLA-DRB")])

    pairs = []
    for a in dqa:
        for b in dqb:
            pairs.append(f"{a}~{b}")
    for a in (dra if dra else []):
        for b in drb:
            pairs.append(f"{a}~{b}")
    if not dra and drb:
        for b in drb:
            pairs.append(f"HLA-DRA*01:01~{b}")

    return sorted(set(pairs))

label_col_candidates = ["example_epitope", "epitope_norm", "epitope_name"]
y_col = next((c for c in label_col_candidates if c in df.columns), None)
labels_vocab, y = [], None
if y_col is not None:
    labels = df[y_col].fillna("").astype(str).tolist()
    labels_vocab = sorted(set(labels))
    lab2id = {v:i for i,v in enumerate(labels_vocab)}
    y = np.array([lab2id[v] for v in labels], dtype=np.int64)

# HLA(token + pair + missing_flag) 
TOK_COL = "example_mhc_tokens" if "example_mhc_tokens" in df.columns else None
PAIR_COL = "example_hla_pairs"  if "example_hla_pairs"  in df.columns else None
RAW_COL  = next((c for c in ["example_mhc","Assay.2 | MHC Allele Names","MHC Allele Names"] if c in df.columns), None)

N = len(df)
if TOK_COL:
    token_lists = df[TOK_COL].fillna("").astype(str).apply(
        lambda s: [t for t in (x.strip() for x in s.split(";")) if t]
    ).tolist()
else:
    token_lists = df[RAW_COL].fillna("").astype(str).apply(parse_hla_tokens).tolist() if RAW_COL else [[] for _ in range(N)]

if PAIR_COL:
    pair_lists = df[PAIR_COL].fillna("").astype(str).apply(
        lambda s: [p for p in (x.strip() for x in s.split(";")) if p]
    ).tolist()
else:
    pair_lists = [build_pairs_from_tokens(tokens) for tokens in token_lists]

hla_token_vocab = sorted(set(t for row in token_lists for t in row))
hla_pair_vocab  = sorted(set(p for row in pair_lists  for p in row))
tok_index = {v:i for i,v in enumerate(hla_token_vocab)}
pair_index = {v:i for i,v in enumerate(hla_pair_vocab)}

# multi-hot / one-hot
X_hla_tok  = np.zeros((N, len(hla_token_vocab)), dtype=np.float32)
X_hla_pair = np.zeros((N, len(hla_pair_vocab)),  dtype=np.float32)
hla_missing = np.zeros((N, 1), dtype=np.float32)

for r in range(N):
    if token_lists[r]:
        for t in token_lists[r]:
            if t in tok_index:  X_hla_tok[r, tok_index[t]] = 1.0
    if pair_lists[r]:
        for p in pair_lists[r]:
            if p in pair_index: X_hla_pair[r, pair_index[p]] = 1.0
    if not token_lists[r]:     # 完全无 HLA
        hla_missing[r, 0] = 1.0

X = np.concatenate([X_hla_tok, X_hla_pair, hla_missing], axis=1)

#CDR3 alpha beta seq as token ids
def split_multi_seq(text: str):
    if not isinstance(text, str) or not text.strip(): return []
    return [p.strip() for p in text.split(";") if p.strip()]

cdr3a_col = "cdr3a_seqs" if "cdr3a_seqs" in df.columns else None
cdr3b_col = "cdr3b_seqs" if "cdr3b_seqs" in df.columns else None

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
AA_TO_ID = {"<PAD>":0,"<BOS>":1,"<EOS>":2,"X":3}
for i, aa in enumerate(AA_ORDER, start=4): AA_TO_ID[aa]=i

def clean_cdr3_seq(seq: str) -> str:
    if not isinstance(seq, str) or not seq: return ""
    s = re.sub(r"\s+","",seq.upper())
    s = re.sub(r"[^A-Z\*]","",s)
    s = "".join(ch if ch in AA_SET else "X" for ch in s)
    return s

def tokenize_seq(seq: str, max_len: int):
    toks = [AA_TO_ID["<BOS>"]] + [AA_TO_ID.get(ch, AA_TO_ID["X"]) for ch in seq] + [AA_TO_ID["<EOS>"]]
    if len(toks) > max_len:
        toks = toks[:max_len]
        toks[-1] = AA_TO_ID["<EOS>"]
    else:
        toks += [AA_TO_ID["<PAD>"]] * (max_len - len(toks))
    return np.array(toks, dtype=np.int32)

def seq_lists_from_col(colname):
    if not colname: return [[""]] * N
    out = []
    for s in df[colname].fillna("").astype(str).tolist():
        seqs = [clean_cdr3_seq(x) for x in split_multi_seq(s)]
        out.append(seqs[:2])
    return out

cdr3a_lists = seq_lists_from_col(cdr3a_col)
cdr3b_lists = seq_lists_from_col(cdr3b_col)

max_len_a = max([max([len(x) for x in seqs], default=0) for seqs in cdr3a_lists] + [0])
max_len_b = max([max([len(x) for x in seqs], default=0) for seqs in cdr3b_lists] + [0])
MAX_TOK_A = min(32, (max_len_a + 2) if max_len_a>0 else 2)   # 短序列，适当收紧上限
MAX_TOK_B = min(32, (max_len_b + 2) if max_len_b>0 else 2)

cdr3a_ids = np.zeros((N, 2, MAX_TOK_A), dtype=np.int32)
cdr3b_ids = np.zeros((N, 2, MAX_TOK_B), dtype=np.int32)

for i in range(N):
    for j in range(2):
        seq_a = cdr3a_lists[i][j] if j < len(cdr3a_lists[i]) else ""
        cdr3a_ids[i,j] = tokenize_seq(seq_a, MAX_TOK_A) if seq_a else np.array([AA_TO_ID["<PAD>"]]*MAX_TOK_A, dtype=np.int32)
        seq_b = cdr3b_lists[i][j] if j < len(cdr3b_lists[i]) else ""
        cdr3b_ids[i,j] = tokenize_seq(seq_b, MAX_TOK_B) if seq_b else np.array([AA_TO_ID["<PAD>"]]*MAX_TOK_B, dtype=np.int32)


np.save(OUTDIR / "X.npy", X)
if y is not None: np.save(OUTDIR / "y.npy", y)
np.save(OUTDIR / "cdr3a_ids.npy", cdr3a_ids)
np.save(OUTDIR / "cdr3b_ids.npy", cdr3b_ids)

meta = {
    "X_shape": tuple(X.shape),
    "alpha_vocab_size": 0,
    "beta_vocab_size": 0,
    "hla_vocab_size": int(X_hla_tok.shape[1] + X_hla_pair.shape[1] + 1),
    "hla_blocks": {
        "tokens": int(X_hla_tok.shape[1]),
        "pairs": int(X_hla_pair.shape[1]),
        "has_missing_flag": True
    },
    "hla_token_vocab": hla_token_vocab,
    "hla_pair_vocab":  hla_pair_vocab,
    "hla_missing_flag_idx": int(X.shape[1] - 1),
    "source_csv": str(FEATURES_IN),
    "label_info": {"label_col": y_col, "labels_vocab": labels_vocab} if y_col else {},
    "cdr3a_col": cdr3a_col, "cdr3b_col": cdr3b_col,
    "cdr3_token_vocab": AA_TO_ID,
    "cdr3_max_tokens_alpha": int(MAX_TOK_A),
    "cdr3_max_tokens_beta":  int(MAX_TOK_B),
    "dual_seq_capacity": {"alpha": 2, "beta": 2},
}
with open(OUTDIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved features to:", OUTDIR)
print(f"  X (HLA only) shape={X.shape}  | tokens={X_hla_tok.shape[1]}  pairs={X_hla_pair.shape[1]}  +missing=1")
if y is not None:
    print(f"  y shape={y.shape}  | #classes={len(labels_vocab)} (label_col={y_col})")
print(f"  cdr3a_ids: {cdr3a_ids.shape}  cdr3b_ids: {cdr3b_ids.shape}")
