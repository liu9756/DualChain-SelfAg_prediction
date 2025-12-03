# no longer use ######################
#######################################
####################################
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/users/PAS2177/liu9756/DualChain-SelfAg_prediction")
FT_DIR = ROOT / "data" / "featurized_ft"            
META_FT = FT_DIR / "meta.json"

CSV_PATH = ROOT / "data" / "my_data" / "TCR_HLA_pairs_for_prediction.csv"
OUT_DIR  = ROOT / "data" / "my_data" / "pred_featurized"   

OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(META_FT, "r") as f:
    meta_ft = json.load(f)

tv = meta_ft["cdr3_token_vocab"]  
pad_id = tv.get("<PAD>", 0)
bos_id = tv.get("<BOS>", 1)
eos_id = tv.get("<EOS>", 2)
unk_id = tv.get("<UNK>", pad_id)

max_a = int(meta_ft.get("cdr3_max_tokens_alpha", 19))
max_b = int(meta_ft.get("cdr3_max_tokens_beta", 19))

hla_dim = int(meta_ft.get("hla_vocab_size", 22))

print(f"Use max_len alpha={max_a}, beta={max_b}, hla_dim={hla_dim}")

def encode_cdr3(seq: str, max_len: int):
    if not isinstance(seq, str):
        seq = "" if seq is None else str(seq)

    tokens = [bos_id]
    for aa in seq.strip():
        tokens.append(tv.get(aa, unk_id))
    tokens.append(eos_id)

    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens += [pad_id] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.int32)

df = pd.read_csv(CSV_PATH)

required_cols = ["sample", "cdr3_alpha", "cdr3_beta"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column '{c}' in {CSV_PATH}")

N = df.shape[0]
print(f"N rows for prediction: {N}")

cdr3a = np.full((N, 2, max_a), pad_id, dtype=np.int32)
cdr3b = np.full((N, 2, max_b), pad_id, dtype=np.int32)

for i, row in df.iterrows():
    a_seq = str(row["cdr3_alpha"]) if not pd.isna(row["cdr3_alpha"]) else ""
    b_seq = str(row["cdr3_beta"])  if not pd.isna(row["cdr3_beta"])  else ""

    a_ids = encode_cdr3(a_seq, max_a)
    b_ids = encode_cdr3(b_seq, max_b)

    cdr3a[i, 0, :] = a_ids
    cdr3b[i, 0, :] = b_ids

X = np.zeros((N, hla_dim), dtype=np.float32)

print("Shapes:")
print("  X:", X.shape)
print("  cdr3a_ids:", cdr3a.shape)
print("  cdr3b_ids:", cdr3b.shape)

np.save(OUT_DIR / "X.npy",           X)
np.save(OUT_DIR / "cdr3a_ids.npy",   cdr3a)
np.save(OUT_DIR / "cdr3b_ids.npy",   cdr3b)

y_dummy = np.zeros((N,), dtype=np.int64)
np.save(OUT_DIR / "y.npy", y_dummy)

meta_pred = {
    "X_shape": [int(N), int(hla_dim)],
    "alpha_vocab_size": 0,
    "beta_vocab_size": 0,
    "hla_vocab_size": hla_dim,
    "cdr3_max_tokens_alpha": max_a,
    "cdr3_max_tokens_beta": max_b,
    "cdr3_token_vocab": meta_ft["cdr3_token_vocab"],
    "label_info": meta_ft.get("label_info", {}),
}
with open(OUT_DIR / "meta.json", "w") as f:
    json.dump(meta_pred, f)

print(f"Saved features to: {OUT_DIR}")
