import pandas as pd
import numpy as np
from pathlib import Path
import re

src = Path("/users/PAS2177/liu9756/DualChain-SelfAg_prediction/data/raw/receptor_table_export_1761161599.csv")
raw = pd.read_csv(src)

sub = raw.iloc[0]
def make_new_name(col):
    subval = sub[col]
    if isinstance(subval, str) and subval.strip() and subval != col:
        return f"{col} | {subval}".strip()
    return col

clean = raw.copy()
clean.columns = [make_new_name(c) for c in raw.columns]
clean = clean.iloc[1:].reset_index(drop=True)

colmap = {
    "receptor_iri":        "Receptor | Group IRI",
    "receptor_id":         "Receptor.1 | IEDB Receptor ID",
    "receptor_ref_name":   "Receptor.2 | Reference Name",
    "receptor_type":       "Receptor.3 | Type",
    "epitope_name":        "Epitope.1 | Name",
    "epitope_source_mol":  "Epitope.2 | Source Molecule",
    "epitope_source_org":  "Epitope.3 | Source Organism",
    "mhc_allele_names":    "Assay.2 | MHC Allele Names",
    "chain1_type":         "Chain 1 | Type",
    "chain1_v_calc":       "Chain 1.4 | Calculated V Gene",
    "chain1_v_cur":        "Chain 1.5 | Curated V Gene",
    "chain1_cdr3_calc":    "Chain 1.12 | CDR3 Calculated",
    "chain1_cdr3_cur":     "Chain 1.11 | CDR3 Curated",
    "chain2_type":         "Chain 2 | Type",
    "chain2_v_calc":       "Chain 2.4 | Calculated V Gene",
    "chain2_v_cur":        "Chain 2.5 | Curated V Gene",
    "chain2_cdr3_calc":    "Chain 2.12 | CDR3 Calculated",
    "chain2_cdr3_cur":     "Chain 2.11 | CDR3 Curated",
}
colmap = {k: v for k, v in colmap.items() if v in clean.columns}
df = clean[list(colmap.values())].rename(columns={v: k for k, v in colmap.items()})

def norm_str(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return s if s else np.nan

for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].map(norm_str)

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
def clean_cdr3_seq(seq):
    if not isinstance(seq, str) or not seq:
        return np.nan
    s = re.sub(r"\s+", "", seq.upper())
    s = re.sub(r"[^A-Z\*]", "", s)
    s = "".join(ch if ch in AA_SET else "X" for ch in s)
    return s if s else np.nan

def _normalize_deam(part: str) -> str:
    if not isinstance(part, str):
        return ""
    part = part.strip()
    m = re.match(r'(?i)DEAM\s*\((.*?)\)\s*$', part)
    if not m:
        return part.strip()
    inside = m.group(1)
    toks = [t.strip().upper() for t in inside.split(",") if t.strip()]
    def _key(t):
        m2 = re.search(r'(\d+)', t)
        return int(m2.group(1)) if m2 else 10**9
    toks = sorted(set(toks), key=_key)
    return "DEAM(" + ", ".join(toks) + ")"

def normalize_epitope_label(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = s.strip()
    parts = [p.strip() for p in re.split(r'\s*\+\s*', s) if p.strip()]
    if not parts:
        return ""
    peptide = parts[0].upper().replace(" ", "")
    mods = parts[1:]
    norm_mods = []
    for m in mods:
        if re.match(r'(?i)^DEAM\s*\(', m):
            norm_mods.append(_normalize_deam(m))
        else:
            norm_mods.append(re.sub(r'\s+', ' ', m.strip()))
    out = peptide
    if norm_mods:
        out += " + " + " + ".join(norm_mods)
    return out

def to_lower_str(x):
    if isinstance(x, str):
        return x.strip().lower()
    return ""

df["chain1_type_norm"] = df.get("chain1_type", "").map(to_lower_str)
df["chain2_type_norm"] = df.get("chain2_type", "").map(to_lower_str)

_HLA_TOKEN_RE = re.compile(r'(HLA-)?([A-Z]{1,3}[A0-9]{0,2})\*[\d:]+', re.I)

def _split_hla_tokens(s: str):
    """将原始 HLA 串按分隔符切开，并保留合法 token"""
    if not isinstance(s, str) or not s.strip():
        return []
    rough = re.split(r'[;,/|\s]+', s.strip())
    toks = []
    for t in rough:
        t = t.strip()
        if not t:
            continue
        m = _HLA_TOKEN_RE.search(t)
        if m:
            core = t.upper()
            core = re.sub(r'^HLA-', '', core)
            toks.append(core)
    return sorted(set(toks))

def _pair_classII(tokens):
    if not tokens:
        return []
    by_locus = {}
    for t in tokens:
        locus = t.split('*', 1)[0]  # DQA1, DQB1, DRA, DRB1, DPA1, DPB1 ...
        by_locus.setdefault(locus, []).append(t)
    pairs = set()

    def _mk_pairs(a_key, b_key):
        a = by_locus.get(a_key, [])
        b = by_locus.get(b_key, [])
        for x in a:
            for y in b:
                pairs.add(f"{x}~{y}")

    _mk_pairs("DQA1", "DQB1")
    _mk_pairs("DPA1", "DPB1")
    _mk_pairs("DRA", "DRB1")

    return sorted(pairs)

def pick_seq_for_chain(row, target):  # target: "alpha" | "beta"
    t1 = row.get("chain1_type_norm", "")
    t2 = row.get("chain2_type_norm", "")
    if t1 == target:
        v = row.get("chain1_cdr3_calc")
        if pd.isna(v) or not isinstance(v, str):
            v = row.get("chain1_cdr3_cur")
        return clean_cdr3_seq(v) if isinstance(v, str) else np.nan
    elif t2 == target:
        v = row.get("chain2_cdr3_calc")
        if pd.isna(v) or not isinstance(v, str):
            v = row.get("chain2_cdr3_cur")
        return clean_cdr3_seq(v) if isinstance(v, str) else np.nan
    else:
        return np.nan

def pick_v_for_chain(row, target):
    t1 = row.get("chain1_type_norm", "")
    t2 = row.get("chain2_type_norm", "")
    if t1 == target:
        v = row.get("chain1_v_calc")
        if pd.isna(v) or not isinstance(v, str):
            v = row.get("chain1_v_cur")
        return v if isinstance(v, str) else np.nan
    elif t2 == target:
        v = row.get("chain2_v_calc")
        if pd.isna(v) or not isinstance(v, str):
            v = row.get("chain2_v_cur")
        return v if isinstance(v, str) else np.nan
    else:
        return np.nan

df["cdr3a"] = df.apply(lambda r: pick_seq_for_chain(r, "alpha"), axis=1)
df["cdr3b"] = df.apply(lambda r: pick_seq_for_chain(r, "beta"),  axis=1)
df["v_alpha_selected"] = df.apply(lambda r: pick_v_for_chain(r, "alpha"), axis=1)
df["v_beta_selected"]  = df.apply(lambda r: pick_v_for_chain(r, "beta"),  axis=1)

df["epitope_name_raw"]  = df.get("epitope_name", "")
df["epitope_name_norm"] = df["epitope_name_raw"].map(normalize_epitope_label)

df["mhc_tokens"]      = df.get("mhc_allele_names", "").map(_split_hla_tokens)
df["mhc_tokens_norm"] = df["mhc_tokens"].map(lambda xs: ";".join(xs) if xs else "")
df["hla_pairs_list"]  = df["mhc_tokens"].map(_pair_classII)
df["hla_pairs"]       = df["hla_pairs_list"].map(lambda xs: ";".join(xs) if xs else "")

n_rows = len(df)
n_alpha_rows = int(df["chain1_type_norm"].eq("alpha").sum() + df["chain2_type_norm"].eq("alpha").sum())
n_beta_rows  = int(df["chain1_type_norm"].eq("beta").sum()  + df["chain2_type_norm"].eq("beta").sum())
n_cdr3a = int(df["cdr3a"].notna().sum())
n_cdr3b = int(df["cdr3b"].notna().sum())
print(f"[INFO] rows={n_rows}, alpha-typed rows≈{n_alpha_rows}, beta-typed rows≈{n_beta_rows}")
print(f"[INFO] CDR3a non-null={n_cdr3a}, CDR3b non-null={n_cdr3b}")
print(f"[INFO] v_alpha_selected non-null={int(df['v_alpha_selected'].notna().sum())}, v_beta_selected non-null={int(df['v_beta_selected'].notna().sum())}")

n_epitope_raw  = df["epitope_name_raw"].fillna("").nunique()
n_epitope_norm = df["epitope_name_norm"].fillna("").nunique()
print(f"[INFO] epitope labels: raw_unique={n_epitope_raw}, normalized_unique={n_epitope_norm}")

has_hla = (df["mhc_tokens_norm"] != "")
has_pair = (df["hla_pairs"] != "")
print(f"[INFO] HLA tokens coverage: {int(has_hla.sum())}/{n_rows} ({has_hla.mean()*100:.1f}%)")
print(f"[INFO] HLA class-II pairs coverage: {int(has_pair.sum())}/{n_rows} ({has_pair.mean()*100:.1f}%)")

def uniq_join(seqs: pd.Series) -> str:
    vals = [s for s in seqs.dropna().astype(str).tolist() if s]
    if not vals:
        return ""
    uniq = sorted(set(vals))
    return ";".join(uniq)

def len_mean(seqs: pd.Series) -> float:
    vals = [len(s) for s in seqs.dropna().astype(str).tolist() if s]
    return float(np.mean(vals)) if vals else 0.0


rid = df["receptor_id"].astype(str)
rid = rid.where(rid.str.strip().ne(""), other=np.nan)
rid = rid.fillna("NOID_" + pd.Series(range(len(df))).astype(str))
df["rid_effective"] = rid

group_cols = ["rid_effective", "epitope_name_norm"]   # receptor × normalized epitope
agg = (
    df.groupby(group_cols, dropna=False)
      .agg(
          n_rows=("rid_effective","size"),
          n_alpha=("cdr3a", lambda s: s.dropna().nunique()),
          n_beta=("cdr3b", lambda s: s.dropna().nunique()),
          v_alpha=("v_alpha_selected", uniq_join),
          v_beta=("v_beta_selected", uniq_join),
          cdr3a_seqs=("cdr3a", uniq_join),
          cdr3b_seqs=("cdr3b", uniq_join),
          cdr3a_len_mean=("cdr3a", len_mean),
          cdr3b_len_mean=("cdr3b", len_mean),
          example_mhc_raw=("mhc_allele_names", "first"),
          example_mhc_tokens=("mhc_tokens_norm", "first"),
          example_hla_pairs=("hla_pairs", uniq_join),  # 多个样本合并取唯一
          receptor_ref=("receptor_ref_name", "first"),
      )
      .reset_index()
)

features = agg.rename(columns={
    "rid_effective":"iedb_receptor_id",
    "v_alpha":"calc_v_alpha",
    "v_beta":"calc_v_beta",
    "epitope_name_norm":"example_epitope"
})[
    ["iedb_receptor_id","example_epitope","calc_v_alpha","calc_v_beta",
     "n_alpha","n_beta","n_rows",
     "cdr3a_seqs","cdr3b_seqs","cdr3a_len_mean","cdr3b_len_mean",
     "example_mhc_raw","example_mhc_tokens","example_hla_pairs",
     "receptor_ref"]
]

print(f"[INFO] aggregated pairs (receptor×epitope_norm) = {len(features)}")
missing_alpha = int((features["cdr3a_seqs"]=="").sum())
missing_beta  = int((features["cdr3b_seqs"]=="").sum())
print(f"[INFO] features rows with empty cdr3a={missing_alpha}, empty cdr3b={missing_beta}")


out_dir = Path("/users/PAS2177/liu9756/DualChain-SelfAg_prediction/data/processed")
out_dir.mkdir(exist_ok=True, parents=True)
df.to_csv(out_dir / "receptors_normalized_rows.csv", index=False)
agg.to_csv(out_dir / "receptors_aggregated_per_receptor.csv", index=False)
features.to_csv(out_dir / "features_for_model.csv", index=False)

print("Saved updated files:")
print(out_dir / "receptors_normalized_rows.csv")
print(out_dir / "receptors_aggregated_per_receptor.csv")
print(out_dir / "features_for_model.csv")
