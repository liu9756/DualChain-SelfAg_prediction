import pandas as pd
import numpy as np
import re

def strip_dual_marker_token(token: str) -> str:
    """Remove any dual marker from a single token (without altering other chars)."""
    if not isinstance(token, str):
        return token
    t = re.sub(r'(\*2|x2|×2)', '', token, flags=re.IGNORECASE)
    return t.strip()

def infer_dual_flag(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r'(\*2|x2|×2)', text, flags=re.IGNORECASE))

def split_multi_values_preserve_raw(v_string: str):
    """Split a semicolon-separated field into tokens; return (raw_tokens, tokens_without_markers)."""
    if not isinstance(v_string, str) or not v_string.strip():
        return [], []
    parts = [p.strip() for p in v_string.split(";") if p.strip()]
    raw = parts[:]  # preserve raw (may include '*2')
    clean = [strip_dual_marker_token(p) for p in parts]
    # Normalize spacing/case for vocab purposes
    clean = [re.sub(r'\s+', '', c).upper() for c in clean if c]
    clean = sorted(set(clean))
    return raw, clean