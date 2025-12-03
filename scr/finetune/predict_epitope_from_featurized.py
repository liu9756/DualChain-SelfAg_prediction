import argparse, json, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_featurized_for_pred(d: Path):
    X = np.load(d / "X.npy").astype(np.float32)
    a_ids = np.load(d / "cdr3a_ids.npy")
    b_ids = np.load(d / "cdr3b_ids.npy")
    meta = json.load(open(d / "meta.json", "r"))

    y_path = d / "y.npy"
    y = np.load(y_path).astype(np.int64) if y_path.exists() else None

    return X, a_ids, b_ids, meta, y


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: (B, L, d)
        L = x.size(1)
        assert L <= self.pe.size(0), f"Seq len {L} > PE max_len {self.pe.size(0)}"
        return x + self.pe[:L, :]

class SharedSeqEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model=32,
                 nhead=4,
                 nlayers=1,
                 dropout=0.0,
                 pad_id=0,
                 use_cnn_frontend=True,
                 cnn_hidden_factor=2,
                 slot_drop=0.0):
        super().__init__()
        self.pad_id = pad_id
        self.use_cnn_frontend = bool(use_cnn_frontend)
        self.slot_drop = float(slot_drop)

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.chain_emb = nn.Embedding(2, d_model)  # 0=alpha, 1=beta
        self.pos = SinusoidalPositionalEncoding(d_model)

        if self.use_cnn_frontend:
            hidden = cnn_hidden_factor * d_model
            self.cnn = nn.Sequential(
                nn.Conv1d(d_model, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, d_model, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.cnn = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.att_tok  = nn.Linear(d_model, 1)
        self.att_slot = nn.Linear(d_model, 1)

    def _encode_slots(self, x_ids, chain_type_id):
        B, S, L = x_ids.shape
        x = x_ids.view(B * S, L)
        pad_mask = (x == self.pad_id)

        h = self.emb(x)
        ct = self.chain_emb.weight[chain_type_id].view(1, 1, -1)
        h = h + ct
        h = self.pos(h)

        if self.cnn is not None:
            h_c = h.transpose(1, 2)
            h_c = self.cnn(h_c)
            h = h_c.transpose(1, 2)

        h = self.encoder(h, src_key_padding_mask=pad_mask)

        score = self.att_tok(h).squeeze(-1)
        score = score.masked_fill(pad_mask, -1e9)
        attw = torch.softmax(score, dim=-1)

        all_pad = pad_mask.all(dim=1)
        attw = attw.masked_fill(all_pad.unsqueeze(1), 0.0)
        v = (h * attw.unsqueeze(-1)).sum(dim=1)
        v = v.view(B, S, -1)
        return v, (~all_pad.view(B, S)).any(dim=1)

    def forward(self, ids_alpha, ids_beta):
        v_a, has_a = self._encode_slots(ids_alpha, chain_type_id=0)
        v_b, has_b = self._encode_slots(ids_beta, chain_type_id=1)

        def aggregate(v, has_mask):
            B, S, D = v.shape
            if self.training and self.slot_drop > 0.0:
                drop = (torch.rand(B, S, device=v.device) < self.slot_drop)
                all_dropped = drop.all(dim=1)
                if all_dropped.any():
                    drop[all_dropped, 0] = False
                v = v.masked_fill(drop.unsqueeze(-1), 0.0)
            slot_score = self.att_slot(v).squeeze(-1)
            slot_attw = torch.softmax(slot_score, dim=-1).unsqueeze(-1)
            out = (v * slot_attw).sum(dim=1)
            out = out * has_mask.float().unsqueeze(-1)
            return out

        e_a = aggregate(v_a, has_a)
        e_b = aggregate(v_b, has_b)
        return e_a, e_b, has_a, has_b

class HLAEncoder(nn.Module):
    def __init__(self, hla_dim, d_model=128, pdrop=0.0, nhead=4, nlayers=1):
        super().__init__()
        self.hla_dim = hla_dim
        self.d_model = d_model          # 记录 d_model，方便兜底时构造向量

        self.emb = nn.Embedding(hla_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=hla_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=pdrop,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.att = nn.Linear(d_model, 1)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x_hla):
   
        B, H = x_hla.shape
        assert H == self.hla_dim, f"HLA dim mismatch: got {H}, expected {self.hla_dim}"
        device = x_hla.device


        pad_mask = (x_hla <= 0)

      
        if pad_mask.all():
            zero = torch.zeros(B, self.d_model,
                               device=device,
                               dtype=self.emb.weight.dtype)
            return self.dropout(self.ln(zero))

        idx = torch.arange(H, device=device).unsqueeze(0).expand(B, H)
        h = self.emb(idx)
        h = self.pos(h)

        h = self.encoder(h, src_key_padding_mask=pad_mask)

        score = self.att(h).squeeze(-1)
        score = score.masked_fill(pad_mask, -1e9)
        w = torch.softmax(score, dim=-1)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)

        return self.dropout(self.ln(pooled))


class CosineMarginClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, s=30.0, m=0.2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_normal_(self.W)
        self.s = float(s)
        self.m = float(m)

    def forward(self, x, y=None):
        x_norm = F.normalize(x, dim=-1)
        W_norm = F.normalize(self.W, dim=-1)
        cos = torch.matmul(x_norm, W_norm.t())
        if y is None:
            return self.s * cos
        cos_y = cos.gather(1, y.view(-1, 1)).squeeze(1)
        cos_y_m = cos_y - self.m
        logits = cos.clone()
        logits.scatter_(1, y.view(-1, 1), cos_y_m.unsqueeze(1))
        return self.s * logits

class EpitopeClassifier(nn.Module):
    def __init__(self, meta, x_dim, num_classes,
                 lambda_hla=1.0, lambda_seq_alpha=1.0, lambda_seq_beta=1.0,
                 d_model=128, use_hla=True, use_seq_a=True, use_seq_b=True,
                 dropout_tab=0.0, dropout_seq=0.0,
                 use_cosine_head=False, margin_m=0.20, scale_s=30.0,
                 slot_drop=0.0):
        super().__init__()

        self.num_classes = num_classes
        self.lambda_hla = lambda_hla
        self.lambda_seq_alpha = lambda_seq_alpha
        self.lambda_seq_beta = lambda_seq_beta
        self.use_hla = bool(use_hla)
        self.use_seq_a = bool(use_seq_a)
        self.use_seq_b = bool(use_seq_b)
        self.use_cosine_head = bool(use_cosine_head)
        self.slot_drop = float(slot_drop)

        hla_dim = int(meta.get("hla_vocab_size", x_dim))
        self.slice_hla = slice(x_dim - hla_dim, x_dim)
        self.slice_alphaV = slice(0, 0)
        self.slice_betaV  = slice(0, 0)
        self.slice_rest   = slice(x_dim, x_dim)
        self.hla_enc = HLAEncoder(hla_dim, d_model=d_model, pdrop=dropout_tab)
        self.hla_scale = nn.Parameter(torch.tensor(1.0))
        self.hla_dim = hla_dim

        vocab_size = len(meta["cdr3_token_vocab"])
        pad_id = meta["cdr3_token_vocab"].get("<PAD>", 0)
        self.seq_enc = SharedSeqEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nlayers=2,
            nhead=4,
            pad_id=pad_id,
            dropout=dropout_seq,
            use_cnn_frontend=True,
            slot_drop=self.slot_drop
        )

        in_dim = 0
        if self.use_hla:   in_dim += d_model
        if self.use_seq_a: in_dim += d_model
        if self.use_seq_b: in_dim += d_model

        if self.use_cosine_head:
            self.head = CosineMarginClassifier(in_dim, num_classes, s=scale_s, m=margin_m)
        else:
            self.head = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, num_classes)
            )

        allowed = None
        li = meta.get("label_info", {})
        if "hla_allowed_map" in meta:
            allowed = np.asarray(meta["hla_allowed_map"])
        elif isinstance(li, dict) and "hla_allowed_map" in li:
            allowed = np.asarray(li["hla_allowed_map"])
        if allowed is not None:
            if allowed.shape == (num_classes, hla_dim):
                mask = torch.tensor(allowed.astype(np.float32))
                self.register_buffer("allowed_map", mask)
            else:
                print(f"[Warn] hla_allowed_map shape {allowed.shape} != ({num_classes},{hla_dim}) -> disabled")

    def forward(self, x_tab, ids_a, ids_b, y=None, debug_block_means=False):
        x = x_tab.clone()
        feats = []
        x_hla = None

        if self.use_hla:
            x_hla = x[:, self.slice_hla]
            assert x_hla.size(1) == self.hla_dim, \
                f"HLA dim mismatch: x has {x_hla.size(1)} but encoder expects {self.hla_dim}"
            h_hla = self.hla_enc(x_hla) * self.hla_scale
            feats.append(h_hla)

        if debug_block_means and self.use_hla:
            with torch.no_grad():
                m_h = (x[:, self.slice_hla].abs().mean().item())
            print(f"[blocks] mean| HLA={m_h:.4f}")

        e_a, e_b, has_a, has_b = self.seq_enc(ids_a, ids_b)
        if self.use_seq_a:
            feats.append(e_a)
        if self.use_seq_b:
            feats.append(e_b)

        h = torch.cat(feats, dim=-1)

        if self.use_cosine_head:
            logits = self.head(h, y) if self.training and y is not None else self.head(h, None)
        else:
            logits = self.head(h)

        if hasattr(self, "allowed_map") and (x_hla is not None):
            allow = (x_hla.unsqueeze(1) * self.allowed_map.unsqueeze(0)).sum(-1) > 0
            no_allow = ~allow.any(dim=-1)
            if no_allow.any():
                allow[no_allow] = True
            logits = logits.masked_fill(~allow, -1e4)

        return logits


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data_dir = Path(args.data)
    X, a_ids, b_ids, data_meta, y = load_featurized_for_pred(data_dir)
    N, D = X.shape
    print(f"[Data] X shape={X.shape}, y={'present' if y is not None else 'None'}")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]
    num_classes = len(meta["label_info"]["labels_vocab"])
    label_vocab = meta["label_info"]["labels_vocab"]
    temperature = float(ckpt.get("temperature", 1.0))

    hp_path = ckpt_path.with_name("hparams_ft.json")
    if hp_path.exists():
        hparams = json.load(open(hp_path, "r"))
        print(f"[HP] loaded hparams_ft.json from {hp_path}")
    else:
        raise FileNotFoundError(f"hparams_ft.json not found next to {ckpt_path}; ")

    model = EpitopeClassifier(
        meta=meta,
        x_dim=D,
        num_classes=num_classes,
        lambda_hla=hparams.get("lambda_hla", 1.0),
        lambda_seq_alpha=hparams.get("lambda_seq", 1.0),
        lambda_seq_beta=hparams.get("lambda_seq", 1.0),
        d_model=hparams.get("d_model", 256),
        use_hla=bool(hparams.get("use_hla", 1)),
        use_seq_a=bool(hparams.get("use_seq_a", 1)),
        use_seq_b=bool(hparams.get("use_seq_b", 1)),
        dropout_tab=hparams.get("dropout_tab", 0.0),
        dropout_seq=hparams.get("dropout_seq", 0.0),
        use_cosine_head=bool(hparams.get("use_cosine_head", 1)),
        margin_m=hparams.get("margin_m", 0.20),
        scale_s=hparams.get("scale_s", 30.0),
        slot_drop=hparams.get("slot_drop", 0.0),
    ).to(device)

    state = ckpt["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    class PredDataset(torch.utils.data.Dataset):
        def __init__(self, X, a_ids, b_ids):
            self.X = X
            self.a_ids = a_ids
            self.b_ids = b_ids

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            return (
                torch.from_numpy(self.X[i]),
                torch.from_numpy(self.a_ids[i]),
                torch.from_numpy(self.b_ids[i]),
                i,
            )

    ds = PredDataset(X, a_ids, b_ids)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, shuffle=False, num_workers=0, drop_last=False
    )

    def apply_temp(logits):
        return logits / max(temperature, 1e-3)

    all_row_idx = []
    all_top1_idx, all_top1_prob = [], []
    all_top5_idx, all_top5_prob = [], []
    all_top10_idx, all_top10_prob = [], []

    with torch.no_grad():
        for xb, aa, bb, ridx in dl:
            xb = xb.to(device)
            aa = aa.to(device)
            bb = bb.to(device)

            logits = model(xb, aa, bb)
            probs = torch.softmax(apply_temp(logits), dim=-1).cpu().numpy()

            p1_idx = probs.argmax(axis=1)
            p1_prob = probs.max(axis=1)
            for k, idx_list, prob_list in [
                (5,  all_top5_idx,  all_top5_prob),
                (10, all_top10_idx, all_top10_prob),
            ]:
                kk = min(k, probs.shape[1])
                tk_idx = np.argsort(-probs, axis=1)[:, :kk]
                tk_prob = np.take_along_axis(probs, tk_idx, axis=1)
                idx_list.extend(tk_idx.tolist())
                prob_list.extend(tk_prob.tolist())

            all_row_idx.extend(ridx.numpy().tolist())
            all_top1_idx.extend(p1_idx.tolist())
            all_top1_prob.extend(p1_prob.tolist())

    pairs_rows = None
    if args.pairs_csv is not None:
        import pandas as pd
        pairs_df = pd.read_csv(args.pairs_csv)
        if len(pairs_df) != N:
            print(f"[Warn] pairs_csv rows {len(pairs_df)} != featurized samples {N}")
        pairs_rows = pairs_df

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "row_idx",
            "top1_idx", "top1_epitope", "top1_prob",
            "top5_idx", "top5_epitopes", "top5_probs",
            "top10_idx", "top10_epitopes", "top10_probs",
        ]
        if pairs_rows is not None:
            header = (
                ["row_idx", "sample", "cdr3_alpha", "cdr3_beta"]
                + header[1:]
            )
        w.writerow(header)

        for i, ridx in enumerate(all_row_idx):
            # top-1
            t1_idx = all_top1_idx[i]
            t1_prob = all_top1_prob[i]
            t1_epi = label_vocab[t1_idx] if 0 <= t1_idx < len(label_vocab) else "<OOR>"

            # top-5 / top-10
            t5_idx  = all_top5_idx[i]
            t5_prob = all_top5_prob[i]
            t5_epi  = [label_vocab[j] if 0 <= j < len(label_vocab) else "<OOR>" for j in t5_idx]

            t10_idx  = all_top10_idx[i]
            t10_prob = all_top10_prob[i]
            t10_epi  = [label_vocab[j] if 0 <= j < len(label_vocab) else "<OOR>" for j in t10_idx]

            base_row = [
                ridx,
                t1_idx, t1_epi, float(t1_prob),
                ";".join(map(str, t5_idx)),
                ";".join(t5_epi),
                ";".join(f"{x:.4f}" for x in t5_prob),
                ";".join(map(str, t10_idx)),
                ";".join(t10_epi),
                ";".join(f"{x:.4f}" for x in t10_prob),
            ]

            if pairs_rows is not None and ridx < len(pairs_rows):
                r = pairs_rows.iloc[ridx]
                prefix = [ridx, r.get("sample", ""), r.get("cdr3_alpha", ""), r.get("cdr3_beta", "")]
                row = prefix + base_row[1:]
            else:
                row = base_row

            w.writerow(row)

    print(f"[Done] saved prediction csv to: {out_path}")


# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--pairs_csv", type=str, default=None)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    main(args)
