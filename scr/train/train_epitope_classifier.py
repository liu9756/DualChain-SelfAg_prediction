# -*- coding: utf-8 -*-
import argparse, json, os, random, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

# Utils
def set_seed(sd=42):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_featurized(d):
    d = Path(d)
    X = np.load(d/"X.npy").astype(np.float32)
    y = np.load(d/"y.npy").astype(np.int64)
    a_ids = np.load(d/"cdr3a_ids.npy")
    b_ids = np.load(d/"cdr3b_ids.npy")
    meta = json.load(open(d/"meta.json","r"))
    return X, y, a_ids, b_ids, meta

def class_weights_from_y(y, num_classes=None):
    uniques, counts = np.unique(y, return_counts=True)
    C = int(num_classes) if num_classes is not None else int(uniques.max()+1)
    w = np.zeros(C, dtype=np.float32)
    if len(uniques) > 0:
        w[uniques] = 1.0 / counts
        w *= (len(y) / len(uniques))
    return torch.tensor(w, dtype=torch.float32)

def topk_acc(y_true, y_score, k=5, num_classes=None):
    return top_k_accuracy_score(
        y_true, y_score, k=min(k, y_score.shape[1]),
        labels=np.arange(num_classes if num_classes is not None else y_score.shape[1])
    )

# Dataset
class TcrDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, a_ids, b_ids, row_idx):
        self.X = X
        self.y = y
        self.a = a_ids
        self.b = b_ids
        self.row_idx = np.asarray(row_idx)

        n = len(self.y)
        assert self.X.shape[0] == n and self.a.shape[0] == n and self.b.shape[0] == n, \
            f"Inconsistent lengths: X={self.X.shape[0]}, a={self.a.shape[0]}, b={self.b.shape[0]}, y={n}"
        assert len(self.row_idx) == n, \
            f"row_idx length {len(self.row_idx)} != {n}"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),     
            torch.from_numpy(self.a[i]),     
            torch.from_numpy(self.b[i]),      
            torch.tensor(self.y[i], dtype=torch.long),
            torch.tensor(int(self.row_idx[i]), dtype=torch.long)  
        )

# Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.2, weight=None, reduction: str = "mean",
                 label_smoothing: float = 0.02, num_classes: int = None):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction
        self.eps = float(label_smoothing)
        self.num_classes = num_classes
    def forward(self, logits, target):
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp()
        pt = prob.gather(1, target.view(-1, 1)).squeeze(1)
        logpt = log_prob.gather(1, target.view(-1, 1)).squeeze(1)
        loss = -(1 - pt).pow(self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight[target]
        if self.eps > 0.0 and self.num_classes is not None:
            uniform_loss = -log_prob.mean(dim=-1)
            loss = (1 - self.eps) * loss + self.eps * uniform_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


# Encoders & Classifier
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)
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
        v_b, has_b = self._encode_slots(ids_beta,  chain_type_id=1)  

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

        idx = torch.arange(H, device=device).unsqueeze(0).expand(B, H)  
        h = self.emb(idx)                                              
        h = self.pos(h)                                            

        pad_mask = (x_hla <= 0)                                       
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
        self.s = float(s); self.m = float(m)
    def forward(self, x, y=None):
        x_norm = F.normalize(x, dim=-1)
        W_norm = F.normalize(self.W, dim=-1)
        cos = torch.matmul(x_norm, W_norm.t())
        if y is None:
            return self.s * cos
        cos_y = cos.gather(1, y.view(-1,1)).squeeze(1)
        cos_y_m = cos_y - self.m
        logits = cos.clone()
        logits.scatter_(1, y.view(-1,1), cos_y_m.unsqueeze(1))
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

        vocab_size = len(meta["cdr3_token_vocab"])
        pad_id = meta["cdr3_token_vocab"].get("<PAD>", 0)
        '''self.seq_enc = SharedSeqEncoder(
            vocab_size=vocab_size, d_model=d_model, nlayers=2, nhead=4, pad_id=pad_id, dropout=dropout_seq
        )'''
        self.hla_dim = hla_dim
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

# Temperature scaling
class _TempScaler(nn.Module):
    def __init__(self): super().__init__(); self.t = nn.Parameter(torch.ones(1))
    def forward(self, logits): return logits / self.t.clamp(min=1e-3)

def fit_temperature(logits_val, y_val, max_iter=1000, lr=1e-2):
    device = logits_val.device
    scaler = _TempScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    def _closure():
        opt.zero_grad()
        loss = F.cross_entropy(scaler(logits_val), y_val)
        loss.backward()
        return loss
    opt.step(_closure)
    with torch.no_grad():
        T = float(scaler.t.clamp(min=1e-3).cpu().item())
    return T

# EMA
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

# Train
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    X, y, a_ids, b_ids, meta = load_featurized(args.data)
    num_classes = len(meta["label_info"]["labels_vocab"])
    x_dim = X.shape[1]
    meta["hla_vocab_size"] = int(meta.get("hla_vocab_size", X.shape[1]))

    rng = np.random.RandomState(args.seed)
    N = len(y)
    all_idx = np.arange(N)

    groups = None
    for key in ["groups", "pair_ids", "donor_ids", "clonotype_ids", "clone_ids"]:
        if key in meta and len(meta[key]) == N:
            groups = np.asarray(meta[key])
            break

    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        tr_idx, te_idx = next(gss.split(np.zeros(N), y, groups))
    else:
        cnt = Counter(y.tolist())
        labels_many   = {lab for lab, c in cnt.items() if c >= 2}
        labels_single = {lab for lab, c in cnt.items() if c == 1}
        idx_many   = np.array([i for i in all_idx if y[i] in labels_many], dtype=int)
        idx_single = np.array([i for i in all_idx if y[i] in labels_single], dtype=int)
        if len(idx_many) == 0:
            tr_idx, te_idx = train_test_split(all_idx, test_size=0.2, random_state=args.seed, shuffle=True, stratify=None)
        else:
            tr_many, te_many = train_test_split(idx_many, test_size=0.2, random_state=args.seed, stratify=y[idx_many])
            tr_idx = np.concatenate([tr_many, idx_single], axis=0)
            te_idx = te_many
            rng.shuffle(tr_idx)

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    a_tr, a_te = a_ids[tr_idx], a_ids[te_idx]
    b_tr, b_te = b_ids[tr_idx], b_ids[te_idx]

    ds_tr_base = TcrDataset(X_tr, y_tr, a_tr, b_tr, tr_idx)
    ds_te      = TcrDataset(X_te, y_te, a_te, b_te, te_idx)

    val_ratio = 0.2
    tr_base_idx = np.arange(len(y_tr))

    if groups is not None:
        inner_groups = groups[tr_idx]
        gss_in = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=args.seed)
        v_idx, t_idx = next(gss_in.split(np.zeros(len(y_tr)), y_tr, inner_groups))
    else:
        cnt_tr = Counter(y_tr.tolist())
        labs_many_tr   = {lab for lab, c in cnt_tr.items() if c >= 2}
        labs_single_tr = {lab for lab, c in cnt_tr.items() if c == 1}
        idx_many_tr   = np.array([i for i in tr_base_idx if y_tr[i] in labs_many_tr], dtype=int)
        idx_single_tr = np.array([i for i in tr_base_idx if y_tr[i] in labs_single_tr], dtype=int)
        if len(idx_many_tr) == 0:
            v_idx, t_idx = train_test_split(tr_base_idx, test_size=1 - val_ratio, random_state=args.seed, shuffle=True, stratify=None)
        else:
            v_many, t_many = train_test_split(idx_many_tr, test_size=1 - val_ratio, random_state=args.seed, stratify=y_tr[idx_many_tr])
            t_idx = np.concatenate([t_many, idx_single_tr], axis=0)
            v_idx = v_many
            rng.shuffle(t_idx)

    ds_val = torch.utils.data.Subset(ds_tr_base, v_idx)
    ds_tr2 = torch.utils.data.Subset(ds_tr_base, t_idx)

    dl_workers = 0 if args.cpu else 1
    if args.use_sampler:
        y_tr2 = y_tr[t_idx]
        class_counts = np.bincount(y_tr2, minlength=num_classes)
        class_counts[class_counts == 0] = 1
        sample_weights = 1.0 / np.sqrt(class_counts[y_tr2])
        sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)  # 兼容性更好
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_tr2), replacement=True)
        dl_tr  = DataLoader(ds_tr2, batch_size=args.bs, sampler=sampler, num_workers=dl_workers, drop_last=False)
    else:
        dl_tr  = DataLoader(ds_tr2, batch_size=args.bs, shuffle=True, num_workers=dl_workers, drop_last=False)

    dl_val = DataLoader(ds_val, batch_size=args.bs, shuffle=False, num_workers=dl_workers, drop_last=False)
    dl_te  = DataLoader(ds_te,  batch_size=args.bs, shuffle=False, num_workers=dl_workers, drop_last=False)

    model = EpitopeClassifier(
        meta=meta, x_dim=x_dim, num_classes=num_classes,
        lambda_hla=args.lambda_hla, lambda_seq_alpha=args.lambda_seq, lambda_seq_beta=args.lambda_seq,
        d_model=args.d_model,
        use_hla=bool(args.use_hla), use_seq_a=bool(args.use_seq_a), use_seq_b=bool(args.use_seq_b),
        use_cosine_head=bool(args.use_cosine_head), margin_m=args.margin_m, scale_s=args.scale_s
    ).to(device)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available() and bool(args.amp) and not args.cpu))
    eff_ls = float(args.label_smoothing)
    if bool(args.use_cosine_head):
        if eff_ls != 0.0:
            print("[Note] use_cosine_head enabled -> forcing label_smoothing=0.0")
        eff_ls = 0.0
    if bool(args.use_focal) and eff_ls > 0.02:
        eff_ls = 0.02

    ce_weight = None if args.use_sampler else class_weights_from_y(y_tr[t_idx], num_classes).to(device)
    if args.use_focal:
        criterion = FocalLoss(gamma=args.gamma, weight=ce_weight,
                              label_smoothing=eff_ls, num_classes=num_classes)
    else:
        criterion = nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=eff_ls)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.sched == "cosine":
        total_steps = max(1, int(len(dl_tr) * args.epochs))
        warmup = max(1, int(total_steps * args.warmup))
        def lr_lambda(step):
            if step < warmup:
                return float(step + 1) / float(warmup)
            progress = (step - warmup) / max(1, (total_steps - warmup))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best_acc.pt"
    best_acc, best_state = 0.0, None
    best_val_logits, best_val_y = None, None
    no_improve = 0
    global_step = 0

    with open(out / "hparams.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        first_batch_printed = False
        for xb, aa, bb, yb, _ in dl_tr:
            xb, aa, bb, yb = xb.to(device), aa.to(device), bb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=scaler.is_enabled()):
                logits = model(xb, aa, bb, y=yb if model.use_cosine_head else None,
                               debug_block_means=(not first_batch_printed and epoch == 1))
                first_batch_printed = True
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update()
            if ema: ema.update(model)
            loss_sum += float(loss.item()) * len(yb)
            if args.sched == "cosine":
                sched.step(); global_step += 1

        #### validate ###########
        model_eval = model
        if ema:
            model_eval = EpitopeClassifier(
                meta=meta, x_dim=x_dim, num_classes=num_classes,
                lambda_hla=args.lambda_hla, lambda_seq_alpha=args.lambda_seq, lambda_seq_beta=args.lambda_seq,
                d_model=args.d_model,
                use_hla=bool(args.use_hla), use_seq_a=bool(args.use_seq_a), use_seq_b=bool(args.use_seq_b),
                use_cosine_head=bool(args.use_cosine_head), margin_m=args.margin_m, scale_s=args.scale_s
            ).to(device)
            ema.apply_to(model_eval)
            model_eval.eval()

        y_true, y_hat, val_loss, all_logits = [], [], 0.0, []
        with torch.no_grad():
            for xb, aa, bb, yb, _ in dl_val:
                xb, aa, bb, yb = xb.to(device), aa.to(device), bb.to(device), yb.to(device)
                logits = model_eval(xb, aa, bb)  
                l = criterion(logits, yb)
                val_loss += float(l.item()) * len(yb)
                all_logits.append(logits)
                y_true.append(yb.cpu().numpy())
                y_hat.append(torch.softmax(logits, dim=-1).cpu().numpy())
        y_true = np.concatenate(y_true, axis=0)
        y_hat  = np.concatenate(y_hat,  axis=0)
        all_logits = torch.cat(all_logits, dim=0)

        acc  = accuracy_score(y_true, y_hat.argmax(1))
        f1m  = f1_score(y_true, y_hat.argmax(1), average="macro")
        top1 = topk_acc(y_true, y_hat, k=1, num_classes=num_classes)
        top5 = topk_acc(y_true, y_hat, k=5, num_classes=num_classes)
        print(f"[Epoch {epoch:03d}] train_loss={loss_sum/len(ds_tr2):.4f}  "
              f"val_loss={val_loss/len(ds_val):.4f}  ACC={acc:.3f}  F1m={f1m:.3f}  Top5={top5:.3f}  Top1={top1:.3f}")

        improved = acc > best_acc + 1e-6
        if improved:
            best_acc = acc
            best_state = {k: v.detach().cpu() for k, v in (model_eval if ema else model).state_dict().items()}
            best_val_logits = all_logits.detach().cpu()
            best_val_y = torch.tensor(y_true, dtype=torch.long)
            torch.save({"meta": meta, "state_dict": best_state, "label_info": meta["label_info"], "temperature": 1.0,
                        "epoch": epoch, "val_acc": float(best_acc), "val_top1": float(top1)}, best_path)
            print(f"[Checkpoint] New best ACC={best_acc:.4f} at epoch {epoch}. Saved to: {best_path}")
            no_improve = 0
        else:
            no_improve += 1

        if args.sched == "plateau":
            sched.step(acc)

        if args.early_stop and no_improve >= args.early_stop_patience:
            print(f"Early stop at epoch {epoch} (no improve {no_improve}/{args.early_stop_patience}).")
            break

    # ###########test #################
    if best_state is not None:
        if ema: ema.apply_to(model)
        model.load_state_dict(best_state, strict=True)

    model.eval()
    temperature = 1.0
    if args.temp_scale and best_val_logits is not None:
        temperature = fit_temperature(best_val_logits.to(device), best_val_y.to(device))
        print(f"[Temp scaling] learned T = {temperature:.3f}")

    def apply_temp(logits): return logits / max(temperature, 1e-3)

    y_true, y_hat = [], []
    rows, top1_pred, top1_prob, top5_pred, top5_prob = [], [], [], [], []
    with torch.no_grad():
        for xb, aa, bb, yb, ridx in dl_te:
            xb, aa, bb = xb.to(device), aa.to(device), bb.to(device)
            logits = model(xb, aa, bb)
            probs = torch.softmax(apply_temp(logits), dim=-1).cpu().numpy()
            y_true.append(yb.numpy()); y_hat.append(probs)
            p1 = probs.argmax(1); s1 = probs.max(1)
            tk = min(5, probs.shape[1])
            topk_idx = np.argsort(-probs, axis=1)[:, :tk]
            topk_p   = np.take_along_axis(probs, topk_idx, axis=1)
            rows.extend(ridx.numpy().tolist())
            top1_pred.extend(p1.tolist()); top1_prob.extend(s1.tolist())
            top5_pred.extend(topk_idx.tolist()); top5_prob.extend(topk_p.tolist())

    y_true = np.concatenate(y_true, axis=0)
    y_hat  = np.concatenate(y_hat,  axis=0)
    acc   = accuracy_score(y_true, y_hat.argmax(1))
    f1m   = f1_score(y_true, y_hat.argmax(1), average="macro")
    top5  = topk_acc(y_true, y_hat, k=5,  num_classes=num_classes)
    top10 = topk_acc(y_true, y_hat, k=10, num_classes=num_classes)
    print(f"\n=== TEST ===  ACC={acc:.3f}  F1m={f1m:.3f}  Top5={top5:.3f}  Top10={top10:.3f}")

    if best_state is not None:
        torch.save({"meta": meta, "state_dict": best_state, "label_info": meta["label_info"],
                    "temperature": float(temperature), "val_acc": float(best_acc)}, best_path)
        print(f"[Best Model Saved] {best_path} (includes temperature).")

    csv_path = Path(args.outdir) / "test_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_idx","y_true","top1_pred","top1_prob","top5_pred","top5_prob"])
        for i in range(len(rows)):
            w.writerow([rows[i], int(y_true[i]), int(top1_pred[i]), float(top1_prob[i]),
                        ";".join(map(str, top5_pred[i])), ";".join(f"{x:.4f}" for x in top5_prob[i])])
    print("Saved:", csv_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="path to featurized dir")
    ap.add_argument("--outdir", type=str, default="./runs/epitope_cls")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--lambda_hla", type=float, default=1)
    ap.add_argument("--lambda_seq", type=float, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")

    # loss
    ap.add_argument("--use_sampler", type=int, default=1)
    ap.add_argument("--use_focal", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=2)
    ap.add_argument("--label_smoothing", type=float, default=0.01)

    # ablition
    ap.add_argument("--use_hla", type=int, default=1)
    ap.add_argument("--use_seq_a", type=int, default=1)
    ap.add_argument("--use_seq_b", type=int, default=1)


    ap.add_argument("--sched", type=str, default="cosine", choices=["cosine","plateau"])
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--early_stop", type=int, default=1)
    ap.add_argument("--early_stop_patience", type=int, default=1000)

    ap.add_argument("--use_cosine_head", type=int, default=1)
    ap.add_argument("--margin_m", type=float, default=0.20)
    ap.add_argument("--scale_s", type=float, default=30.0)

    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--temp_scale", type=int, default=1)

    # ema
    ap.add_argument("--ema", type=int, default=1)
    ap.add_argument("--ema_decay", type=float, default=0.999)

    args = ap.parse_args()
    run(args)
