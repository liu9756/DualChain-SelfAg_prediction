import numpy as np
from pathlib import Path

RA_DIR = Path("/users/PAS2177/liu9756/DualChain-SelfAg_prediction/data/featurized_ft")

y = np.load(RA_DIR / "y.npy") 
N = y.shape[0]
labels = np.unique(y)

rng = np.random.default_rng(42)

idx_train, idx_val, idx_test = [], [], []

for lab in labels:
    idxs = np.where(y == lab)[0]
    idxs = rng.permutation(idxs)  
    n = len(idxs)

    if n == 1:
        idx_train.extend(idxs)
    elif n == 2:
        idx_train.append(idxs[0])
        idx_val.append(idxs[1])
    elif n == 3:
        idx_train.append(idxs[0])
        idx_val.append(idxs[1])
        idx_test.append(idxs[2])
    else:
        n_train = max(1, int(round(n * 0.7)))
        n_val   = max(1, int(round(n * 0.15)))
        n_test  = n - n_train - n_val

        if n_test == 0 and n_val > 1:
            n_test = 1
            n_val -= 1

        if n_train + n_val + n_test != n:
            idx_train.append(idxs[0])
            if n > 1:
                idx_val.extend(idxs[1:])
        else:
            idx_train.extend(idxs[:n_train])
            idx_val.extend(idxs[n_train:n_train + n_val])
            idx_test.extend(idxs[n_train + n_val:])

idx_train = np.array(idx_train, dtype=int)
idx_val   = np.array(idx_val,   dtype=int)
idx_test  = np.array(idx_test,  dtype=int)

rng.shuffle(idx_train)
rng.shuffle(idx_val)
rng.shuffle(idx_test)

print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
print(f"Total unique idx: {len(set(idx_train) | set(idx_val) | set(idx_test))}")

np.save(RA_DIR / "idx_train.npy", idx_train)
np.save(RA_DIR / "idx_val.npy",   idx_val)
np.save(RA_DIR / "idx_test.npy",  idx_test)

from collections import Counter

print("Train label counts:", Counter(y[idx_train]))
print("Val label counts:",   Counter(y[idx_val]))
print("Test label counts:",  Counter(y[idx_test]))
