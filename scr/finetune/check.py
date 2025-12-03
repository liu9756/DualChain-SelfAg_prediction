import numpy as np
from collections import Counter
import json
from pathlib import Path
import pandas as pd 


y = np.load("/users/PAS2177/liu9756/DualChain-SelfAg_prediction/data/featurized_300_ft/y.npy")
ctr = Counter(y.tolist())

print("总样本数:", len(y))
print("标签数:", len(ctr))

