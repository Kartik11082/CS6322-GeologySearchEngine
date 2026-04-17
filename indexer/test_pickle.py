import time
import pickle
import sys
from pathlib import Path

data_dir = Path("data")
with open(data_dir / "inverted_index.json", "r") as f:
    pass

import json
print("Reading JSON...")
t0 = time.time()
with open(data_dir / "inverted_index.json", "r", encoding="utf-8") as f:
    obj = json.load(f)
t1 = time.time()
print(f"JSON Load time: {t1 - t0:.2f}s")

print("Dumping Pickle...")
t0 = time.time()
with open(data_dir / "inverted_index.pkl", "wb") as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
t1 = time.time()
print(f"Pickle dump time: {t1 - t0:.2f}s")

print("Reading Pickle...")
t0 = time.time()
with open(data_dir / "inverted_index.pkl", "rb") as f:
    obj2 = pickle.load(f)
t1 = time.time()
print(f"Pickle Read time: {t1 - t0:.2f}s")
