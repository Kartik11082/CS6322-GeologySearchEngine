import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from index import load_index
from search import SearchEngine

t0 = time.time()
print("Loading engine...")
engine = SearchEngine()
engine.load()
t1 = time.time()
print(f"Load time: {t1 - t0:.2f}s")

print("Searching...")
res = engine.search("geology")
t2 = time.time()
print(f"Search time: {t2 - t1:.4f}s")
