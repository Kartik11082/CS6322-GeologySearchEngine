import json
from collections import Counter
from urllib.parse import urlparse

with open("../crawled_data/pages.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

domains = Counter(urlparse(p["url"]).netloc for p in data)

print(f"Total pages    : {len(data):,}")
print(f"Unique URLs    : {len(set(p['url'] for p in data)):,}")
print(f"Unique domains : {len(domains):,}")
print("Top 20 domains  :")
for d, n in domains.most_common(20):
    print(f"  {n:6,}  {d}")
