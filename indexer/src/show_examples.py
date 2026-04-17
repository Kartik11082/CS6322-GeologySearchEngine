"""Print example data for explanation purposes."""

import json

# 1. Inverted index sample
idx = json.load(open("data/inverted_index.json"))
print("=" * 60)
print("INVERTED INDEX SUMMARY")
print("=" * 60)
print(f"  Total documents (N): {idx['N']}")
print(f"  Average doc length: {idx['avg_dl']:.1f} stems")
print(f"  Total unique terms: {len(idx['index'])}")
print()

# Show postings for a sample term
for term in ["earthquak", "fault", "volcan", "geolog"]:
    postings = idx["index"].get(term, {})
    sample = dict(list(postings.items())[:4])
    print(f'  Term "{term}" -> appears in {len(postings)} docs')
    print(f"    Sample postings: {sample}")
print()

# 2. Doc store sample
ds = json.load(open("data/doc_store.json"))
print("=" * 60)
print("DOC STORE SAMPLE (doc_id=1)")
print("=" * 60)
doc = ds.get("1", {})
if doc:
    for k, v in doc.items():
        print(f"  {k}: {str(v)[:100]}")
else:
    print("  (doc_id=1 not found)")
print()

# 3. Web graph
g = json.load(open("data/web_graph.json"))
print("=" * 60)
print("WEB GRAPH")
print("=" * 60)
print(f"  Nodes: {len(g['nodes'])}")
print(f"  Edges: {len(g['edges'])}")
print(f"  First 3 edges:")
for e in g["edges"][:3]:
    print(f"    doc {e['source']} -> doc {e['target']}")

stats = json.load(open("data/graph_stats.json"))
print(
    f"  Max in-degree: {stats['max_in_degree']} (doc_id={stats['max_in_degree_node']})"
)
print(
    f"  Max out-degree: {stats['max_out_degree']} (doc_id={stats['max_out_degree_node']})"
)
print()

# 4. PageRank top 5
pr = json.load(open("data/pagerank_scores.json"))
top5 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
print("=" * 60)
print("TOP 5 PAGERANK SCORES")
print("=" * 60)
for d, s in top5:
    title = ds.get(d, {}).get("title", "?")
    print(f"  doc_id={d:>4}  score={s:.6f}  {title[:60]}")
print()

# 5. List files in data/
import os

print("=" * 60)
print("FILES IN indexer/data/")
print("=" * 60)
for f in sorted(os.listdir("data")):
    size = os.path.getsize(f"data/{f}")
    print(f"  {f:40s} {size:>12,} bytes")
