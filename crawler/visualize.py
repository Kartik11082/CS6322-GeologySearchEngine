import csv
import networkx as nx
from pyvis.network import Network
from urllib.parse import urlparse
from collections import Counter


# ---------------- Load Graph ----------------
def load_graph(file_path):
    G = nx.DiGraph()
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["src_url"].strip()
            dst = row["dst_url"].strip()
            if src and dst:
                G.add_edge(src, dst)
    return G


# ---------------- Domain Helper ----------------
def get_domain(url):
    try:
        return urlparse(url).netloc
    except:
        return "unknown"


# ---------------- Filter Graph ----------------
def filter_graph(G, min_degree=5, max_nodes=800):
    nodes = [n for n in G.nodes if G.degree(n) >= min_degree]
    G = G.subgraph(nodes).copy()
    if len(G.nodes) > max_nodes:
        nodes = list(G.nodes)[:max_nodes]
        G = G.subgraph(nodes).copy()
    return G


# ---------------- Analyze Graph ----------------
def analyze_graph(G):
    print("Total nodes:", len(G.nodes))
    print("Total edges:", len(G.edges))

    # Top hubs (out-degree)
    top_hubs = sorted(G.out_degree, key=lambda x: x[1], reverse=True)[:10]
    print("\nTop hubs (pages linking out a lot):")
    for url, deg in top_hubs:
        print(deg, url)

    # Top authorities (in-degree)
    top_auth = sorted(G.in_degree, key=lambda x: x[1], reverse=True)[:10]
    print("\nTop authorities (pages linked a lot):")
    for url, deg in top_auth:
        print(deg, url)

    # PageRank
    pr = nx.pagerank(G)
    top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop PageRank pages:")
    for url, score in top_pr:
        print(f"{score:.4f}", url)

    # Dead ends
    dead_ends = [n for n in G.nodes if G.out_degree(n) == 0]
    print("\nDead-end pages:", len(dead_ends))

    # Isolated nodes
    isolated = list(nx.isolates(G))
    print("Isolated nodes:", len(isolated))

    # Domain stats
    domains = [get_domain(n) for n in G.nodes]
    domain_counts = Counter(domains)
    print("\nTop 10 domains:")
    for d, c in domain_counts.most_common(10):
        print(c, d)


# ---------------- Visualize ----------------
def visualize_graph(G, output_file="graph.html"):
    net = Network(height="800px", width="100%", directed=True)
    net.barnes_hut()  # better layout
    for node in G.nodes:
        domain = get_domain(node)
        net.add_node(
            node, label=domain, title=node, value=G.degree(node)  # URL on hover
        )
    for src, dst in G.edges:
        net.add_edge(src, dst)

    # Write HTML and open in browser
    net.write_html(output_file, open_browser=True)


# ---------------- Main ----------------
def main():
    file_path = "./output/web_graph.csv"  # your CSV file
    G = load_graph(file_path)

    print("=== Original Graph Stats ===")
    analyze_graph(G)

    # Filter graph for visualization
    G_filtered = filter_graph(G, min_degree=5, max_nodes=800)
    print("\n=== Filtered Graph Stats ===")
    analyze_graph(G_filtered)

    # Visualize filtered graph
    visualize_graph(G_filtered)


if __name__ == "__main__":
    main()
