"""Expansion evaluation script.

Run with spaCy available (morphological lemmas in reports), e.g.::

    conda run -n nlp python expander/evaluate.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "indexer" / "src"))

from search import SearchEngine
from core import QueryExpander

# 20 Queries for Rocchio Relevance Feedback (Geology Domain)
ROCCHIO_QUERIES = [
    "volcano",
    "earthquake",
    "fossil",
    "minerals",
    "magma composition",
    "tectonic plates",
    "sedimentary rock",
    "carbon dating",
    "erthsquake",
    "volcnos",
    "fosiils",
    "ignous",
    "yellowstone caldera",
    "san andreas fault",
    "hawaiian pahoehoe",
    "mount vesuvius",
    "software engineering",
    "artificial intelligence",
    "stock market trends",
    "football rules"
]

# 50 Queries for Pseudo-Relevance Feedback (PRF)
# Includes the 20 Rocchio queries + 30 geology-specific queries
PRF_QUERIES = ROCCHIO_QUERIES + [
    "pyroclastic flow danger",
    "richter scale magnitude",
    "foreshock and aftershock sequence",
    "subduction zone tremor",
    "epicenter and hypocenter distance",
    "volcanic tephra and lahar",
    "caldera collapse mechanism",
    "seismic wave velocity",
    "quartz cleavage and luster",
    "metamorphic gneiss formation",
    "silicate and carbonate minerals",
    "feldspar mica olivine",
    "basalt vs granite",
    "igneous intrusion",
    "shale limestone sandstone",
    "strata unconformity",
    "karst landform erosion",
    "alluvial fan deposition",
    "fluvial weathering process",
    "glacial moraine deposits",
    "stratigraphy facies",
    "rock diagenesis",
    "mesozoic extinction event",
    "trilobite specimen identification",
    "ammonite fossil outcrop",
    "paleozoic evolution",
    "groundwater aquifer permeability",
    "watershed porosity",
    "artesian spring water",
    "soil heavy metal contamination"
]

def write_top10_results(out, engine: SearchEngine, query_text: str, label: str):
    """Write top-10 BM25 doc_ids and URLs for a query string."""
    results = engine.search(query_text, method="bm25", top_k=10)
    out.write(f"{label} (Top 10 BM25)\n")
    if not results:
        out.write("  [no results]\n")
        return
    for r in results:
        out.write(f"  {r['rank']:>2}. doc_id={r['doc_id']} | {r.get('url', '')}\n")

def run_experiments():
    print("Loading Search Engine... This might take ~5 seconds.")
    engine = SearchEngine()
    engine.load()
    expander = QueryExpander(engine)
    
    report_path = PROJECT_ROOT / "expander" / "expansion_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as out:
        out.write("="*70 + "\n")
        out.write("GEOLOGY SEARCH ENGINE - QUERY EXPANSION EVALUATION REPORT\n")
        out.write("="*70 + "\n\n")
        
        # ================================================================
        # EXPERIMENT 1: ROCCHIO
        # ================================================================
        out.write("="*60 + "\n")
        out.write("EXPERIMENT 1: ROCCHIO RELEVANCE FEEDBACK (20 QUERIES)\n")
        out.write("="*60 + "\n\n")
        
        print("Running Rocchio Evaluation against 20 benchmark queries...")
        for q in ROCCHIO_QUERIES:
            normalized_q = expander.normalize_query(q)
            # Simulate manual relevance judgments by taking top 2 as relevant, 3rd as irrelevant
            initial_results = engine.search(normalized_q, method="bm25", top_k=3)
            rel_docs = [str(res["doc_id"]) for res in initial_results[:2]]
            irrel_docs = [str(res["doc_id"]) for res in initial_results[2:3]]
            
            expanded = expander.expand_rocchio(q, relevant_doc_ids=rel_docs, irrelevant_doc_ids=irrel_docs)
            
            out.write(f"Original Query   : {q}\n")
            out.write(f"Normalized Query : {normalized_q}\n")
            out.write(f"Relevant Docs    : {rel_docs}\n")
            out.write(f"Irrelevant Docs  : {irrel_docs}\n")
            out.write(f"Expanded Query   : {expanded}\n")
            write_top10_results(out, engine, normalized_q, "Original Query Results")
            write_top10_results(out, engine, expanded, "Expanded Query Results")
            out.write("-" * 40 + "\n")

        # ================================================================
        # EXPERIMENT 2: LOCAL CLUSTERING
        # ================================================================
        out.write("\n" + "="*60 + "\n")
        out.write("EXPERIMENT 2: LOCAL CLUSTERING METHODS (50 QUERIES)\n")
        out.write("="*60 + "\n\n")
        
        print(f"Running Local Clustering on {len(PRF_QUERIES)} benchmark queries...")
        for i, q in enumerate(PRF_QUERIES, 1):
            if i % 10 == 0:
                print(f"Processed {i}/{len(PRF_QUERIES)} queries...")
                
            assoc = expander.expand_association(q, normalized=True)
            scalar = expander.expand_scalar(q)
            metric = expander.expand_metric(q)
            
            out.write(f"Query {i:<2}         : {q}\n")
            out.write(f"Association      : {assoc}\n")
            out.write(f"Scalar           : {scalar}\n")
            out.write(f"Metric           : {metric}\n")
            write_top10_results(out, engine, assoc, "Association Results")
            write_top10_results(out, engine, scalar, "Scalar Results")
            write_top10_results(out, engine, metric, "Metric Results")
            out.write("-" * 60 + "\n")

    print(f"\n✅ Experiments complete! Results saved to: {report_path}")

if __name__ == "__main__":
    run_experiments()