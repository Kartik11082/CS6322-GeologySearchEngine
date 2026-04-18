import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "indexer" / "src"))

from search import SearchEngine
from core import QueryExpander

def run_experiments():
    print("Loading Search Engine... This might take ~5 seconds.")
    engine = SearchEngine()
    engine.load()
    expander = QueryExpander(engine)
    
    # Load Student 4's benchmark queries
    queries_path = PROJECT_ROOT / "cluster_service" / "benchmarks" / "queries_50.json"
    try:
        with open(queries_path, "r", encoding="utf-8") as f:
            benchmarks = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {queries_path}")
        return
        
    queries_50 = [q["query_text"] for q in benchmarks]
    queries_20 = queries_50[:20] 
    
    report_path = PROJECT_ROOT / "expander" / "expansion_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as out:
        out.write("="*60 + "\n")
        out.write("EXPERIMENT 1: ROCCHIO RELEVANCE FEEDBACK (20 QUERIES)\n")
        out.write("="*60 + "\n\n")
        
        print("Running Rocchio (Simulating Manual Judgments)...")
        for q in queries_20:
            # SIMULATING A USER: We run a quick search and arbitrarily say the top 2 
            # docs are relevant, and the 3rd doc is irrelevant to satisfy the formula.
            # *For your report, you should note these specific document IDs!*
            initial_results = engine.search(q, method="bm25", top_k=3)
            rel_docs = [str(res["doc_id"]) for res in initial_results[:2]]
            irrel_docs = [str(res["doc_id"]) for res in initial_results[2:3]]
            
            expanded = expander.expand_rocchio(q, relevant_doc_ids=rel_docs, irrelevant_doc_ids=irrel_docs)
            
            out.write(f"Original Query : {q}\n")
            out.write(f"Relevant Docs  : {rel_docs}\n")
            out.write(f"Irrelevant Docs: {irrel_docs}\n")
            out.write(f"Expanded Query : {expanded}\n")
            out.write("-" * 40 + "\n")

        out.write("\n" + "="*60 + "\n")
        out.write("EXPERIMENT 2: LOCAL CLUSTERING METHODS (50 QUERIES)\n")
        out.write("="*60 + "\n\n")
        
        print("Running Local Clustering (Association, Scalar, Metric) on 50 queries...")
        for i, q in enumerate(queries_50, 1):
            if i % 10 == 0:
                print(f"Processed {i}/50 queries...")
                
            assoc = expander.expand_association(q, normalized=True)
            scalar = expander.expand_scalar(q)
            metric = expander.expand_metric(q)
            
            out.write(f"Query {i}      : {q}\n")
            out.write(f"Association  : {assoc}\n")
            out.write(f"Scalar       : {scalar}\n")
            out.write(f"Metric       : {metric}\n")
            out.write("-" * 60 + "\n")

    print(f"\n✅ Experiments complete! Results saved to: {report_path}")

if __name__ == "__main__":
    run_experiments()