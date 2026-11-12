"""Compare actual retrieval variants without Redis or provider calls."""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.rag.retrieval import KnowledgeBaseManager, BM25Retriever, DenseRetriever, HybridRetriever
from backend.rag.reranker import CrossEncoderReranker, select_passages


def run():
    manager = KnowledgeBaseManager()
    bm25 = BM25Retriever(manager.chunks)
    dense = DenseRetriever(manager.chunks)
    hybrid = HybridRetriever(bm25, dense)
    reranker = CrossEncoderReranker()
    if not reranker.is_available:
        raise RuntimeError("Reranker is unavailable; refusing to label fallback results as reranked.")
    cases = [item for item in json.loads((ROOT / "benchmarks/evaluation_dataset.json").read_text()) if item["category"] != "unsupported"]
    results = {name: [] for name in ("bm25", "dense", "hybrid", "hybrid_reranked", "hybrid_reranked_with_anchor")}
    for item in cases:
        query = item["question"]
        expected = set(item["expected_chunk_ids"])
        for name in results:
            start = time.perf_counter()
            if name in ("bm25", "dense"):
                retriever = bm25 if name == "bm25" else dense
                chunks = [chunk for chunk, score in retriever.search(query, top_k=4) if name == "dense" or score > 0]
            else:
                candidates = hybrid.search(query, candidate_pool_size=50, final_top_k=50)
                if name.startswith("hybrid_reranked"):
                    ranked = reranker.rerank(query, candidates)
                    selected = select_passages(candidates, ranked, 4) if name.endswith("with_anchor") else ranked[:4]
                else:
                    selected = candidates[:4]
                if name.startswith("hybrid_reranked") and not all("rerank_score" in p for p in selected):
                    raise RuntimeError("Reranking failed during comparison.")
                chunks = [p["chunk"] for p in selected]
            recall = len(expected & {c.chunk_id for c in chunks}) / len(expected)
            results[name].append({"id": item["id"], "recall_at_4": recall, "latency_ms": (time.perf_counter() - start) * 1000})
    report = {"corpus_chunks": len(manager.chunks), "cases": len(cases), "results": results,
              "summary": {name: {"recall_at_4_pct": round(100 * statistics.mean(r["recall_at_4"] for r in rows), 2),
                                  "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in rows), 2)}
                          for name, rows in results.items()},
              "limitations": "Small curated corpus. Sequential local timings exclude model startup, generation, Redis and HTTP. Not a production load test."}
    path = ROOT / "benchmarks/retrieval_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    run()
