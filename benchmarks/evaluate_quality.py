"""
benchmarks/evaluate_quality.py
RAG Answer-Quality Evaluation Suite aligned with standard RAG metrics (RAGAS framework).

Evaluates 40 customer support benchmark questions across:
1. Retrieval Recall@K: Did the hybrid retriever & reranker fetch the expected chunks/pages?
2. Citation Accuracy: Are all citations traced, validated, and aligned with expected sources?
3. Faithfulness / Groundedness: Does the model refrain from hallucinating on unsupported questions?
4. Latency Distribution: Cold miss execution performance.
"""

import json
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag.pipeline import CustomerSupportRAGPipeline, SupportRAGResponse


def run_quality_evaluation():
    dataset_path = Path(__file__).resolve().parent / "evaluation_dataset.json"
    if not dataset_path.exists():
        print(f"Error: Evaluation dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("\n" + "=" * 85)
    print("      NOVACART CUSTOMER SUPPORT RAG — ANSWER QUALITY EVALUATION SUITE")
    print(f"      Evaluating {len(dataset)} Questions (Exact Facts · Cross-Doc · Unsupported)")
    print("=" * 85 + "\n")

    from backend.rag.generator import CustomerSupportGenerator
    generator = None
    if "--live-api" not in sys.argv:
        # Fast deterministic grounded generator without network timeouts
        generator = CustomerSupportGenerator(api_key="")

    pipeline = CustomerSupportRAGPipeline(generator=generator)

    results = []
    category_metrics = {
        "exact_fact": {"count": 0, "recall": [], "citation_acc": [], "supportedness": [], "latencies": []},
        "cross_document": {"count": 0, "recall": [], "citation_acc": [], "supportedness": [], "latencies": []},
        "unsupported": {"count": 0, "refusal_correct": [], "citation_clean": [], "latencies": []},
    }

    start_eval_time = time.time()

    for idx, item in enumerate(dataset, 1):
        q_id = item["id"]
        cat = item["category"]
        question = item["question"]
        expected_docs = set(item.get("expected_doc_ids", []))
        expected_chunks = set(item.get("expected_chunk_ids", []))
        expected_pages = set(item.get("expected_pages", []))

        # Always evaluate cold generation (skip cache)
        resp: SupportRAGResponse = pipeline.answer_question(question, skip_cache=True)

        retrieved_chunk_ids = {c.get("chunk_id", "") for c in resp.retrieved_context}
        retrieved_doc_ids = {c.get("doc_id", "") for c in resp.retrieved_context}

        cited_chunk_ids = {c.get("chunk_id", "") for c in resp.citations}
        cited_doc_ids = {c.get("doc_id", "") for c in resp.citations}

        # 1. Retrieval Recall: Exact chunk recall when expected chunk IDs exist; fallback to doc IDs
        if cat == "unsupported":
            retrieval_recall = 1.0  # N/A for unsupported
        elif expected_chunks:
            hits = len(expected_chunks.intersection(retrieved_chunk_ids))
            retrieval_recall = min(1.0, hits / len(expected_chunks))
        elif expected_docs:
            hits = len(expected_docs.intersection(retrieved_doc_ids))
            retrieval_recall = min(1.0, hits / len(expected_docs))
        else:
            retrieval_recall = 1.0

        # 2. Citation Accuracy: Score citations against expected chunk and page
        citation_valid = resp.citation_validation_passed
        if cat == "unsupported":
            # For unsupported, citation accuracy is 1.0 if it made NO false citations
            citation_accuracy = 1.0 if len(resp.citations) == 0 else 0.0
        elif expected_chunks:
            chunk_hits = 0
            for c in resp.citations:
                cid = c.get("chunk_id", "")
                pnum = c.get("page_number", 1)
                if cid in expected_chunks:
                    if not expected_pages or pnum in expected_pages:
                        chunk_hits += 1
            citation_accuracy = min(1.0, chunk_hits / len(expected_chunks))
        elif expected_docs:
            doc_overlap = len(expected_docs.intersection(cited_doc_ids))
            citation_accuracy = (doc_overlap / len(expected_docs)) if expected_docs else 1.0
        else:
            citation_accuracy = 1.0

        # 3. Supportedness (formerly faithfulness check)
        if cat == "unsupported":
            # Correct if system refused and flagged is_supported=False
            supportedness = 1.0 if not resp.is_supported else 0.0
            category_metrics["unsupported"]["count"] += 1
            category_metrics["unsupported"]["refusal_correct"].append(supportedness)
            category_metrics["unsupported"]["citation_clean"].append(citation_accuracy)
            category_metrics["unsupported"]["latencies"].append(resp.latency_ms)
        else:
            supportedness = 1.0 if (resp.is_supported and len(resp.citations) > 0) else 0.0

            # Answer-level check for EF-08: require answer to contain overnight cutoff (1:00 PM EST) with correct supporting chunk
            if q_id == "EF-08":
                ans_lower = resp.answer.lower()
                has_cutoff_fact = "1:00 pm" in ans_lower or "1:00pm" in ans_lower or "1 pm" in ans_lower
                has_supporting_chunk = any(c.get("chunk_id") == "shipping_delivery_c1" for c in resp.citations)
                if not (has_cutoff_fact and has_supporting_chunk):
                    supportedness = 0.0

            category_metrics[cat]["count"] += 1
            category_metrics[cat]["recall"].append(retrieval_recall)
            category_metrics[cat]["citation_acc"].append(citation_accuracy)
            category_metrics[cat]["supportedness"].append(supportedness)
            category_metrics[cat]["latencies"].append(resp.latency_ms)

        print(f"[{idx:02d}/{len(dataset):02d}] {q_id} ({cat:<14}) | Recall: {retrieval_recall:.0%} | CiteAcc: {citation_accuracy:.0%} | Supp: {supportedness:.0%} | Latency: {resp.latency_ms:.1f}ms")

        results.append({
            "id": q_id,
            "category": cat,
            "question": question,
            "answer": resp.answer,
            "retrieval_recall": retrieval_recall,
            "citation_accuracy": citation_accuracy,
            "supportedness": supportedness,
            "citation_validation_passed": citation_valid,
            "latency_ms": resp.latency_ms,
            "cited_sources_count": len(resp.citations),
        })

    total_eval_duration = time.time() - start_eval_time

    # Calculate overall aggregates
    all_recalls = category_metrics["exact_fact"]["recall"] + category_metrics["cross_document"]["recall"]
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0

    all_cite_acc = category_metrics["exact_fact"]["citation_acc"] + category_metrics["cross_document"]["citation_acc"]
    avg_cite_acc = sum(all_cite_acc) / len(all_cite_acc) if all_cite_acc else 0.0

    all_supp = category_metrics["exact_fact"]["supportedness"] + category_metrics["cross_document"]["supportedness"]
    avg_supp = sum(all_supp) / len(all_supp) if all_supp else 0.0

    unsupported_refusals = category_metrics["unsupported"]["refusal_correct"]
    unsupported_acc = sum(unsupported_refusals) / len(unsupported_refusals) if unsupported_refusals else 0.0

    val_pass_count = sum(1 for r in results if r["citation_validation_passed"])
    val_pass_rate = (val_pass_count / len(results) * 100.0) if results else 100.0

    all_latencies = [r["latency_ms"] for r in results]
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

    print("\n" + "=" * 85)
    print("                         EVALUATION METRICS SUMMARY (RAGAS)")
    print("=" * 85)
    print(f"{'Evaluation Metric':<40} | {'Target Benchmark':<18} | {'Measured Score':<18}")
    print("-" * 85)
    print(f"{'Retrieval Recall@K (Factual & Cross-Doc)':<40} | {'> 90.0%':<18} | {avg_recall * 100:.1f}%")
    print(f"{'Citation Accuracy (Chunk & Page)':<40} | {'> 90.0%':<18} | {avg_cite_acc * 100:.1f}%")
    print(f"{'Supportedness (Factual & Cross-Doc)':<40} | {'> 90.0%':<18} | {avg_supp * 100:.1f}%")
    print(f"{'Anti-Hallucination Refusal Rate':<40} | {'100.0%':<18} | {unsupported_acc * 100:.1f}%")
    print(f"{'Citation Reference Validation Pass Rate':<40} | {'100.0%':<18} | {val_pass_rate:.1f}%")
    print(f"{'Average Cold Generation Latency':<40} | {'< 500 ms':<18} | {avg_latency:.1f} ms")
    print(f"{'Total Questions Evaluated':<40} | {'40':<18} | {len(results)}")
    print("-" * 85)

    print("\nCATEGORY PERFORMANCE BREAKDOWN:")
    for cat_name, data in category_metrics.items():
        count = data["count"]
        if cat_name == "unsupported":
            ref_rate = (sum(data["refusal_correct"]) / count) * 100 if count else 0
            print(f"• {cat_name.upper():<16} : {count:02d} items | Correct Refusal Rate: {ref_rate:.1f}%")
        else:
            rec = (sum(data["recall"]) / count) * 100 if count else 0
            cite = (sum(data["citation_acc"]) / count) * 100 if count else 0
            supp = (sum(data["supportedness"]) / count) * 100 if count else 0
            print(f"• {cat_name.upper():<16} : {count:02d} items | Recall: {rec:.1f}% | CiteAcc: {cite:.1f}% | Supported: {supp:.1f}%")

    report_path = Path(__file__).resolve().parent / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "retrieval_recall_pct": round(avg_recall * 100, 2),
                "citation_accuracy_pct": round(avg_cite_acc * 100, 2),
                "supportedness_pct": round(avg_supp * 100, 2),
                "anti_hallucination_rate_pct": round(unsupported_acc * 100, 2),
                "citation_validation_rate_pct": round(val_pass_rate, 2),
                "avg_cold_latency_ms": round(avg_latency, 2),
                "total_questions": len(results),
            },
            "results": results,
        }, f, indent=2)

    print(f"\nDetailed evaluation report saved to: {report_path}\n" + "=" * 85)


if __name__ == "__main__":
    run_quality_evaluation()
