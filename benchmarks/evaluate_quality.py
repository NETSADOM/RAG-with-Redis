'Independent, deterministic factual regression checks plus optional live evaluation.'
import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.rag.pipeline import CustomerSupportRAGPipeline, RETRIEVAL_VERSION
from backend.rag.generator import CustomerSupportGenerator

DATASET = ROOT / "benchmarks/evaluation_dataset.json"


def normalize_answer(answer):
    answer = re.sub(r"\[\d+\]", "", answer.lower()).replace(",", "")
    answer = answer.replace("\u2013", "-").replace("\u2014", "-")
    words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
             "six": "6", "seven": "7", "ten": "10", "fifteen": "15", "thirty": "30"}
    return re.sub(r"\b(" + "|".join(words) + r")\b", lambda match: words[match[0]], answer)


def factual_checks(item, answer):
    text = normalize_answer(answer)
    required = item.get("required_facts", [])
    missing = [pattern for pattern in required if not re.search(pattern, text, re.I | re.S)]
    forbidden = [pattern for pattern in item.get("forbidden_claims", []) if re.search(pattern, text, re.I | re.S)]
    numbers = {Decimal(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}
    allowed = {Decimal(n) for n in item.get("allowed_numbers", [])}
    unexpected = sorted(str(n) for n in numbers - allowed)
    return {"passed": bool(required) and not missing and not forbidden and not unexpected,
            "missing_facts": missing, "forbidden_claims": forbidden, "unexpected_numbers": unexpected}


def validate_dataset(dataset):
    ids = set()
    for item in dataset:
        if item["id"] in ids:
            raise ValueError("Duplicate evaluation ID: " + item["id"])
        ids.add(item["id"])
        if item["category"] == "unsupported":
            if item.get("expected_chunk_ids"):
                raise ValueError("Unsupported case has expected evidence: " + item["id"])
            continue
        if not item.get("expected_chunk_ids") or not item.get("required_facts") or "allowed_numbers" not in item:
            raise ValueError("Missing factual expectations: " + item["id"])
        for group in item.get("expected_evidence_groups", []):
            if not group or not set(group).issubset(item["expected_chunk_ids"]):
                raise ValueError("Invalid alternative evidence group: " + item["id"])
        for pattern in item["required_facts"] + item.get("forbidden_claims", []):
            re.compile(pattern)
        result = factual_checks(item, item["ground_truth"])
        if not result["passed"]:
            raise ValueError(f"Ground truth fails its own expectations for {item['id']}: {result}")


def score_response(item, response):
    expected = set(item.get("expected_chunk_ids") or item.get("expected_doc_ids", []))
    field = "chunk_id" if item.get("expected_chunk_ids") else "doc_id"
    retrieved = {p.get(field) for p in response.retrieved_context}
    pages = set(item.get("expected_pages", []))
    correct_cites = {c.get(field) for c in response.citations if c.get(field) in expected and (not pages or c.get("page_number") in pages)}
    groups = item.get("expected_evidence_groups") or [[source] for source in expected]
    covered_groups = sum(bool(set(group) & correct_cites) for group in groups)
    supported_case = item["category"] != "unsupported"
    facts = factual_checks(item, response.answer) if supported_case else None
    answer_supported = None
    if supported_case:
        answer_supported = bool(response.is_supported and response.citations and response.citation_validation_passed
                                and not response.error and groups and covered_groups == len(groups) and facts["passed"])
    return {
        "id": item["id"], "category": item["category"], "question": item["question"],
        "answer": response.answer, "model_used": response.model_used,
        "retrieval_recall": len(expected & retrieved) / len(expected) if expected else None,
        "expected_citation_coverage": covered_groups / len(groups) if groups else None,
        "citations": response.citations,
        "retrieved_chunk_ids": [p.get("chunk_id") for p in response.retrieved_context],
        "citation_validation_passed": response.citation_validation_passed,
        "factual_checks": facts, "supported_answer_check": answer_supported,
        "refusal_reason": response.refusal_reason,
        "clean_refusal": (not response.is_supported and not response.citations and not response.error
                          and response.refusal_reason in {"insufficient_evidence", "audit_rejected"}) if not supported_case else None,
        "latency_ms": response.latency_ms, "error": response.error,
    }


def quality_failures(metrics):
    failures = []
    if metrics["errors"]:
        failures.append("Provider or pipeline errors occurred.")
    for field, minimum in (("supported_answer_check_pct", 90), ("clean_refusal_pct", 90),
                           ("retrieval_recall_pct", 90), ("expected_citation_coverage_pct", 90)):
        if metrics.get(field) is None or metrics[field] < minimum:
            failures.append(f"{field} must be at least {minimum}%.")
    return failures


def run_quality_evaluation(validate_only=False, interval=2):
    if interval < 0:
        raise ValueError("Evaluation interval cannot be negative.")
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    dataset_digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    validate_dataset(dataset)
    if validate_only:
        print(f"Validated {len(dataset)} evaluation cases and their factual expectations. No live model calls.")
        return 0
    generator = CustomerSupportGenerator()
    generator.validate_configuration()  # Fail before loading models or calling Redis.
    pipeline = CustomerSupportRAGPipeline(generator=generator)
    results = []
    try:
        for index, item in enumerate(dataset):
            if index and interval:
                time.sleep(interval)
            response = pipeline.answer_question(item["question"], skip_cache=True)
            result = score_response(item, response)
            results.append(result)
            print(f"{item['id']}: factual={result['supported_answer_check']}, refusal={result['clean_refusal']}, error={bool(result['error'])}")
    finally:
        pipeline.cache.close()

    def percent(field):
        values = [r[field] for r in results if r[field] is not None]
        return round(statistics.mean(values) * 100, 2) if values else None

    metrics = {
        "retrieval_recall_pct": percent("retrieval_recall"),
        "expected_citation_coverage_pct": percent("expected_citation_coverage"),
        "citation_reference_validation_pct": percent("citation_validation_passed"),
        "supported_answer_check_pct": percent("supported_answer_check"),
        "clean_refusal_pct": percent("clean_refusal"),
        "mean_pipeline_latency_ms": round(statistics.mean(r["latency_ms"] for r in results), 2),
        "errors": sum(bool(r["error"]) for r in results), "total_questions": len(results),
    }
    failures = quality_failures(metrics)
    report = {"mode": pipeline.generator.model, "passed": not failures, "failures": failures,
              "dataset_sha256": dataset_digest, "retrieval_version": RETRIEVAL_VERSION,
              "knowledge_base_revision": pipeline.kb_manager.get_collection_revision(),
              "limitations": "Deterministic factual regression checks are not a complete semantic accuracy measure. Review sampled answers for contradictions and omitted conditions.",
              "metrics": metrics, "results": results}
    report_path = ROOT / "benchmarks/reports/evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    for failure in failures:
        print("FAIL: " + failure)
    print(f"Report: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Validate labels and factual expectations without external services.")
    parser.add_argument("--interval", type=float, default=2, help="Seconds between live cases; increase for provider quotas.")
    args = parser.parse_args()
    try:
        raise SystemExit(run_quality_evaluation(args.validate_only, args.interval))
    except (ValueError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
