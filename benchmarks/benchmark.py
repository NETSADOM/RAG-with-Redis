"""Measure sequential caching and bounded concurrent load using isolated Redis keys."""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.rag.pipeline import CustomerSupportRAGPipeline
from backend.rag.generator import CustomerSupportGenerator
from backend.cache.redis_cache import CustomerSupportCache
from backend.admission import SupportBusy

QUESTIONS = [
    "How long do I have to return an item?",
    "What are your shipping options and costs?",
    "Can I cancel my order after placing it?",
    "My order arrived damaged, what should I do?",
    "What payment methods do you accept?",
]


def summarize(results, elapsed):
    successes = [r for r in results if r is not None and r.is_supported and not r.error]
    latencies = [r.latency_ms for r in successes]
    return {"requests": len(results), "supported_answers": len(successes),
            "busy_rejections": sum(r is None for r in results),
            "errors": sum(bool(r.error) for r in results if r is not None),
            "refusals": sum(not r.is_supported and not r.error for r in results if r is not None),
            "cache_hits": sum(r.cache_hit for r in successes),
            "successful_answers_per_second": round(len(successes) / max(elapsed, 0.000001), 2),
            "mean_success_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
            "p95_success_latency_ms": round(statistics.quantiles(latencies, n=20, method="inclusive")[18], 2) if len(latencies) > 1 else (latencies[0] if latencies else None)}


def run_benchmark(repeats=2, concurrency=4):
    if repeats < 1 or concurrency < 1:
        raise ValueError("Repeats and concurrency must be positive.")
    generator = CustomerSupportGenerator()
    generator.validate_configuration()
    cache = CustomerSupportCache(default_ttl=3600, namespace="cs_rag:benchmark:" + uuid4().hex)
    report = {"repeats": repeats, "concurrency": concurrency, "phases": {},
              "limitations": "Pipeline timings exclude HTTP/browser overhead and startup. Small corpus; no production capacity claim."}
    try:
        pipeline = CustomerSupportRAGPipeline(cache=cache, generator=generator)
        def execute(question, bypass=False):
            try:
                return pipeline.answer_question(question, skip_cache=bypass)
            except SupportBusy:
                return None
        for label, questions, bypass, parallel in (
            ("cache_bypass", QUESTIONS * repeats, True, False),
            ("first_lookup", QUESTIONS, False, False),
            ("repeat_lookup", QUESTIONS * repeats, False, False),
            ("concurrent_bypass", QUESTIONS * repeats, True, True),
        ):
            started = time.perf_counter()
            if parallel:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results = list(pool.map(lambda question: execute(question, bypass), questions))
            else:
                results = [execute(question, bypass) for question in questions]
            report["phases"][label] = summarize(results, time.perf_counter() - started)
            print(label + ": " + json.dumps(report["phases"][label]))
    finally:
        # The unique namespace prevents deleting any application or other run's keys.
        try:
            cache.clear()
        finally:
            cache.close()
    (ROOT / "benchmarks/benchmark_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    run_benchmark(args.repeats, args.concurrency)
