"""
benchmarks/simulate_cache_throughput.py
ISOLATED SYNTHETIC CACHE EXPERIMENT (Separate from live application benchmarks).

Simulates high-concurrency theoretical throughput by substituting an artificial 45ms delay
for vector retrieval and generation. Used exclusively for theoretical cache-aside latency
and throughput modeling.

Live end-to-end application benchmarks are measured separately in benchmarks/benchmark.py.
"""

import time
import statistics
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.cache.redis_cache import CustomerSupportCache


SIMULATED_QUERIES = [
    "How long do I have to return an item?",
    "What are your shipping rates?",
    "Can I cancel my order within 30 minutes?",
    "My package arrived damaged, what should I do?",
    "Do you accept PayPal and Apple Pay?",
]


def simulated_rag_workload(query: str) -> Dict[str, Any]:
    """Synthetic mock simulating 45ms retrieval + generation latency."""
    time.sleep(0.045)
    return {
        "answer": f"Synthetic response for: {query}",
        "sources": [{"doc_id": "sim_doc_01", "chunk_id": "c1", "page_number": 1}],
    }


def run_simulation(iterations: int = 50):
    print("\n" + "=" * 80)
    print("        ISOLATED SYNTHETIC CACHE SIMULATION EXPERIMENT")
    print("   (Theoretical model using artificial 45ms synthetic workload)")
    print("=" * 80)

    cache = CustomerSupportCache(use_fallback=True)
    cache.clear()

    cold_latencies = []
    warm_latencies = []

    print("\n1. Running Cold Simulation (Simulated 45ms generation)...")
    for q in SIMULATED_QUERIES:
        t0 = time.perf_counter()
        data, hit = cache.get_answer(q, "v_sim", "sim_model")
        if not hit:
            result = simulated_rag_workload(q)
            cache.set_answer(q, "v_sim", "sim_model", result["answer"], result["sources"])
        elapsed = (time.perf_counter() - t0) * 1000
        cold_latencies.append(elapsed)

    print("2. Running Warm Cache Simulation...")
    for _ in range(iterations):
        for q in SIMULATED_QUERIES:
            t0 = time.perf_counter()
            data, hit = cache.get_answer(q, "v_sim", "sim_model")
            elapsed = (time.perf_counter() - t0) * 1000
            if hit:
                warm_latencies.append(elapsed)

    avg_cold = statistics.mean(cold_latencies)
    avg_warm = statistics.mean(warm_latencies)
    p50_warm = statistics.median(warm_latencies)
    p95_warm = sorted(warm_latencies)[int(len(warm_latencies) * 0.95)]

    print("\n" + "-" * 80)
    print(f"Synthetic Cold Workload Avg Latency : {avg_cold:.2f} ms")
    print(f"Warm Redis Cache Avg Latency        : {avg_warm:.3f} ms")
    print(f"Warm Redis Cache Median (P50)       : {p50_warm:.3f} ms")
    print(f"Warm Redis Cache P95 Latency        : {p95_warm:.3f} ms")
    print(f"Theoretical Speedup Factor          : {avg_cold / max(avg_warm, 0.001):.1f}x faster")
    print("-" * 80)
    print("Note: Live end-to-end pipeline results are measured separately via benchmarks/benchmark.py\n")


if __name__ == "__main__":
    run_simulation()

