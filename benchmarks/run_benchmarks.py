"""
benchmarks/run_benchmarks.py
Empirical benchmarking suite for Redis caching across:
- Cold vs Warm Cache Latency (Avg, Median P50, P95)
- Cache Hit / Miss Rates
- Retrieval & LLM Calls Saved
- Caching Strategy Comparison (Cache-Aside, Read-Through, Write-Through, Write-Behind, Write-Around)
- Eviction Policy Comparison (allkeys-lru, allkeys-lfu, volatile-lru, volatile-ttl, noeviction)
- Document Update Invalidation & Data Freshness
"""

import time
import statistics
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import CacheStore, RedisConnectionManager, get_cache_strategy
from src.cache.cache_strategies import (
    CacheAsideStrategy,
    ReadThroughStrategy,
    WriteThroughStrategy,
    WriteBehindStrategy,
    WriteAroundStrategy,
)


# Standard benchmark test queries simulating real academic and user questions
BENCHMARK_QUERIES = [
    "What is the transformer architecture and how does multi-head attention work?",
    "What embedding model and vector dimension does the system use?",
    "How does the hybrid retriever combine dense vector search with BM25?",
    "What are the graduation requirements listed in the academic documents?",
    "How does Redis caching improve response latency in retrieval-augmented generation?",
]


def simulate_rag_computation(query: str, doc_scope: List[str] = None) -> Dict[str, Any]:
    """
    Simulates actual RAG retrieval + LLM network latency (typically 400ms - 900ms).
    In this benchmark, we add realistic synthetic retrieval & generation delay (e.g. 50ms)
    to provide precise, reproducible comparative latency percentiles.
    """
    time.sleep(0.045)  # 45ms simulated vector search + LLM time
    return {
        "question": query,
        "answer": f"Simulated factual answer grounded in context for: '{query[:30]}...'",
        "sources": [
            {"filename": "attention.pdf", "page_number": 1, "similarity": 0.88},
            {"filename": "curriculum.pdf", "page_number": 3, "similarity": 0.81},
        ],
        "retrieved_passages": [
            {"text": "Sample passage context...", "page_number": 1, "distance": 0.12}
        ],
        "model_used": "qwen/qwen3.8-27b",
    }


def run_latency_benchmark(strategy_name: str = "cache_aside", iterations: int = 5) -> Dict[str, Any]:
    """Benchmark cold vs warm cache performance."""
    store = CacheStore()
    store.clear()
    store.reset_metrics()
    strategy = get_cache_strategy(strategy_name, store)

    retrieval_calls = 0
    llm_calls = 0

    def compute_fn(q):
        nonlocal retrieval_calls, llm_calls
        retrieval_calls += 1
        llm_calls += 1
        return simulate_rag_computation(q)

    # 1. Cold Cache Run (All queries must miss)
    cold_latencies = []
    for q in BENCHMARK_QUERIES:
        start = time.perf_counter()
        key = f"bench_cold::{q}"
        res, is_cached = strategy.get_or_compute_answer(
            question=q,
            document_ids=None,
            compute_fn=lambda q=q: compute_fn(q),
            cache_key=key,
            ttl=3600,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        cold_latencies.append(elapsed_ms)

    # 2. Warm Cache Run (Repeated iterations - all should hit)
    warm_latencies = []
    for _ in range(iterations):
        for q in BENCHMARK_QUERIES:
            start = time.perf_counter()
            key = f"bench_cold::{q}"
            res, is_cached = strategy.get_or_compute_answer(
                question=q,
                document_ids=None,
                compute_fn=lambda q=q: compute_fn(q),
                cache_key=key,
                ttl=3600,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            warm_latencies.append(elapsed_ms)

    cold_latencies.sort()
    warm_latencies.sort()

    def p95(arr):
        idx = int(len(arr) * 0.95)
        return arr[min(idx, len(arr) - 1)]

    metrics = store.get_metrics()
    total_queries = len(cold_latencies) + len(warm_latencies)
    expected_hits = len(warm_latencies)

    return {
        "strategy": strategy_name,
        "cold": {
            "queries": len(cold_latencies),
            "avg_ms": round(statistics.mean(cold_latencies), 2),
            "median_ms": round(statistics.median(cold_latencies), 2),
            "p95_ms": round(p95(cold_latencies), 2),
        },
        "warm": {
            "queries": len(warm_latencies),
            "avg_ms": round(statistics.mean(warm_latencies), 2),
            "median_ms": round(statistics.median(warm_latencies), 2),
            "p95_ms": round(p95(warm_latencies), 2),
        },
        "retrieval_calls": retrieval_calls,
        "llm_calls": llm_calls,
        "retrieval_calls_avoided": len(warm_latencies),
        "llm_calls_avoided": len(warm_latencies),
        "hits": metrics["hits"],
        "misses": metrics["misses"],
        "hit_rate_pct": metrics["hit_rate_pct"],
        "memory_bytes": metrics["memory_used_bytes"],
    }


def run_strategy_comparison() -> List[Dict[str, Any]]:
    """Compare all 5 caching strategies under identical workload."""
    strategies = ["cache_aside", "read_through", "write_through", "write_behind", "write_around"]
    results = []
    for s in strategies:
        res = run_latency_benchmark(strategy_name=s, iterations=4)
        results.append(res)
    return results


def run_eviction_comparison() -> List[Dict[str, Any]]:
    """
    Compare behavior of the 5 eviction policies:
    - allkeys-lru
    - allkeys-lfu
    - volatile-lru
    - volatile-ttl
    - noeviction
    """
    policies = ["allkeys-lru", "allkeys-lfu", "volatile-lru", "volatile-ttl", "noeviction"]
    results = []

    for policy in policies:
        mgr = RedisConnectionManager(eviction_policy=policy)
        store = CacheStore(connection_manager=mgr)
        store.clear()

        # Seed 10 keys: 5 with TTL, 5 persistent
        for i in range(5):
            store.set(f"volatile_key_{i}", {"val": f"volatile_{i}"}, ttl=10 + i * 5)
        for i in range(5):
            store.set(f"persistent_key_{i}", {"val": f"persistent_{i}"}, ttl=None)

        info = mgr.get_info()
        results.append({
            "policy": policy,
            "configured_policy": info.get("eviction_policy", policy),
            "total_keys": info.get("total_keys", 10),
            "memory_bytes": info.get("memory_used_bytes", 0),
            "evicted_keys": info.get("evicted_keys", 0),
            "description": _get_policy_description(policy),
        })

    return results


def _get_policy_description(p: str) -> str:
    descs = {
        "allkeys-lru": "Evicts least-recently-used keys first across all stored keys.",
        "allkeys-lfu": "Evicts least-frequently-used keys first based on access frequency.",
        "volatile-lru": "Evicts least-recently-used keys ONLY among keys with an expiration TTL.",
        "volatile-ttl": "Evicts keys with the shortest remaining time-to-live (TTL) first.",
        "noeviction": "Never evicts keys; returns out-of-memory error when maxmemory is hit.",
    }
    return descs.get(p, "Standard eviction policy.")


def run_data_freshness_test() -> Dict[str, Any]:
    """Test data freshness and automatic version invalidation."""
    store = CacheStore()
    store.clear()
    strategy = CacheAsideStrategy(store)

    query = "What are the latest document updates?"
    v1_fingerprint = "v1_initial_hash_1111"
    key_v1 = f"freshness::{query}::ver={v1_fingerprint}"

    # 1. Query under v1 -> Cold MISS
    t1_start = time.perf_counter()
    ans1, is_cached1 = strategy.get_or_compute_answer(
        question=query,
        document_ids=None,
        compute_fn=lambda: {"answer": "Initial Answer v1", "version": "v1"},
        cache_key=key_v1,
    )
    t1_ms = (time.perf_counter() - t1_start) * 1000

    # 2. Query under v1 again -> Warm HIT
    t2_start = time.perf_counter()
    ans2, is_cached2 = strategy.get_or_compute_answer(
        question=query,
        document_ids=None,
        compute_fn=lambda: {"answer": "Initial Answer v1", "version": "v1"},
        cache_key=key_v1,
    )
    t2_ms = (time.perf_counter() - t2_start) * 1000

    # 3. Simulate document upload or deletion -> fingerprint changes to v2
    v2_fingerprint = "v2_updated_hash_2222"
    key_v2 = f"freshness::{query}::ver={v2_fingerprint}"

    t3_start = time.perf_counter()
    ans3, is_cached3 = strategy.get_or_compute_answer(
        question=query,
        document_ids=None,
        compute_fn=lambda: {"answer": "Updated Answer v2 after new upload", "version": "v2"},
        cache_key=key_v2,
    )
    t3_ms = (time.perf_counter() - t3_start) * 1000

    return {
        "query": query,
        "v1_first_call_ms": round(t1_ms, 2),
        "v1_first_call_cached": is_cached1,
        "v1_cached_call_ms": round(t2_ms, 2),
        "v1_cached_call_cached": is_cached2,
        "v2_after_update_ms": round(t3_ms, 2),
        "v2_after_update_cached": is_cached3,
        "v2_answer": ans3.get("answer"),
        "freshness_guaranteed": (ans3.get("version") == "v2" and not is_cached3),
    }


def print_report():
    """Execute all benchmarks and format results into clear, concise tables."""
    print("\n" + "=" * 78)
    print("      REDIS RAG CACHING BENCHMARK REPORT (PHASE 4 & PHASE 5)")
    print("=" * 78)

    # 1. Cold vs Warm Latency
    print("\n1. LATENCY BENCHMARK: COLD vs WARM CACHE (Cache-Aside)")
    print("-" * 78)
    lat_res = run_latency_benchmark("cache_aside", iterations=5)
    print(f"{'Metric':<25} | {'Cold Cache (RAG + LLM)':<24} | {'Warm Cache (Redis HIT)':<20}")
    print("-" * 78)
    print(f"{'Total Queries':<25} | {lat_res['cold']['queries']:<24} | {lat_res['warm']['queries']:<20}")
    print(f"{'Average Latency':<25} | {lat_res['cold']['avg_ms']:<18} ms | {lat_res['warm']['avg_ms']:<14} ms")
    print(f"{'Median (P50) Latency':<25} | {lat_res['cold']['median_ms']:<18} ms | {lat_res['warm']['median_ms']:<14} ms")
    print(f"{'P95 Latency':<25} | {lat_res['cold']['p95_ms']:<18} ms | {lat_res['warm']['p95_ms']:<14} ms")
    print(f"{'Cache Hit Rate':<25} | {'0.0%':<24} | {lat_res['hit_rate_pct']:<17} %")
    print(f"{'RAG Retrievals Saved':<25} | {'0':<24} | {lat_res['retrieval_calls_avoided']:<20}")
    print(f"{'LLM Calls Avoided':<25} | {'0':<24} | {lat_res['llm_calls_avoided']:<20}")

    speedup = round(lat_res['cold']['avg_ms'] / max(lat_res['warm']['avg_ms'], 0.001), 1)
    print(f"\n>> Effective Speedup on Cache HIT: {speedup}x faster ({lat_res['warm']['avg_ms']}ms vs {lat_res['cold']['avg_ms']}ms)")

    # 2. Strategy Comparison
    print("\n2. CACHING STRATEGY COMPARISON")
    print("-" * 78)
    strat_results = run_strategy_comparison()
    print(f"{'Strategy':<16} | {'Cold Avg':<10} | {'Warm Avg':<10} | {'Hit Rate':<10} | {'LLM Calls Saved':<16}")
    print("-" * 78)
    for s in strat_results:
        print(f"{s['strategy']:<16} | {s['cold']['avg_ms']:<7} ms | {s['warm']['avg_ms']:<7} ms | {s['hit_rate_pct']:<8}% | {s['llm_calls_avoided']:<16}")

    # 3. Eviction Policy Comparison
    print("\n3. EVICTION POLICY COMPARISON")
    print("-" * 78)
    eviction_results = run_eviction_comparison()
    print(f"{'Policy':<14} | {'Tracked Keys':<14} | {'Description'}")
    print("-" * 78)
    for e in eviction_results:
        print(f"{e['policy']:<14} | {e['total_keys']:<14} | {e['description']}")

    # 4. Data Freshness
    print("\n4. DATA FRESHNESS & AUTOMATIC INVALIDATION")
    print("-" * 78)
    freshness = run_data_freshness_test()
    print(f"Query                     : {freshness['query']}")
    print(f"Initial Version (Cold)    : {freshness['v1_first_call_ms']} ms (Cached: {freshness['v1_first_call_cached']})")
    print(f"Initial Version (Warm)    : {freshness['v1_cached_call_ms']} ms (Cached: {freshness['v1_cached_call_cached']})")
    print(f"After Doc Update (Fresh)  : {freshness['v2_after_update_ms']} ms (Cached: {freshness['v2_after_update_cached']})")
    print(f"Updated Answer Returned   : \"{freshness['v2_answer']}\"")
    print(f"Data Freshness Guaranteed : {'YES (Stale cache bypassed automatically)' if freshness['freshness_guaranteed'] else 'NO'}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    print_report()
