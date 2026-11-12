# NovaCart: Policy-Grounded Support with Redis

NovaCart is a customer-support application that answers questions using a local policy knowledge base. It combines document retrieval, reranking and a language model to produce cited answers, then uses Redis to reuse verified results and reduce repeated generation work.

The application includes a FastAPI backend and a plain HTML, CSS and JavaScript interface. Policy documents remain the source of truth; the language model interprets questions and explains the retrieved facts in customer-friendly language.

## How answers are produced

1. Matching requests share an in-process result and coordinate through a Redis lock when Redis is available.
2. The pipeline checks the answer cache using the normalized question, policy revision, model and retrieval settings.
3. On a miss, keyword and semantic retrieval identify candidate passages, followed by reranking when enabled and available.
4. The language model generates concise claims with supporting quotations. A separate model call audits those claims against the retrieved documents.
5. The application checks quotation provenance, citation structure and numeric evidence before returning the answer. Eligible supported answers are cached with their citations.

A cache hit avoids retrieval and model generation. Failed and unsupported answers are not cached. Evidence checks reduce unsupported responses, but do not guarantee factual accuracy.

The Redis check requires a working connection. The application itself can start in degraded mode without Redis. Open **http://127.0.0.1:8000**; API documentation is available at **http://127.0.0.1:8000/docs**. Retrieval models may require downloads on first use.

Restart the backend after editing `.env`. Existing process environment variables can override values in that file. Keep `.env` private; it is excluded from Git.

## Cache design and trade-offs

### Cache-aside versus read-through

| Strategy | Behavior | Trade-off |
| --- | --- | --- |
| **Cache-aside: used for answers** | The application checks Redis, computes a missing answer and stores an eligible result. | Gives the application control over retrieval, verification and cache eligibility. |
| Read-through | A cache layer invokes a loader when data is missing. | Simplifies callers, but requires the custom loading workflow inside that layer. Redis alone does not supply this loader. |

NovaCart uses **cache-aside instead of read-through** because a miss requires a custom workflow: **retrieval -> reranking -> generation -> evidence verification -> caching**.

### Write strategies

| Strategy | Behavior | Trade-off |
| --- | --- | --- |
| **Write-through** | Updates cache and storage synchronously before acknowledging success. | Adds write latency and requires failure handling. NovaCart uses an application-managed variant for policy updates. |
| Write-behind / write-back | Updates cache first and persists changes asynchronously. | Allows fast writes and batching, but storage can lag and pending changes can be lost. Not implemented here. |
| Write-around | Writes directly to storage; later reads populate the cache. | Avoids caching rarely read writes, but requires invalidation of existing copies and may cause a later miss. Not used for policy updates. |

For policy updates, NovaCart prepares replacement indexes, writes the new revision snapshot to Redis, atomically replaces the authoritative JSON file and activates the revision. Redis write failure prevents activation. This is a write-through-style sequence, not a distributed transaction.

### TTL, invalidation and eviction

| Mechanism | Purpose | NovaCart behavior |
| --- | --- | --- |
| TTL | Expires an entry after a specified lifetime. | Answers and policy snapshots expire after **24 hours** by default through `CACHE_TTL`. Reads do not refresh answer TTLs. |
| Invalidation | Prevents reuse of outdated data after a change. | Policy updates change the revision in answer keys. New requests cannot reuse the previous revision's answers. |
| Eviction | Removes entries under memory pressure, potentially before expiration. | Redis applies its server-configured eviction policy. Missing entries can be regenerated. |

Revision-based invalidation is simple, but also invalidates answers unrelated to an edited policy. Old entries remain until expiration or eviction.

### Eviction algorithms

| Algorithm | Evicts | Trade-off |
| --- | --- | --- |
| LRU | Least recently accessed entries. | Favors recent demand; one-time scans may displace useful entries. |
| LFU | Least frequently accessed entries. | Favors popular entries; adaptation to changing demand may be slower. |
| FIFO | Earliest inserted entries. | Simple ordering that ignores later usage. |
| Random | Randomly selected entries. | Requires no popularity tracking but can discard frequently used data. |

Redis 7 supports `allkeys-lru`, `allkeys-lfu`, `allkeys-random`, `volatile-lru`, `volatile-lfu`, `volatile-random`, `volatile-ttl` and `noeviction`. Redis 8.6 adds `allkeys-lrm` and `volatile-lrm`, based on least recent modification.

`allkeys-*` considers all keys; `volatile-*` considers only expiring keys. `volatile-ttl` favors eviction of keys with the shortest remaining TTL. `noeviction` rejects memory-growing writes at the limit. Redis approximates LRU and LFU and has no built-in FIFO policy. See the [Redis eviction reference](https://redis.io/docs/latest/develop/reference/eviction/).

### Verified Redis policy: `volatile-lru`

The connected instance previously reported **`volatile-lru`** through `INFO memory`. Under memory pressure, this policy evicts approximately the least recently used entries among keys with TTLs. It favors recently reused answers, reducing repeated retrieval and model calls.

Answer keys, policy snapshots and distributed request locks all expire and are therefore eligible. Eviction of a request lock can allow duplicate generation before its lease ends; the lock reduces duplicate work but does not guarantee exactly-once execution. Non-expiring keys are excluded.

## Redis outages and recovery

Redis is the primary shared cache. If it is unavailable at startup or fails during operation, chat runs the retrieval and generation pipeline without caching, while policy browsing uses the loaded snapshot. No replacement in-memory answer cache is introduced.

Redis calls use one-second connection and socket timeouts. Failures trigger a **30-second cooldown**, after which demand can trigger a recovery attempt. Caching resumes after successful recovery. The health endpoint reports degraded cache availability instead of treating a Redis outage alone as a complete application failure.


## Concurrent requests

Matching requests in one process share an in-progress result. Redis additionally coordinates matching requests using expiring locks with unique ownership tokens. A waiting process rechecks the answer cache after acquiring the lock; release checks ownership before deleting it.

`ANSWER_WAIT_SECONDS` defaults to **3000 seconds (50 minutes)** and also controls the Redis lease. After the waiting limit, a request may execute independently, subject to generation capacity. Unsupported answers are not cached and may be generated again by waiting processes. Expired or evicted locks can permit overlapping work.

The default limits are **20 requests per minute per client IP** and **2 simultaneous generations per process**. Capacity exhaustion can reject a request rather than start additional generation. The 50-minute setting limits coordination waiting; it does not guarantee that a browser, proxy or provider will keep a request open that long.

The application retains its **single-worker restriction** for the local policy store. Distributed request locking alone does not make policy updates safe across multiple workers or deployments.

## Repository structure

```text
backend/
  cache/           Redis answers, snapshots and request locks
  rag/             Retrieval, reranking, generation and orchestration
  data/            Authoritative policy knowledge base
  tests/           Cache, API, retrieval and generation tests
  main.py          FastAPI routes and application lifecycle
  admission.py     Request and generation limits
  locking.py       Synchronization and single-worker protection
frontend/public/   Browser interface
benchmarks/        Retrieval and answer-quality evaluation tools
check_redis.py     Redis connectivity check
redis.conf.example Example self-hosted Redis configuration
requirements.txt   Application and test dependencies
run.py             Local startup entry point
```

## Validation

```powershell
python -m pytest backend/tests -q -p no:cacheprovider
python benchmarks/evaluate_quality.py --validate-only
```


