# NovaCart: Caching Design and Trade-offs

NovaCart uses Redis to cache verified answers and reduce repeated retrieval and language-model calls. This report compares caching strategies and explains the application's approach to freshness, memory management and Redis outages.

## Caching strategy comparisons

The following tables compare read and write strategies. The implementation sections below explain the choices used in NovaCart.

| Read strategy | How it works | Trade-off |
| --- | --- | --- |
| **Cache-aside (used)** | The application checks the cache, handles a miss and caches the result. | Supports our custom retrieval and generation workflow, but requires application logic. |
| Read-through | A cache layer loads missing data through a configured loader. | Simplifies callers, but moves the custom miss workflow into that layer. Redis alone does not provide this loader. |

| Write strategy | How it works | Trade-off |
| --- | --- | --- |
| **Write-through** | Updates the cache and backing store synchronously before acknowledging success. | Keeps them aligned on successful writes, but adds latency and requires failure handling. NovaCart uses the write-through-style policy update process described below. |
| Write-behind / write-back | Updates the cache first and persists changes asynchronously. | Fast writes and possible batching, but pending writes can be lost and storage can lag. |
| Write-around | Writes directly to storage; later reads populate the cache. | Avoids caching rarely read writes, but the next read may miss; existing cached copies must be invalidated. |

## TTL, invalidation and eviction

- **TTL:** A time limit after which an entry expires. NovaCart defaults to 24 hours; TTL alone does not reflect policy changes immediately.
- **Invalidation:** Makes outdated entries unusable after data changes. NovaCart changes the policy revision in cache keys.
- **Eviction:** Removes entries to free memory, potentially before their TTL expires. NovaCart's connected Redis uses `volatile-lru`.

## Eviction algorithm comparisons

| Algorithm | Removes | Trade-off |
| --- | --- | --- |
| LRU | Least recently accessed entries. | Adapts to recent demand, but one-time scans can displace useful entries. |
| LFU | Least frequently accessed entries. | Retains popular entries, but can adapt slowly when popularity changes. |
| FIFO | Earliest inserted entries. | Simple ordering, but ignores subsequent use. |
| Random | Randomly selected entries. | Requires no popularity tracking, but may discard frequently used entries. |

## Supported Redis eviction policies

Redis 7 supports `allkeys-lru`, `allkeys-lfu`, `allkeys-random`, `volatile-lru`, `volatile-lfu`, `volatile-random`, `volatile-ttl` and `noeviction`. Redis 8.6 adds `allkeys-lrm` and `volatile-lrm` (least recently modified).

`allkeys-*` considers all keys; `volatile-*` considers only keys with TTLs. `volatile-ttl` selects the shortest remaining TTL. `noeviction` rejects memory-growing writes when the limit is reached. Redis approximates LRU and LFU and has no built-in FIFO policy. LRM tracks modification time, not insertion order. See the [official Redis eviction documentation](https://redis.io/docs/latest/develop/reference/eviction/).

## Application implementation

### Cache-aside for answers

NovaCart uses **cache-aside instead of read-through** because a cache miss requires a custom workflow:

**Retrieval -> reranking -> generation -> evidence verification -> cache the supported answer.**

The application checks Redis first. A hit returns the saved answer and citations without repeating that workflow. Cache keys include the normalized question, policy revision, model and retrieval settings. Failed or unsupported answers are not cached.

### Synchronous policy updates

Policy updates use an application-managed **write-through-style** process: prepare replacement indexes, write the new revision's snapshot to Redis, atomically replace the authoritative JSON file, then activate the revision. A Redis write failure prevents the update from becoming active. This is not a distributed transaction.

### TTL and invalidation

Answer and policy-snapshot keys expire after **24 hours by default**, configurable through `CACHE_TTL`. Reading an answer does not extend its TTL.

A policy update changes the collection revision included in answer keys. New requests therefore bypass old answers immediately; those old keys remain until expiration or eviction. This simple approach also invalidates answers unrelated to the edited policy.

### Redis eviction policy: `volatile-lru`

 When the configured memory limit is reached, Redis uses an approximation of least recently used eviction to remove keys that have an expiration time.

All NovaCart answer and policy-snapshot keys have TTLs and are therefore eligible for eviction. This policy favors recently accessed entries, helping retain reusable answers and reduce retrieval work and language-model calls. Evicted entries can be rebuilt on demand, although subsequent requests may incur additional latency and generation costs.

Keys without TTLs are excluded from this policy. The application's cache configuration is managed by the Redis server; NovaCart does not modify server eviction settings. The live memory limit was not available during verification, so this report makes no claim about the instance's configured memory capacity.

### Runtime fallback

Redis is **mandatory at startup**. If it fails later, chat generates fresh verified answers without caching, and policy browsing uses the loaded snapshot. Redis operations use one-second connection/socket timeouts and a **30-second recovery cooldown**. Caching resumes after a successful recovery attempt.

This keeps chat available but increases response time and LLM usage. Request and generation limits remain active, and administrative policy updates still require Redis.
