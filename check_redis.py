"""Verify hosted Redis credentials without loading ML models or printing secrets."""
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv
from backend.cache.redis_cache import CustomerSupportCache, RedisUnavailable


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    cache = None
    try:
        cache = CustomerSupportCache()
        key = "cs_rag:connection-check:" + uuid4().hex
        try:
            cache._client.set(key, "connected", ex=30)
            if cache._client.get(key) != "connected":
                raise RedisUnavailable("Redis read/write check failed.")
        finally:
            cache._client.delete(key)
        print("Redis connected. Authentication, write, read and delete checks passed.")
        return 0
    except Exception:
        print("Redis check failed. Verify REDIS_URL, TLS settings and database access in .env. Credentials have not been printed.")
        return 1
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
