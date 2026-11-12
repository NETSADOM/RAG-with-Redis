"""Bound resource use in the explicitly single-worker application."""
import os
import time
from contextlib import contextmanager
from threading import Lock


class SupportBusy(RuntimeError):
    pass


def positive_setting(name, default):
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


class RequestLimiter:
    """Fixed per-client windows; bounded memory, with no client eviction bypass."""
    def __init__(self, limit=20, window=60, max_clients=4096, clock=time.monotonic):
        self.limit, self.window, self.max_clients = limit, window, max_clients
        self.clock = clock
        self._clients = {}
        self._lock = Lock()

    def allow(self, client):
        now = self.clock()
        with self._lock:
            self._clients = {key: value for key, value in self._clients.items() if value[0] > now}
            expires, count = self._clients.get(client, (now + self.window, 0))
            if count >= self.limit or (client not in self._clients and len(self._clients) >= self.max_clients):
                return False
            self._clients[client] = (expires, count + 1)
            return True


@contextmanager
def generation_slot(semaphore):
    if not semaphore.acquire(blocking=False):
        raise SupportBusy("Support is busy. Please retry shortly.")
    try:
        yield
    finally:
        semaphore.release()
