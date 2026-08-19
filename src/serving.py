"""
Briques du service d'inference — logique PURE, sans torch ni FastAPI.

Isolee pour la meme raison que `src/metrics.py` : ces mecanismes (comptage de
latences, limitation de debit) sont exactement le genre de code ou une erreur
passe inapercue en production. Les tester en CI, sans GPU, coute quelques
millisecondes.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
#  Metriques
# ---------------------------------------------------------------------------
def percentile(values: list[float], p: float) -> float:
    """Percentile par interpolation lineaire. `p` dans [0, 1]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = p * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


@dataclass
class MetricsCollector:
    """Compteurs et latences du service.

    On conserve une fenetre glissante des dernieres latences plutot que toutes
    les valeurs : la memoire reste bornee et les percentiles refletent le
    comportement RECENT, ce qui est ce qu'on surveille en production.
    """
    window: int = 512
    _latencies_ms: deque[float] = field(default_factory=deque, init=False)
    _tokens_per_s: deque[float] = field(default_factory=deque, init=False)
    requests_total: int = 0
    errors_total: int = 0
    tokens_generated_total: int = 0
    rejected_rate_limit: int = 0
    rejected_unauthorized: int = 0
    _started_at: float = field(default_factory=time.monotonic, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._latencies_ms = deque(maxlen=self.window)
        self._tokens_per_s = deque(maxlen=self.window)

    def record_success(self, latency_ms: float, n_tokens: int) -> None:
        with self._lock:
            self.requests_total += 1
            self.tokens_generated_total += n_tokens
            self._latencies_ms.append(latency_ms)
            if latency_ms > 0:
                self._tokens_per_s.append(n_tokens / (latency_ms / 1000))

    def record_error(self) -> None:
        with self._lock:
            self.requests_total += 1
            self.errors_total += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = list(self._latencies_ms)
            tps = list(self._tokens_per_s)
            return {
                "uptime_s": round(time.monotonic() - self._started_at, 1),
                "requests_total": self.requests_total,
                "errors_total": self.errors_total,
                "error_rate": (self.errors_total / self.requests_total
                               if self.requests_total else 0.0),
                "rejected_rate_limit": self.rejected_rate_limit,
                "rejected_unauthorized": self.rejected_unauthorized,
                "tokens_generated_total": self.tokens_generated_total,
                "latency_ms": {
                    "count": len(lat),
                    "p50": round(percentile(lat, 0.50), 1),
                    "p95": round(percentile(lat, 0.95), 1),
                    "p99": round(percentile(lat, 0.99), 1),
                    "max": round(max(lat), 1) if lat else 0.0,
                },
                "tokens_per_s": {
                    "p50": round(percentile(tps, 0.50), 2),
                    "mean": round(sum(tps) / len(tps), 2) if tps else 0.0,
                },
            }

    def prometheus(self) -> str:
        """Exposition au format texte Prometheus."""
        s = self.snapshot()
        lines = [
            "# HELP inference_requests_total Nombre total de requetes",
            "# TYPE inference_requests_total counter",
            f"inference_requests_total {s['requests_total']}",
            "# HELP inference_errors_total Nombre total d'erreurs",
            "# TYPE inference_errors_total counter",
            f"inference_errors_total {s['errors_total']}",
            "# HELP inference_tokens_generated_total Tokens generes",
            "# TYPE inference_tokens_generated_total counter",
            f"inference_tokens_generated_total {s['tokens_generated_total']}",
            "# HELP inference_latency_ms Latence de generation",
            "# TYPE inference_latency_ms summary",
            f'inference_latency_ms{{quantile="0.5"}} {s["latency_ms"]["p50"]}',
            f'inference_latency_ms{{quantile="0.95"}} {s["latency_ms"]["p95"]}',
            f'inference_latency_ms{{quantile="0.99"}} {s["latency_ms"]["p99"]}',
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#  Limitation de debit
# ---------------------------------------------------------------------------
@dataclass
class TokenBucket:
    """Seau a jetons : `rate` requetes/seconde, avec une reserve de `capacity`.

    Choisi plutot qu'une fenetre fixe car il tolere de petites rafales tout en
    bornant le debit moyen — comportement attendu d'une API, ou un client
    legitime envoie parfois plusieurs requetes d'affilee.
    """
    rate: float
    capacity: float
    _tokens: float = field(init=False)
    _last: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        if self.rate <= 0 or self.capacity <= 0:
            raise ValueError("rate et capacity doivent etre strictement positifs.")
        self._tokens = self.capacity

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def retry_after_s(self) -> float:
        """Delai avant qu'un jeton soit de nouveau disponible."""
        if self._tokens >= 1.0:
            return 0.0
        return round((1.0 - self._tokens) / self.rate, 2)


class RateLimiter:
    """Un seau a jetons par client, cree a la demande."""

    def __init__(self, rate: float = 1.0, capacity: float = 5.0):
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, client_id: str) -> tuple[bool, float]:
        with self._lock:
            bucket = self._buckets.get(client_id)
            if bucket is None:
                bucket = TokenBucket(rate=self.rate, capacity=self.capacity)
                self._buckets[client_id] = bucket
            ok = bucket.allow()
            return ok, (0.0 if ok else bucket.retry_after_s())
