from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Iterable


def percentile(values: Iterable[float], p: float) -> float:
    xs = sorted(float(x) for x in values)
    if not xs:
        raise ValueError("values must not be empty")
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def summarize(values: Iterable[float]) -> dict:
    xs = [float(x) for x in values]
    if not xs:
        raise ValueError("values must not be empty")
    return {
        "mean": statistics.mean(xs),
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


@dataclass
class Timer:
    use_cuda_sync: bool = False

    def __enter__(self):
        if self.use_cuda_sync and __import__("torch").cuda.is_available():
            __import__("torch").cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.use_cuda_sync and __import__("torch").cuda.is_available():
            __import__("torch").cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start
