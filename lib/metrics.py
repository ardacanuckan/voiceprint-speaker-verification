"""Measurement utilities: cosine similarity, timing, memory."""

import time
import os
import numpy as np
import psutil


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def measure_inference(fn, *args, **kwargs):
    """Run fn(*args) and return (result, time_ms, ram_delta_mb)."""
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    mem_after = proc.memory_info().rss
    ram_mb = max(0, (mem_after - mem_before)) / 1024 / 1024
    return result, elapsed_ms, ram_mb
