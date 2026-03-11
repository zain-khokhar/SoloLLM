"""
Performance Dashboard API for SoloLLM — Phase 6.

Tracks and reports:
- Request latency and throughput
- Token usage statistics
- Model performance metrics
- Document/RAG pipeline stats
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


# ── Metrics Tracker ─────────────────────────────────────────

@dataclass
class RequestMetric:
    """A single request metric entry."""
    timestamp: float
    endpoint: str
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    success: bool = True


class MetricsTracker:
    """In-memory metrics tracker with rolling window."""

    def __init__(self, max_entries: int = 1000):
        self.metrics: deque[RequestMetric] = deque(maxlen=max_entries)
        self.total_requests: int = 0
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0
        self.total_errors: int = 0
        self.start_time: float = time.time()

    def record(
        self,
        endpoint: str,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        model: str = "",
        success: bool = True,
    ):
        """Record a request metric."""
        self.metrics.append(RequestMetric(
            timestamp=time.time(),
            endpoint=endpoint,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            success=success,
        ))
        self.total_requests += 1
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        if not success:
            self.total_errors += 1

    def get_summary(self) -> dict:
        """Get a summary of all metrics."""
        if not self.metrics:
            return {
                "uptime_seconds": round(time.time() - self.start_time),
                "total_requests": 0,
                "total_tokens": 0,
                "avg_latency_ms": 0,
                "error_rate": 0,
                "requests_per_minute": 0,
            }

        latencies = [m.latency_ms for m in self.metrics]
        uptime = time.time() - self.start_time

        # Per-model breakdown
        model_stats: dict[str, dict] = {}
        for m in self.metrics:
            if m.model:
                if m.model not in model_stats:
                    model_stats[m.model] = {
                        "requests": 0,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "avg_latency_ms": 0,
                        "latencies": [],
                    }
                model_stats[m.model]["requests"] += 1
                model_stats[m.model]["tokens_in"] += m.tokens_in
                model_stats[m.model]["tokens_out"] += m.tokens_out
                model_stats[m.model]["latencies"].append(m.latency_ms)

        # Compute avg latency per model
        for model, stats in model_stats.items():
            lats = stats.pop("latencies")
            stats["avg_latency_ms"] = round(sum(lats) / len(lats), 1) if lats else 0

        # Per-endpoint breakdown
        endpoint_stats: dict[str, int] = {}
        for m in self.metrics:
            endpoint_stats[m.endpoint] = endpoint_stats.get(m.endpoint, 0) + 1

        return {
            "uptime_seconds": round(uptime),
            "total_requests": self.total_requests,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens": self.total_tokens_in + self.total_tokens_out,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "min_latency_ms": round(min(latencies), 1),
            "max_latency_ms": round(max(latencies), 1),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
            "error_rate": round(self.total_errors / max(self.total_requests, 1) * 100, 2),
            "requests_per_minute": round(self.total_requests / max(uptime / 60, 1), 2),
            "model_stats": model_stats,
            "endpoint_stats": endpoint_stats,
        }

    def get_recent(self, count: int = 20) -> list[dict]:
        """Get recent metrics."""
        recent = list(self.metrics)[-count:]
        return [
            {
                "timestamp": datetime.fromtimestamp(m.timestamp, tz=timezone.utc).isoformat(),
                "endpoint": m.endpoint,
                "latency_ms": m.latency_ms,
                "tokens_in": m.tokens_in,
                "tokens_out": m.tokens_out,
                "model": m.model,
                "success": m.success,
            }
            for m in recent
        ]


# Singleton
metrics = MetricsTracker()


# ── Dashboard Endpoints ─────────────────────────────────────

@router.get("/summary")
async def get_dashboard_summary():
    """Get performance dashboard summary."""
    summary = metrics.get_summary()

    # Add system info
    try:
        import psutil
        summary["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": round(psutil.virtual_memory().used / 1024 / 1024),
            "memory_total_mb": round(psutil.virtual_memory().total / 1024 / 1024),
        }
    except ImportError:
        summary["system"] = {}

    return summary


@router.get("/recent")
async def get_recent_metrics(count: int = 20):
    """Get recent request metrics."""
    return {"metrics": metrics.get_recent(count)}


@router.get("/health")
async def dashboard_health():
    """Quick health check with basic stats."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - metrics.start_time),
        "total_requests": metrics.total_requests,
    }
