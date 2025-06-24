"""observability.py
Central observability (metrics & tracing) setup for the Credit Risk API.

This module exposes:
- Prometheus custom metrics (Counter, Histogram, Gauge)
- Middleware to collect latency & request counts
- OpenTelemetry tracing (OTLP exporter)

Import and call `init_observability(app)` once after creating the FastAPI app.
"""
from __future__ import annotations

import os
import time
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import REGISTRY  # default registry used by Instrumentator

# ---- Prometheus Metrics ----
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint", "method"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10),
)

MODEL_VERSION_GAUGE = Gauge(
    "credit_model_loaded",
    "Loaded ML model version label",
    ["version"],
)


# ---- Helper for model version ----

def set_model_version(version: str) -> None:  # noqa: D401 simple doc
    """Expose currently loaded model version to Prometheus."""
    # Remove previous labels to keep only the current version at value 1
    MODEL_VERSION_GAUGE.clear()  # type: ignore[attr-defined]
    MODEL_VERSION_GAUGE.labels(version=version).set(1)


# ---- Middleware ----

def _metrics_middleware_factory(app: FastAPI) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """Factory returning a middleware coroutine to capture Prometheus metrics."""

    async def _middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:  # type: ignore[name-defined]
        start_time = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start_time

        path = request.url.path
        method = request.method
        status_code = response.status_code

        REQUEST_COUNT.labels(endpoint=path, method=method, http_status=status_code).inc()
        REQUEST_LATENCY.labels(endpoint=path, method=method).observe(elapsed)
        return response

    return _middleware


# ---- OpenTelemetry Tracing ----

def _setup_tracing() -> None:
    """Configure basic OTLP tracing if OTEL_EXPORTER_OTLP_ENDPOINT env var is set."""
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otlp_endpoint:
        # Tracing disabled by configuration
        return

    # Lazy import to avoid dependency if tracing not enabled
    from opentelemetry import trace  # type: ignore
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore

    resource = Resource.create({SERVICE_NAME: "credit-risk-api"})
    provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    processor = BatchSpanProcessor(span_exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument()


# ---- Public entry ----

def init_observability(app: FastAPI) -> None:  # noqa: D401 simple doc
    """Attach middleware and configure tracing/metrics."""
    # Add middleware for custom Prometheus metrics
    app.middleware("http")(_metrics_middleware_factory(app))  # type: ignore[arg-type]

    # Setup OpenTelemetry tracing (noop if env var not set)
    _setup_tracing()

    # Register custom collectors in default registry – already added by default via module import 