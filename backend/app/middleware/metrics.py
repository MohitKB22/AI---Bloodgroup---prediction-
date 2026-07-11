"""Minimal request-count/latency metrics, exposed at /metrics for Prometheus
to scrape. Deliberately hand-rolled rather than pulling in a full
instrumentator library, since three metrics is all this service needs.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

REQUEST_COUNT = Counter(
    "bloodprint_http_requests_total", "Total HTTP requests", ["method", "path", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "bloodprint_http_request_duration_seconds", "Request latency", ["method", "path"]
)
PREDICTION_COUNT = Counter(
    "bloodprint_predictions_total", "Total predictions served", ["predicted_label"]
)


def register_metrics(app: FastAPI) -> None:
    @app.middleware("http")
    async def track_metrics(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        route = request.scope.get("route")
        path_template = route.path if route is not None else request.url.path
        REQUEST_COUNT.labels(request.method, path_template, response.status_code).inc()
        REQUEST_LATENCY.labels(request.method, path_template).observe(duration)
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
