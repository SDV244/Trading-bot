"""Simple in-process API rate limiting middleware."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from packages.core.config import get_settings


@dataclass(slots=True)
class _RateLimitBucket:
    timestamps: deque[float]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window request limiter by client address."""

    WINDOW_SECONDS = 60.0

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._buckets: dict[str, _RateLimitBucket] = {}
        self._lock = threading.Lock()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        settings = get_settings()
        if not settings.api.rate_limit_enabled:
            return await call_next(request)

        path = request.url.path
        method = request.method.upper()
        if method == "OPTIONS" or self._is_exempt_path(path, settings.api.rate_limit_exempt_path_list):
            return await call_next(request)

        is_auth_path = path.startswith("/api/auth/")
        limit = (
            settings.api.rate_limit_auth_requests_per_minute
            if is_auth_path
            else settings.api.rate_limit_requests_per_minute
        )

        forwarded_for = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        ip = forwarded_for or (request.client.host if request.client else "unknown")
        bucket_key = f"{'auth' if is_auth_path else 'api'}:{ip}"
        now = time.monotonic()

        with self._lock:
            bucket = self._buckets.setdefault(bucket_key, _RateLimitBucket(timestamps=deque()))
            while bucket.timestamps and (now - bucket.timestamps[0]) > self.WINDOW_SECONDS:
                bucket.timestamps.popleft()

            remaining_before = max(0, limit - len(bucket.timestamps))
            if len(bucket.timestamps) >= limit:
                retry_after = 1
                if bucket.timestamps:
                    retry_after = max(
                        1,
                        int(math.ceil(self.WINDOW_SECONDS - (now - bucket.timestamps[0]))),
                    )
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "category": "auth" if is_auth_path else "api",
                        "limit_per_minute": limit,
                        "retry_after_seconds": retry_after,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Window": str(int(self.WINDOW_SECONDS)),
                    },
                )

            bucket.timestamps.append(now)
            remaining_after = max(0, remaining_before - 1)

        response = await call_next(request)
        response.headers.setdefault("X-RateLimit-Limit", str(limit))
        response.headers.setdefault("X-RateLimit-Remaining", str(remaining_after))
        response.headers.setdefault("X-RateLimit-Window", str(int(self.WINDOW_SECONDS)))
        return response

    @staticmethod
    def _is_exempt_path(path: str, exempt_prefixes: list[str]) -> bool:
        return any(path.startswith(prefix) for prefix in exempt_prefixes)
