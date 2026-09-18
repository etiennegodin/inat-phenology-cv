import asyncio
import logging
import random

import aiohttp
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

THROTTLE_STATUSES = {429, 503}  # 429 = rate limited, 503 = service unavailable


class RateLimiterFetcher:
    def __init__(
        self,
        rate: int,
        ignore_not_found: bool,
        max_retries: int = 3,
        base_delay: float = 2.0,  # seconds
        max_delay: float = 60.0,  # cap so you don't wait 10 minutes
        backoff_factor: float = 2.0,
    ):
        # Convert limiter value to AsyncLimiter object
        self.limiter = AsyncLimiter(rate, 60)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.ignore_not_found = ignore_not_found

    async def fetch(self, session: aiohttp.ClientSession, url: str, params: dict):
        """Fetch data for multiple IDs in
        a single request using comma-separated ID string"""

        last_exc = None

        async with self.limiter:
            for attempt in range(self.max_retries + 1):
                try:
                    async with session.get(url, params=params, timeout=10) as response:
                        if response.status in THROTTLE_STATUSES:
                            # Respect Retry-After header if server sends one
                            retry_after = response.headers.get("Retry-After")
                            wait = (
                                float(retry_after)
                                if retry_after
                                else self._backoff(attempt)
                            )
                            logger.warning(
                                "Throttled (HTTP %d) on attempt %d — waiting %.1fs",
                                response.status,
                                attempt,
                                wait,
                            )
                            await asyncio.sleep(wait)  # reactive: server said stop
                            continue

                        if response.status == 500 and self.ignore_not_found:
                            logger.debug(
                                "HTTP 500 for %s %s — returning empty response",
                                url,
                                params,
                            )
                            return self._empty_response(params)  # ← bail out cleanly

                        response.raise_for_status()
                        return await response.json()

                except aiohttp.ClientError as e:
                    last_exc = e
                    if attempt == self.max_retries:
                        break
                    wait = self._backoff(attempt)
                    logger.warning(
                        "Request error on attempt %d — waiting %.1fs: %s",
                        attempt,
                        wait,
                        e,
                    )
                    await asyncio.sleep(wait)

            raise RuntimeError(
                f"Failed after {self.max_retries} retries for {url} {params}"
            ) from last_exc

    def _backoff(self, attempt: int) -> float:
        delay = self.base_delay * (self.backoff_factor**attempt)
        jitter = random.uniform(0, self.base_delay)  # jitter ≤ base_delay
        return min(delay + jitter, self.max_delay)

    def _empty_response(self, params: dict) -> dict:
        """Mirrors iNat's response shape so
        the pipeline never needs to special-case it."""
        return {
            "total_results": 0,
            "page": 1,
            "per_page": params.get("per_page", 0),
            "results": [],
        }


class MockFetcher:
    def __init__(self, fixture: dict):
        self.fixture = fixture

    async def fetch(self, session: aiohttp.ClientSession, url: str, params: dict):
        return self.fixture
