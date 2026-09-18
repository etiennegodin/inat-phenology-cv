import asyncio
import logging
import random
from typing import Any, Callable, Coroutine

import aiohttp
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

THROTTLE_STATUSES = {429, 503}


class BaseRateLimiterFetcher:
    """
    Base class for rate-limited fetching with exponential backoff.
    Logic for retries and throttling is centralized here.
    """

    def __init__(
        self,
        rate: int = 60,
        period: int = 60,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
    ):
        self.limiter = AsyncLimiter(rate, period)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    async def _fetch_with_retries(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        handler: Callable[[aiohttp.ClientResponse], Coroutine[Any, Any, Any]],
        fallback_extensions: list[str] | None = None,
        **kwargs,
    ) -> Any:
        """
        Internal retry loop.
        'handler' is a coroutine that processes the response (e.g., response.json())
        """
        if fallback_extensions is None:
            fallback_extensions = []

        last_exc = None

        urls_to_try = [base_url] + [
            self._swap_extension(base_url, ext) for ext in fallback_extensions
        ]
        for url in urls_to_try:
            async with self.limiter:
                for attempt in range(self.max_retries + 1):
                    try:
                        async with session.get(url, **kwargs) as response:
                            if response.status in THROTTLE_STATUSES:
                                retry_after = response.headers.get("Retry-After")
                                wait = (
                                    float(retry_after)
                                    if retry_after
                                    else self._backoff(attempt)
                                )
                                logger.warning(
                                    "Throttled (HTTP %d) on %s try %d — waiting %.1fs",
                                    response.status,
                                    url,
                                    attempt,
                                    wait,
                                )
                                await asyncio.sleep(wait)
                                continue

                            # Let the handler decide what to do with the response
                            # (including raising for status)
                            return await handler(response)

                    except aiohttp.ClientError as e:
                        last_exc = e
                        if attempt == self.max_retries:
                            break
                        wait = self._backoff(attempt)
                        logger.warning(
                            "Request error on %s attempt %d — waiting %.1fs: %s",
                            url,
                            attempt,
                            wait,
                            e,
                        )
                        await asyncio.sleep(wait)
            # All retries failed for this URL; try next extension

        raise RuntimeError(
            f"Failed after {self.max_retries} retries "
            f"for {base_url} and extensions {fallback_extensions}"
        ) from last_exc

    def _backoff(self, attempt: int) -> float:
        delay = self.base_delay * (self.backoff_factor**attempt)
        jitter = random.uniform(0, self.base_delay)
        return min(delay + jitter, self.max_delay)

    def _swap_extension(self, url: str, new_extension: str) -> str:
        """Replace the file extension in the URL."""
        base = url.rsplit(".", 1)[0] if "." in url.rsplit("/", 1)[-1] else url
        return f"{base}{new_extension}"
