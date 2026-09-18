import logging

import aiohttp

from ...infra.network import BaseRateLimiterFetcher

logger = logging.getLogger(__name__)


class RateLimiterFetcher(BaseRateLimiterFetcher):
    """
    Specialized fetcher for iNaturalist JSON API.
    Handles iNat-specific 'empty' responses for 404s/500s.
    """

    def __init__(
        self,
        rate: int,
        ignore_not_found: bool,
        **kwargs,
    ):
        super().__init__(rate=rate, **kwargs)
        self.ignore_not_found = ignore_not_found

    async def fetch(self, session: aiohttp.ClientSession, url: str, params: dict):
        """Fetch data for multiple IDs in a single request
        using comma-separated ID string"""

        async def _handle_inat_json(response: aiohttp.ClientResponse):
            if response.status in (404, 500) and self.ignore_not_found:
                logger.debug(
                    "HTTP %d for %s — returning empty response",
                    response.status,
                    url,
                )
                return self._empty_response(params)

            response.raise_for_status()
            return await response.json()

        return await self._fetch_with_retries(
            session, url, _handle_inat_json, params=params, timeout=10
        )

    def _empty_response(self, params: dict) -> dict:
        """Mirrors iNat's response shape so the pipeline never
        needs to special-case it."""
        return {
            "total_results": 0,
            "page": 1,
            "per_page": params.get("per_page", 0),
            "results": [],
        }


class BinaryFetcher(BaseRateLimiterFetcher):
    """
    Generic fetcher for raw binary data (e.g. photos).
    """

    def __init__(
        self,
        fallback_extensions: list[str],
        rate: int = 60,
        period: int = 60,
        max_retries: int = 3,
        base_delay: float = 2,
        max_delay: float = 60,
        backoff_factor: float = 2,
    ):
        super().__init__(
            rate, period, max_retries, base_delay, max_delay, backoff_factor
        )
        self.fallback_extensions = fallback_extensions

    async def fetch(self, session: aiohttp.ClientSession, url: str):
        async def _handle_binary(response: aiohttp.ClientResponse):
            response.raise_for_status()
            return await response.read()

        return await self._fetch_with_retries(
            session,
            url,
            _handle_binary,
            timeout=30,
            fallback_extensions=self.fallback_extensions,
        )


class MockFetcher:
    def __init__(self, fixture: dict):
        self.fixture = fixture

    async def fetch(self, session: aiohttp.ClientSession, url: str, params: dict):
        print(url)
        async with session.get(url, params=params, timeout=10) as response:
            print(response.url)
            return await response.json()
