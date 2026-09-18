from typing import Protocol

import aiohttp


class Fetcher(Protocol):
    async def fetch(
        self, session: aiohttp.ClientSession, url: str, params: dict
    ) -> dict:
        """Make an HTTP request, return parsed JSON."""
        ...


class Writer(Protocol):
    async def write(self, results: list[dict]) -> None:
        """Persist a page of results."""
        ...
