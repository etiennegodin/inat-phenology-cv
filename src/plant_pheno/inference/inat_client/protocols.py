from typing import Protocol, runtime_checkable

import aiohttp


@runtime_checkable
class JsonFetcherProtocol(Protocol):
    async def fetch(
        self, session: aiohttp.ClientSession, url: str, params: dict
    ) -> dict:
        """Make an HTTP request, return parsed JSON."""
        ...


@runtime_checkable
class BinaryFetcherProtocol(Protocol):
    async def fetch(self, session: aiohttp.ClientSession, url: str) -> bytes:
        """Make an HTTP request, return raw binary bytes."""
        ...


@runtime_checkable
class JsonWriterProtocol(Protocol):
    async def write(self, results: list[dict]) -> None:
        """Persist a page of results."""
        ...


@runtime_checkable
class BinaryWriterProtocol(Protocol):
    async def write(self, data: bytes, filename: str) -> None:
        """Persist raw binary bytes with a filename."""
        ...
