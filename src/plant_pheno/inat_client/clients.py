import logging
from typing import Any, Iterator

import aiohttp
from tqdm.asyncio import tqdm_asyncio

from .base import BaseInatClient, _chunked
from .config import EndpointConfig, PhotoConfig
from .protocols import BinaryFetcherProtocol, BinaryWriterProtocol

logger = logging.getLogger(__name__)


class BatchEndpointClient(BaseInatClient):
    """
    For endpoints like GET /observations/
    - Sends chunk_size IDs as a comma-joined 'id' param
    - Supports sparse fieldsets via ?fields=
    - Supports per_page pagination
    """

    def _iter_requests(self, ids: list) -> Iterator[tuple[Any, dict]]:
        for chunk in _chunked(ids, self.config.chunk_size):
            id_string = ",".join(str(id_) for id_ in chunk)
            logger.debug(id_string)
            # Get params from config
            params = {**self.config.params, "id": id_string}

            # Add fields string to params
            if self.config.fields:
                params["fields"] = self.config.fields

            # Add first id of chunk as fallback id
            yield chunk[0], params


class ParametrizedEndpointClient(BaseInatClient):
    """
    For endpoints like GET /observations?taxon_id=1234
    - One ID per request (chunk_size is ignored / forced to 1)
    - Named param specified by config.id_param
    - Still paginates — can return many results per ID
    """

    def __init__(self, config: EndpointConfig, fetcher, writer):
        assert config.id_param is not None, (
            "ParametrizedEndpointClient requires config.id_param"
        )
        super().__init__(config, fetcher, writer)

    def _iter_requests(self, ids: list) -> Iterator[tuple[Any, dict]]:
        for id_ in ids:  # always 1 ID per request
            yield (
                id_,
                {
                    **self.config.params,
                    self.config.id_param: id_,
                },
            )


class PhotoClient:
    """
    Client for downloading iNaturalist photo binaries (e.g. from AWS S3).
    - Takes PhotoConfig, BinaryFetcherProtocol, and BinaryWriterProtocol.
    - Concurrently downloads photos for a list of photo IDs or item records.
    """

    def __init__(
        self,
        config: PhotoConfig,
        fetcher: BinaryFetcherProtocol,
        writer: BinaryWriterProtocol,
    ):
        self.config = config
        self.fetcher = fetcher
        self.writer = writer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if hasattr(self.writer, "close"):
            self.writer.close()

    async def _download_one(self, session: aiohttp.ClientSession, item_id: str) -> None:
        extension = self.config.extensions[0] if self.config.extensions else ".jpg"
        size = self.config.size
        url = f"{self.config.base_url}/{item_id}/{size}{extension}"
        filename = f"{item_id}{extension}"

        try:
            data = await self.fetcher.fetch(session, url)
            await self.writer.write(data, filename)
        except Exception as e:
            logger.error("Failed to download photo %s: %s", item_id, e)

    async def execute(self, ids: list[Any]) -> None:
        """Download photos concurrently
        for the given list of photo IDs or dict records."""
        if not ids:
            return

        normalized_ids = [
            str(item[self.config.item_id_key]) if isinstance(item, dict) else str(item)
            for item in ids
        ]

        async with aiohttp.ClientSession() as session:
            tasks = [self._download_one(session, item_id) for item_id in normalized_ids]
            await tqdm_asyncio.gather(*tasks)
