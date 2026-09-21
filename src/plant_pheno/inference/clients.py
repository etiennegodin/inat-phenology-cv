import asyncio
from typing import TYPE_CHECKING, Dict, List

import aiohttp
from pandas.core.api import DataFrame as DataFrame
from tqdm.asyncio import tqdm_asyncio

from ..config import OBSERVATIONS_FIELDS, IngestPhotosParams
from ..data import DuckDBAdapter, DuckDbSQL
from .base import BaseInferenceClient
from .inat_client import (
    BinaryFetcher,
    DuckDbWriter,
    EndpointConfig,
    LocalBinaryWriter,
    RateLimiterFetcher,
    make_client,
)

if TYPE_CHECKING:
    pass


class InatInferenceClient(BaseInferenceClient):
    def _format_observations_ids(self, urls: list[str]):
        return [u.split(sep="/")[-1] for u in urls]

    def _init_sql_api(self, con):
        return DuckDbSQL(con, self.paths.sql_dir)

    async def _download_photo(
        self,
        session: aiohttp.ClientSession,
        fetcher: BinaryFetcher,
        writer: LocalBinaryWriter,
        item_id: str,
        params: IngestPhotosParams,
    ):
        """Orchestrate the download and write of a single photo."""

        extension = params.extensions[0]
        size = params.size

        url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{item_id}/{size}{extension}"
        filename = f"{item_id}{extension}"

        try:
            data = await fetcher.fetch(session, url)
            await writer.write(data, filename)
        except Exception as e:
            print("Failed to download photo %s: %s", item_id, e)

    async def download_photos_async(
        self, items: List[Dict], target_dir: str, rate: int, params: IngestPhotosParams
    ):
        """Async execution of the photo download batch."""
        fetcher = BinaryFetcher(
            fallback_extensions=params.extensions, rate=rate, max_retries=0
        )
        writer = LocalBinaryWriter(target_dir)

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._download_photo(
                    session=session,
                    fetcher=fetcher,
                    writer=writer,
                    item_id=str(item[params.item_id]),
                    params=params,
                )
                for item in items
            ]
            await tqdm_asyncio.gather(*tasks)

        writer.close()

    def get_observation_data(self, obs_ids: list[int]):
        with DuckDBAdapter(self.paths.inference_db_path) as con:
            sql_api = self._init_sql_api(con)
            sql_api.execute(
                script_name="create_api_raw_table",
                module="api",
                table_name="raw.inat_api",
            )

            config = EndpointConfig(
                "observations",
                write_empty_rows=True,
                fields=OBSERVATIONS_FIELDS,
                chunk_size=200,
                per_page=200,
            )

            fetcher = RateLimiterFetcher(rate=10, ignore_not_found=True)
            with DuckDbWriter(con, "raw.inat_api") as writer:
                client = make_client(config, fetcher, writer)
                asyncio.run(client.execute(obs_ids))

            sql_api.execute(script_name="stage_inat_requests", module="stage")
            sql_api.execute(script_name="stage_inat_request_photos", module="stage")

    def get_missing_photos(self):

        with DuckDBAdapter(self.paths.inference_db_path) as con:
            sql_api = self._init_sql_api(con)
            df_photos = sql_api.fetch_df_query(
                "SELECT photo_id FROM staged.inat_request_photos"
            )
        print(df_photos)
        return df_photos.to_dict("records")

    def execute(
        self, urls: list[str], photos_params: IngestPhotosParams, rate: int = 10
    ):
        obs_ids = self._format_observations_ids(urls)
        self.get_observation_data(obs_ids)

        photos_ids = self.get_missing_photos()

        asyncio.run(
            self.download_photos_async(
                items=photos_ids,
                target_dir=self.paths.photo_target_dir,
                rate=rate,
                params=photos_params,
            )
        )
