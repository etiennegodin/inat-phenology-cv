from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import aiohttp
from pandas.core.api import DataFrame as DataFrame
from tqdm.asyncio import tqdm_asyncio

from ..config import OBSERVATIONS_FIELDS, IngestPhotosParams
from ..data import DuckDBAdapter, DuckDbSQL
from ..utils import df_img_to_path
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
    import pandas as pd

logger = logging.getLogger(__name__)


class InatInferenceClient(BaseInferenceClient):
    def _format_observations_ids(self, urls: list[str]) -> list[int]:
        return [int(u.split(sep="/")[-1]) for u in urls]

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

    def get_observation_data(self, obs_ids: list[int]) -> pd.DataFrame:
        with DuckDBAdapter(self.paths.inference_db_path) as con:
            sql_api = self._init_sql_api(con)
            sql_api.execute("init", module="inat")

            config = EndpointConfig(
                "observations",
                write_empty_rows=True,
                fields=OBSERVATIONS_FIELDS,
                chunk_size=200,
                per_page=200,
            )

            fetcher = RateLimiterFetcher(rate=10, ignore_not_found=True)
            with DuckDbWriter(con, "raw.obs_requests") as writer:
                client = make_client(config, fetcher, writer)
                asyncio.run(client.execute(obs_ids))

            sql_api.execute("stage_obs_requests", module="inat")
            sql_api.execute("stage_img_requests", module="inat")

            # Get photo ids
            df = sql_api.fetch_df(
                "get_photo_ids",
                module="inat",
            )

        return df[df["observation_id"].isin(obs_ids)]

    def filter_photo_ids(self, df: pd.DataFrame) -> pd.DataFrame | None:
        missing = []
        local = []
        for p in df["photo_id"].to_list():
            matches = list(Path(self.paths.photo_target_dir).glob(f"{p}.*"))
            if matches == []:
                missing.append(p)
            else:
                local.append(p)

        # Update record table
        logger.info("Updating staged.img_requests with local photo ids")
        with DuckDBAdapter(self.paths.inference_db_path) as con:
            if len(local) > 0:
                placeholders_local = ",".join(["?"] * len(local))
                con.execute(
                    f"""
                UPDATE staged.img_requests
                SET downloaded = ?
                WHERE photo_id IN ({placeholders_local})
                """,
                    [1, *local],
                )

            if len(missing) > 0:
                placeholders_missing = ",".join(["?"] * len(missing))
                con.execute(
                    f"""
                UPDATE staged.img_requests
                SET downloaded = ?
                WHERE photo_id IN ({placeholders_missing})
                """,
                    [0, *missing],
                )

        if len(missing) > 0:
            logger.info("Found missing photo ids")
            return df[df["photo_id"].isin(missing)]

        logger.info("All requested images are local")
        return None

    def execute(
        self, urls: list[str], photos_params: IngestPhotosParams, rate: int = 10
    ):
        # Pull ids from urls
        obs_ids = self._format_observations_ids(urls)

        # Query observations data and photo id list
        photos_df = self.get_observation_data(obs_ids)

        # Filter with previously downloaded photos
        filtered_photos_df = self.filter_photo_ids(photos_df)

        # Download missing photos for inference
        if filtered_photos_df is not None:
            asyncio.run(
                self.download_photos_async(
                    items=filtered_photos_df.to_dict("records"),
                    target_dir=self.paths.photo_target_dir,
                    rate=rate,
                    params=photos_params,
                )
            )

        # Construct images paths
        df = df_img_to_path(photos_df, self.paths.photo_target_dir, column_name="paths")

        # Collapse df by observation
        df = (
            df.groupby("observation_id")
            .agg({"paths": list})
            .reset_index(drop=False)
            .sort_values(by="observation_id")
            .reset_index(drop=True)
        )

        x, y = self.model.predict(df)
        print(x, y)
