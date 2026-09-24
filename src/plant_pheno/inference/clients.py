from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import aiohttp
from pandas.core.api import DataFrame as DataFrame
from tqdm.asyncio import tqdm_asyncio

from ..config import ANCESTOR_ID, OBSERVATIONS_FIELDS, IngestPhotosParams
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
    def execute(
        self, urls: list[str], photos_params: IngestPhotosParams, rate: int = 10
    ):
        # Pull ids from urls
        obs_ids = self._format_observations_ids(urls)

        # Query observations data and photo id list
        self.get_observation_data(obs_ids)
        photos_df = self.get_photo_ids(obs_ids)

        # Filter with previously downloaded photos and trac
        filtered_photos_df = self._filter_photo_ids(photos_df)

        # Download missing photos for inference
        if filtered_photos_df is not None:
            asyncio.run(
                self.download_photos_async(
                    items=filtered_photos_df.to_dict("records"),
                    target_dir=self.params.photo_target_dir,
                    rate=rate,
                    params=photos_params,
                )
            )

        # Construct images paths
        df = df_img_to_path(
            photos_df, self.params.photo_target_dir, column_name="paths"
        )

        # Collapse df by observation
        df = (
            df.groupby("observation_id")
            .agg({"paths": list})
            .reset_index(drop=False)
            .sort_values(by="observation_id")
            .reset_index(drop=True)
        )
        logger.debug(df)

        x, y = self.model.predict(df)
        print(x, y)

    def _format_observations_ids(self, urls: list[str]) -> list[int]:
        ids = [int(u.split(sep="/")[-1]) for u in urls]
        logger.debug(ids)
        return ids

    def _init_sql_api(self, con):
        return DuckDbSQL(con, self.params.sql_dir)

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

        with DuckDBAdapter(self.params.db_path) as con:
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

    def get_photo_ids(self, obs_ids: list[int]):

        with DuckDBAdapter(self.params.db_path) as con:
            sql_api = self._init_sql_api(con)

            # Get observation data
            df = sql_api.fetch_df_query(
                """
                    SELECT *
                    FROM staged.obs_requests
                """
            )

            # Keep only flowering plants
            """
            missing_ids = {ANCESTOR_ID} - set(df["ancestor_ids"])
            if missing_ids:
                logger.warning(
                    f"Observation ids of non flowering plants: {missing_ids}"
                )
            """
            df_filtered = df[df["ancestor_ids"].apply(lambda lst: ANCESTOR_ID in lst)]

            # Keep only requested observations
            df_filtered = df_filtered[df_filtered["observation_id"].isin(obs_ids)]

            # Get photo data
            df_img = sql_api.fetch_df_query(
                """
                    SELECT *
                    FROM staged.img_requests
                """
            )

        df_out = df_img[df_img["observation_id"].isin(df_filtered["observation_id"])]
        logger.debug(df_out)
        return df_out

    def _filter_photo_ids(self, df: pd.DataFrame) -> pd.DataFrame | None:
        missing = []
        local = []
        for p in df["photo_id"].to_list():
            matches = list(Path(self.params.photo_target_dir).glob(f"{p}.*"))
            if matches == []:
                missing.append(p)
            else:
                local.append(p)

        # Update record table
        logger.info("Updating staged.img_requests with local photo ids")
        with DuckDBAdapter(self.params.db_path) as con:
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
            df_out = df[df["photo_id"].isin(missing)]
            logger.debug(df_out)
            return df_out

        logger.info("All requested images are local")
        return None
