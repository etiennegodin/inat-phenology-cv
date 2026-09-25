from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pandas as pd
from pandas.core.api import DataFrame as DataFrame

from ..config import ANCESTOR_ID, OBSERVATIONS_FIELDS, IngestPhotosParams
from ..data import DuckDBAdapter, DuckDbSQL
from ..inat_client import (
    BinaryFetcher,
    DuckDbWriter,
    EndpointConfig,
    LocalBinaryWriter,
    PhotoConfig,
    RateLimiterFetcher,
    make_client,
)
from ..utils import df_img_to_path

logger = logging.getLogger(__name__)


class InatDataSynchronizer:
    """
    Synchronizes iNaturalist observation metadata and photos with local storage.
    Ensures images for requested observations exist locally on disk.
    """

    def __init__(self, db_path: str, photo_target_dir: str, sql_dir: str):
        self.db_path = db_path
        self.photo_target_dir = photo_target_dir
        self.sql_dir = sql_dir

    def prepare_observation_images(
        self,
        urls: list[str] | str,
        photos_params: IngestPhotosParams | None = None,
        rate: int = 10,
    ) -> pd.DataFrame:
        """
        End-to-end sync:
        1. Extract IDs from URLs
        2. Fetch & stage metadata in DuckDB via inat_client (BatchEndpointClient)
        3. Query photo IDs and check local disk
        4. Download missing photos via inat_client (PhotoClient)
        5. Return DataFrame with columns: ['observation_id', 'paths']
        """

        if isinstance(urls, str):
            urls = [urls]
        if photos_params is None:
            photos_params = IngestPhotosParams()

        # Pull ids from urls
        obs_ids = self._format_observations_ids(urls)

        self._fetch_stage_observations(obs_ids)

        photos_df = self._get_photo_ids(obs_ids)

        missing_photos_df = self._filter_local_photos(photos_df)

        # Download missing photos for inference using inat_client PhotoClient
        if missing_photos_df is not None and not missing_photos_df.empty:
            self.download_photos(
                photo_ids=missing_photos_df["photo_id"].tolist(),
                target_dir=self.photo_target_dir,
                rate=rate,
                params=photos_params,
            )

        # Construct images paths
        df = df_img_to_path(photos_df, self.photo_target_dir, column_name="paths")

        # Collapse df by observation
        df = (
            df.groupby("observation_id")
            .agg({"paths": list})
            .reset_index(drop=False)
            .sort_values(by="observation_id")
            .reset_index(drop=True)
        )
        logger.debug(df)
        return df

    def _format_observations_ids(self, urls: list[str]) -> list[int]:
        ids = [int(u.split(sep="/")[-1]) for u in urls]
        logger.debug(ids)
        return ids

    def _init_sql_api(self, con):
        return DuckDbSQL(con, self.sql_dir)

    def _fetch_stage_observations(self, obs_ids: list[int]):

        with DuckDBAdapter(self.db_path) as con:
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

    def _get_photo_ids(self, obs_ids: list[int]):

        with DuckDBAdapter(self.db_path) as con:
            sql_api = self._init_sql_api(con)

            # Get observation data
            df = sql_api.fetch_df_query(
                """
                        SELECT *
                        FROM serving.observations
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
                        FROM serving.photos
                    """
            )

        df_out = df_img[df_img["observation_id"].isin(df_filtered["observation_id"])]
        logger.debug(df_out)
        return df_out

    def _filter_local_photos(self, df: pd.DataFrame) -> pd.DataFrame | None:
        missing = []
        local = []
        for p in df["photo_id"].to_list():
            matches = list(Path(self.photo_target_dir).glob(f"{p}.*"))
            if matches == []:
                missing.append(p)
            else:
                local.append(p)

        # Update record table
        logger.info("Updating serving.photos with local photo ids")
        with DuckDBAdapter(self.db_path) as con:
            if len(local) > 0:
                placeholders_local = ",".join(["?"] * len(local))
                con.execute(
                    f"""
                    UPDATE serving.photos
                    SET downloaded = ?
                    WHERE photo_id IN ({placeholders_local})
                    """,
                    [1, *local],
                )

            if len(missing) > 0:
                placeholders_missing = ",".join(["?"] * len(missing))
                con.execute(
                    f"""
                    UPDATE serving.photos
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

    def download_photos(
        self,
        photo_ids: list[int | str],
        target_dir: str,
        rate: int,
        params: IngestPhotosParams,
    ) -> None:
        """Download missing photos via inat_client PhotoClient."""
        config = PhotoConfig.from_params(params)
        fetcher = BinaryFetcher(
            fallback_extensions=params.extensions, rate=rate, max_retries=0
        )
        with LocalBinaryWriter(target_dir) as writer:
            client = make_client(config, fetcher, writer)
            asyncio.run(client.execute(photo_ids))
