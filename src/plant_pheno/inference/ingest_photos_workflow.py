import asyncio
import logging
from typing import Dict, List

import aiohttp
from tqdm.asyncio import tqdm_asyncio

from ..config import IngestPhotosParams
from ..data import DuckDBAdapter, DuckDbSQL
from .inat_client import BinaryFetcher, LocalBinaryWriter

logger = logging.getLogger(__name__)


async def _download_photo(
    session: aiohttp.ClientSession,
    fetcher: BinaryFetcher,
    writer: LocalBinaryWriter,
    item_id: str,
    subfolder: str,
    extension: str,
    size: str,
):
    """Orchestrate the download and write of a single photo."""
    url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{item_id}/{size}.{extension}"
    filename = f"{item_id}.{extension}"

    try:
        data = await fetcher.fetch(session, url)
        await writer.write(data, subfolder, filename)
    except Exception as e:
        logger.error("Failed to download photo %s: %s", item_id, e)


async def execute_async(
    items: List[Dict], parent_folder: str, rate: int, params: IngestPhotosParams
):
    """Async execution of the photo download batch."""
    fetcher = BinaryFetcher(rate=rate)
    writer = LocalBinaryWriter(parent_folder)

    async with aiohttp.ClientSession() as session:
        tasks = [
            _download_photo(
                session,
                fetcher,
                writer,
                str(item[params.item_id]),
                str(item[params.label]),
                params.extension,
                params.size,
            )
            for item in items
        ]
        await tqdm_asyncio.gather(*tasks)

    writer.close()


def execute(
    deps: Dependencies,
    rate: int,
) -> None:

    SOURCE_TABLE_NAME = "tests.cv_photos"
    """
    Workflow entry point to download a list of photos.

    Args:
        items: List of dicts with 'item_id' and 'attribute_id'.
        parent_folder: Path to save the photos.
        extension: Image extension (e.g. 'jpg').
        size: iNat image size (e.g. 'medium', 'large', 'original').
        rate: Requests per minute limit.
    """
    params = IngestPhotosParams(label="controlled_value_id")
    with DuckDBAdapter(deps.RAW_DB_PATH) as con:
        sql_api = DuckDbSQL(con, deps.SQL_API_PATH)
        # 2 Get missing items not collected
        df = sql_api.fetch_df_query(f"SELECT * FROM {SOURCE_TABLE_NAME}")
        # Convert rows to list of dicts
        items = df.to_dict(orient="records")

    logger.info("Starting photo download for %d items", len(items))
    asyncio.run(execute_async(items, deps.PHOTO_PARENT_FOLDER, rate, params))

    logger.info("Photo download complete.")
