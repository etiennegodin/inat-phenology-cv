import logging

import pandas as pd
from pydantic import BaseModel, FilePath, ValidationError

from ..config import ANGIOSPERMAE_ID

logger = logging.getLogger(__name__)


class ObservationRecord(BaseModel):
    observation_id: int
    uuid: str
    ancestor_ids: list[int]
    paths: list[FilePath]

    @property
    def is_flowering_plant(self) -> bool:
        """Tag as flowering plant record"""
        return ANGIOSPERMAE_ID in self.ancestor_ids


def validate_records(raw_records: list[dict]) -> list[ObservationRecord]:
    valid = []
    for raw in raw_records:
        try:
            record = ObservationRecord(**raw)
            if record.is_flowering_plant:
                valid.append(record)
            else:
                logger.info(
                    f"Skipping observation {raw.get('observation_id')} "
                    "since not a flowering plant"
                )

        except ValidationError as e:
            logger.warning(f"Skipping observation {raw.get('observation_id')}: {e}")
    return valid


def records_to_dataframe(records: list[ObservationRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        {"observation_id": r.observation_id, "paths": r.paths} for r in records
    )
