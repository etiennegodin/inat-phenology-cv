import logging
from typing import Any

import duckdb

from ...exceptions import DBConnectionError, DBError

logger = logging.getLogger(__name__)


class DuckDBAdapter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._con: duckdb.DuckDBPyConnection | None = None

    def __enter__(self):
        logger.debug("Opening DuckDB connection: %s", self.db_path)
        try:
            self._con = duckdb.connect(
                self.db_path, config={"allow_unsigned_extensions": "true"}
            )
        except duckdb.IOException as e:
            raise DBConnectionError(str(e), file=self.db_path)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._con:
            self._con.close()
            logger.debug("DuckDB connection closed")
        return False

    def execute(self, query: str, params: Any | None = None, script: str | None = None):
        # logger.debug(query)
        # logger.debug(params)
        try:
            return self._con.execute(query, params)
        except duckdb.ParserException as e:
            raise DBError(str(e), script=script, details={"params": params}) from e
        except duckdb.InvalidInputException as e:
            raise DBError(str(e), script=script, details={"params": params}) from e
        except duckdb.CatalogException as e:
            raise DBError(str(e), script=script) from e
        except duckdb.BinderException as e:
            raise DBError(str(e), script=script) from e
        except duckdb.Error as e:
            raise DBError(str(e), script=script) from e

    def executemany(self, query: str, params: list[tuple]):
        return self._con.executemany(query, params)

    def commit(self):
        return self._con.commit()
