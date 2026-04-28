import duckdb
import pandas as pd

from .params import PathsParams


def get_df_from_table(db_path: str, table_name: str) -> pd.DataFrame:
    with duckdb.connect(db_path) as con:
        con = duckdb.connect(db_path)
        df = con.execute(f"SELECT * FROM {table_name}").fetch_df()
    return df


def update_dataset(paths: PathsParams):
    with duckdb.connect(paths.db_path) as con:
        try:
            con.execute(f"ATTACH '{paths.source_db_path}' AS source_db (READ_ONLY)")
        except duckdb.BinderException:
            pass

        con.execute(
            """CREATE OR REPLACE TABLE downloaded AS
            SELECT *
            FROM source_db.raw.ingested_photos;"""
        )
        con.execute("""CREATE OR REPLACE TABLE cv_photos AS
            SELECT *
            FROM source_db.tests.cv_photos
            WHERE photo_id IN (SELECT CAST(raw_id AS BIGINT) FROM downloaded);""")
