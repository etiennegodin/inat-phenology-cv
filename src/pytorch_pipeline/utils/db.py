import duckdb
import pandas as pd


def get_df_from_table(db_path: str, table_name: str) -> pd.DataFrame:
    with duckdb.connect(db_path) as con:
        con = duckdb.connect(db_path)
        df = con.execute(f"SELECT * FROM {table_name}").fetch_df()
    return df
