import os

import duckdb
import pandas as pd
from PIL import Image


def clean_data(root: str):
    for root, dirs, files in os.walk(root):
        for file in files:
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                print(f"Deleting corrupted image: {path}")
                os.remove(path)


def get_df_from_table(db_path: str, table_name: str) -> pd.DataFrame:
    with duckdb.connect(db_path) as con:
        con = duckdb.connect(db_path)
        df = con.execute(f"SELECT * FROM {table_name}").fetch_df()
    return df
