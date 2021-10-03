import logging
import os

import pandas as pd

from config import settings


def lowercase_all_string_values(df: pd.DataFrame) -> pd.DataFrame:
    object_columns = [x for x in df.columns if df[x].dtype == "object"]
    df[object_columns] = df[object_columns].applymap(
        lambda x: x.lower() if isinstance(x, str) else x
    )
    return df


def read_and_prepare_csv(data_path: str) -> pd.DataFrame:
    assert os.path.isfile(data_path), "No Data"
    df = pd.read_csv(data_path)
    logging.info(f"{data_path} Data Set: \n" + str(df.shape))
    df.columns = df.columns.str.lower()

    # convert to datetime
    time_cols = [col for col in df.columns if "date" in col]
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])

    df = lowercase_all_string_values(df)

    # if target in the data set – preprocess it
    if settings.target_column in df.columns:
        df["target"] = df[settings.target_column].map({"good": False, "bad": True})
        df.drop(columns=[settings.target_column], inplace=True)
        logging.info("Target Distribution: \n" + str(df.target.value_counts(True)))
    return df
