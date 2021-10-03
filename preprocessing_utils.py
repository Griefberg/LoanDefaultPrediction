import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import settings


def handle_missings(
    df: pd.DataFrame, should_drop: bool = False, threshold: float = 99.9
) -> pd.DataFrame:
    """Replace missings from with 0 or unknown; Remove  columns with missing % > threshold"""
    missings_perc = df.isnull().sum() * 100 / len(df)
    columns_to_replace_na = missings_perc[missings_perc > 0].keys()
    cat_columns_to_replace_na = [
        col for col in columns_to_replace_na if df[col].dtype.kind not in "if"
    ]
    num_columns_to_replace_na = [
        col for col in columns_to_replace_na if df[col].dtype.kind in "if"
    ]
    df[cat_columns_to_replace_na] = df[cat_columns_to_replace_na].fillna("unknown")
    df[num_columns_to_replace_na] = df[num_columns_to_replace_na].fillna(0)
    if should_drop:
        columns_to_drop = missings_perc[missings_perc > threshold].keys()
        df.drop(columns=columns_to_drop, inplace=True)
        logging.info("\nDropped Highly Missing Features: " + ", ".join(columns_to_drop))
    return df


def drop_highly_correlated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude multicollinearity (if two features are highly correlated
    then we don't get useful information)
    """
    predictors = sorted([x for x in df.columns if x != "target"], reverse=True)
    df[predictors].head()
    corrs = np.corrcoef(df[predictors].values.astype(float), rowvar=False)
    corr_matrix = pd.DataFrame(
        corrs, columns=df[predictors].columns, index=df[predictors].columns
    ).abs()
    correlated_features = []
    for feature_name in predictors:
        if (corr_matrix.loc[feature_name] > 0.999).sum() > 1:
            correlated_features.append(feature_name)
            corr_matrix.drop([feature_name], axis=1, inplace=True)
    df.drop(correlated_features, axis=1, inplace=True)
    logging.info("\nDropped Correlated Features: " + ", ".join(correlated_features))
    return df


def drop_near_zero_features(df: pd.DataFrame) -> pd.DataFrame:
    freq = (df == 0).sum() / len(df)
    nero_zero_features = [
        x for x in freq[freq > 0.999].index.tolist() if x != "target"
    ]  # it was 0.995 before
    logging.info("\nDropped Near Zero Features: " + ", ".join(nero_zero_features))
    return df.drop(nero_zero_features, axis=1)


def preprocess_train_features(
    df: pd.DataFrame, max_categories_to_binarize: int = 100
) -> Tuple[pd.DataFrame, dict]:
    """
    Binarize categorical variables when number of categories < max_categories_to_binarize.
    Otherwise, map category value to its value counts in the training data set.
    """
    freq_dicts = dict()
    ids = [col for col in df.columns if "_id" in col]
    categoricals = [
        col for col in df.columns if (df[col].dtype == "object") & (col not in ids)
    ]
    dummy_cats = [
        x for x in categoricals if df[x].nunique() < max_categories_to_binarize
    ]
    freq_cats = [x for x in categoricals if x not in dummy_cats]
    df = pd.concat([df, pd.get_dummies(df[dummy_cats], prefix=dummy_cats)], axis=1)
    for col in freq_cats:
        counts_dict = df[col].value_counts().to_dict()
        df[col + "_freq"] = df[col].map(counts_dict)
        freq_dicts[col] = counts_dict
    df.drop(columns=categoricals, axis=1, inplace=True)
    df.columns = df.columns.str.lower()
    return df, freq_dicts


def prepare_train_df(
    df: pd.DataFrame,
    columns_to_drop: List[str] = [],
    should_drop_highly_missings_features: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    df = df.copy()
    df.columns = df.columns.str.lower()
    df.drop(columns=columns_to_drop, inplace=True)
    df = handle_missings(df, should_drop=should_drop_highly_missings_features)
    df, freq_dicts = preprocess_train_features(df)
    df.index = range(len(df))
    df = df.select_dtypes(include=["float64", "int64", "bool", "uint8"])
    df = drop_near_zero_features(df)
    df = drop_highly_correlated_features(df)
    df.fillna(0, inplace=True)

    return df, freq_dicts


def add_absent_features(df: pd.DataFrame, required_features: list) -> pd.DataFrame:
    df = df.copy()
    features_absent = list(set(required_features) - set(df.columns))
    for col in features_absent:
        df[col] = None
    return df


def preprocess_test_features(df: pd.DataFrame, freq_dicts: dict) -> pd.DataFrame:
    """
    Map category value to its value counts from the training data set.
    Otherwise, binarize it.
    """
    ids = [col for col in df.columns if "_id" in col]
    for col in ids:
        df[col] = df[col].astype(str)
    categoricals = [
        col for col in df.columns if (df[col].dtype == "object") & (col not in ids)
    ]
    freq_cats = [col for col in df.columns if col in freq_dicts.keys()]
    for col in freq_cats:
        df[col] = df[col].astype(str)
    freq_names = [x + "_freq" for x in freq_cats]
    dummy_cats = [col for col in categoricals if col not in freq_cats]
    df = pd.concat([df, pd.get_dummies(df[dummy_cats], prefix=dummy_cats)], axis=1)
    df[freq_names] = df[freq_cats].apply(
        lambda row: row.map(freq_dicts[row.name]), axis=0
    )
    df.drop(columns=categoricals, axis=1, inplace=True)
    df.columns = df.columns.str.lower()
    return df


def prepare_test_df(
    df: pd.DataFrame,
    freq_dicts: dict,
    features: list,
    should_drop_highly_missings_features: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = handle_missings(df, should_drop=should_drop_highly_missings_features)
    df = preprocess_test_features(df, freq_dicts)
    df = add_absent_features(df, features)
    df.fillna(0, inplace=True)
    return df[features]


def prepare_dfs_by_time(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict, list]:
    """
    Split data set on train and validation without shuffle to
    keep time series structure.
    """
    df.sort_values("creationdate", inplace=True)
    x_train, x_validation, y_train, y_validation = train_test_split(
        df[[col for col in df.columns if col != "target"]],
        df.target,
        test_size=0.2,
        random_state=settings.random_state,
    )
    weights_train = x_train.loanamount

    # prepare train and validation datasets
    x_train, freq_dicts = prepare_train_df(x_train)
    predictors = [col for col in x_train.columns if col != "target"]

    x_validation = prepare_test_df(x_validation, freq_dicts, predictors)

    return (
        x_train[predictors],
        x_validation[predictors],
        y_train,
        y_validation,
        freq_dicts,
        weights_train,
    )
