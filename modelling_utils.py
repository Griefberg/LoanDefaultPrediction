import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Tuple, Union

import catboost as cb
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from config import settings


def get_top_features(
    clf: cb.CatBoostClassifier, columns: list, n_features: int = 20
) -> dict:
    """Get N top features by classifier"""
    importances = np.abs(clf.feature_importances_)
    importance_prop = [x / sum(importances) for x in importances]
    return {
        col: imp
        for imp, col in sorted(zip(importance_prop, columns), reverse=True)[
            0:n_features
        ]
    }


def get_selected_precision_metrics(
    val_threshold_metrics_df: pd.DataFrame, precision_threshold: float
) -> dict:
    val_threshold_metrics_df = val_threshold_metrics_df.copy()
    val_threshold_metrics_df["precision_round"] = round(
        val_threshold_metrics_df.precision, 2
    )
    val_threshold_metrics_df = val_threshold_metrics_df.sort_values("threshold")
    selected_threshold = None
    for i in range(len(val_threshold_metrics_df)):
        if val_threshold_metrics_df.precision.iloc[i] == precision_threshold:
            selected_threshold = val_threshold_metrics_df.threshold.iloc[i]
            break
        elif val_threshold_metrics_df.precision.iloc[i] > precision_threshold:
            selected_threshold = val_threshold_metrics_df.threshold.iloc[i - 1]
            break
    if selected_threshold:
        return (
            val_threshold_metrics_df[
                val_threshold_metrics_df.threshold == selected_threshold
            ]
            .iloc[0]
            .to_dict()
        )
    else:
        # if all precision values are lower than desired threshold then choose the largest value
        fixed_precision_metrics = val_threshold_metrics_df[
            val_threshold_metrics_df.precision_round <= precision_threshold
        ]
        fixed_precision_metrics = fixed_precision_metrics[
            fixed_precision_metrics.precision_round
            == max(fixed_precision_metrics.precision_round)
        ]
        return fixed_precision_metrics.sort_values("threshold").iloc[0].to_dict()


def mertics_per_threshold(threshold: float, predprobs: pd.DataFrame) -> dict:
    predprobs = predprobs.copy()
    """Calulate Model Performance (e.g. precision, recall) for a given threshold"""
    predprobs["pred"] = predprobs.predprob >= threshold
    predprobs_true = predprobs[predprobs.pred == 1]

    return {
        "threshold": threshold,
        "fps_n": (predprobs_true.target == 0).sum(),
        "tps_n": (predprobs_true.target == 1).sum(),
        "precision": predprobs_true.target.sum() / predprobs_true.pred.sum(),
        "recall": predprobs_true.target.sum() / predprobs.target.sum(),
        "money_precision": (
            predprobs_true[predprobs_true.target == 1].loanamount.sum()
            / predprobs_true.loanamount.sum()
        ),
        "money_recall": predprobs_true.loanamount.sum() / predprobs.loanamount.sum(),
    }


def get_fixed_precision_threshold(
    clf: cb.CatBoostClassifier,
    x_val_df: pd.DataFrame,
    y_val: pd.Series,
    predictors: list,
    precision_th: float,
) -> dict:
    """
    Get the threshold which has a precision score value maximally closed to to a given
    precision threshold
    """
    x_val_df = x_val_df.copy()
    validation_predprob = clf.predict_proba(x_val_df[predictors])

    thresholds = [x / 400 for x in range(0, 400)]
    predprobs = pd.DataFrame(
        {
            "predprob": validation_predprob[:, 1],
            "target": y_val,
            "loanamount": x_val_df.loanamount.values,
        }
    )
    avg_precision = metrics.average_precision_score(
        y_true=predprobs.target, y_score=predprobs.predprob
    )
    threshold_metrics_lists = [
        mertics_per_threshold(threshold, predprobs) for threshold in thresholds
    ]
    threshold_metrics = pd.DataFrame(threshold_metrics_lists)
    threshold_metrics.sort_values("precision", inplace=True)
    fixed_precision_metrics = get_selected_precision_metrics(
        threshold_metrics, precision_th
    )
    fixed_precision_metrics["avg_precision"] = avg_precision
    logging.info(f"Selected Precision Threshold: {settings.precision_threshold}")
    logging.info(f"Selected TPs: {fixed_precision_metrics['tps_n']}")
    logging.info(f"Selected FPs: {fixed_precision_metrics['fps_n']}")
    logging.info(
        f"Precision (Fixed at Equals or less {settings.precision_threshold}): "
        + str(fixed_precision_metrics["precision"])
    )
    logging.info(f"Selected Recall: {fixed_precision_metrics['recall']}")
    logging.info(
        f"Selected Money Precision: {fixed_precision_metrics['money_precision']}"
    )
    logging.info(f"Selected Money Recall: {fixed_precision_metrics['money_recall']}")
    return fixed_precision_metrics


def train_model(
    param_grid: dict,
    features: list,
    x_df: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    random_state: int = settings.random_state,
    scoring_metric: str = "average_precision",
) -> Tuple[cb.CatBoostClassifier, float]:
    """Train model with cross validation and return best model"""
    init_clf = cb.CatBoostClassifier(
        random_seed=random_state, loss_function="Logloss", verbose=10
    )
    clf = GridSearchCV(
        init_clf,
        param_grid,
        scoring=scoring_metric,
        refit=True,
        cv=TimeSeriesSplit(n_splits=2).split(x_df),
        n_jobs=4,
    )
    clf.fit(x_df[features], y, sample_weight=weights)
    logging.info(f"Best Parameters:\n {clf.best_params_}")
    logging.info(f"Average Precision score (CV): {clf.best_score_}")
    best_model = clf.best_estimator_
    return best_model, clf.best_score_


def get_best_model(
    x_train_df: pd.DataFrame,
    y_train: pd.Series,
    x_val_df: pd.DataFrame,
    y_val: pd.Series,
    weights_train: Union[list, pd.Series],
    precision_threshold: float,
    random_state: int = settings.random_state,
) -> Tuple[cb.CatBoostClassifier, list, float, dict, dict]:
    t0 = datetime.now()
    x_train_df = x_train_df.copy()
    y_train = y_train.copy()
    x_val_df = x_val_df.copy()
    y_val = y_val.copy()

    predictors = x_train_df.columns.to_list()

    # train model on all features
    logging.info("Start to train a model with all features")
    # param_grid = {
    #     "iterations": [700, 1000],
    #     "max_depth": [8, 10],
    #     "learning_rate": [0.05, 0.07],
    #     "l2_leaf_reg": [1, 10],
    #     "random_strength": [1, 10],
    #     "min_data_in_leaf": [1, 2],
    #     "colsample_bylevel": [1],
    #     "subsample": [1],
    #     "auto_class_weights": ["None", "Balanced"],
    # }
    # temporary fixed grid
    param_grid = {
        "iterations": [1000],
        "max_depth": [10],
        "learning_rate": [0.07],
        "l2_leaf_reg": [1],
        "random_strength": [1],
        "min_data_in_leaf": [1],
        "colsample_bylevel": [1],
        "subsample": [1],
        "auto_class_weights": ["None"],
    }
    best_model, cv_average_precision = train_model(
        param_grid,
        predictors,
        x_train_df,
        y_train,
        weights_train,
        random_state=random_state,
    )

    # best features
    best_features = get_top_features(best_model, predictors)

    # choose threshold with fixed precision
    fixed_metrics_validation = get_fixed_precision_threshold(
        best_model,
        x_val_df,
        y_val,
        predictors,
        precision_threshold,
    )
    logging.info(f"Retraining Model Time: {datetime.now() - t0}")

    return (
        best_model,
        predictors,
        fixed_metrics_validation["threshold"],
        fixed_metrics_validation,
        best_features,
    )


def object_to_json(x: Any, file_name: str) -> None:
    with open(f"artefacts/{file_name}", "w") as f:
        json.dump(x, f)


def save_artefacts(
    model: cb.CatBoostClassifier,
    features: list,
    model_threshold: float,
    performance_metrics: dict,
    best_features: dict,
    feature_dicts: dict,
) -> None:
    if os.path.isdir("artefacts"):
        shutil.rmtree("artefacts")
    os.mkdir("artefacts")

    model.save_model("artefacts/catboost_model.cbm")
    object_to_json(features, "features.json")
    object_to_json(model_threshold, "model_threshold.json")
    object_to_json(performance_metrics, "performance_metrics.json")
    object_to_json(best_features, "best_features_dict.json")
    object_to_json(feature_dicts, "feature_dicts.json")
