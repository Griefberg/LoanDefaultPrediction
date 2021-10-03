import logging
from datetime import datetime

import pandas as pd

import getting_data_utils as gt_utils
import modelling_utils
import preprocessing_utils as prep_utils
from config import settings

logging.getLogger().setLevel(logging.INFO)


def main():
    try:
        # getting data
        t = datetime.now()
        loans = gt_utils.read_and_prepare_csv(settings.loan_data_path)
        prevloans = gt_utils.read_and_prepare_csv(settings.prev_loan_data_path)
        demographics = gt_utils.read_and_prepare_csv(
            settings.demographics_loan_data_path
        )
        dem_loans = pd.merge(loans, demographics, on="customerid", how="left")
        logging.info(f"Getting Data: {datetime.now() - t}")

        # data preparation and features engineering
        t = datetime.now()
        (
            x_train,
            x_validation,
            y_train,
            y_validation,
            feature_dicts,
            weights_train,
        ) = prep_utils.prepare_dfs_by_time(
            df=dem_loans,
        )
        logging.info(
            f"Splitting data into train and validation datasets & Features Generation: {datetime.now() - t}"
        )

        # train model
        t = datetime.now()
        (
            best_model,
            features,
            model_threshold,
            performance_metrics,
            best_features_dict,
        ) = modelling_utils.get_best_model(
            x_train_df=x_train,
            y_train=y_train,
            x_val_df=x_validation,
            y_val=y_validation,
            weights_train=weights_train,
            precision_threshold=settings.precision_threshold,
        )
        logging.info(f"Training model: {datetime.now() - t}")

        # save artefacts
        modelling_utils.save_artefacts(
            best_model,
            features,
            model_threshold,
            performance_metrics,
            best_features_dict,
            feature_dicts,
        )
    except Exception as error:
        logging.critical(f"encountered an exception, {error}")
        raise error


if __name__ == "__main__":
    main()
