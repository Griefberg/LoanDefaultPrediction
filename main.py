import logging
from datetime import datetime

import pandas as pd

import getting_data_utils as gt_utils
import preprocessing_utils as prep_utils
from config import settings

logging.getLogger().setLevel(logging.INFO)

# getting data
t = datetime.now()
loans = gt_utils.read_and_prepare_csv(settings.loan_data_path)
prevloans = gt_utils.read_and_prepare_csv(settings.prev_loan_data_path)
demographics = gt_utils.read_and_prepare_csv(settings.demographics_loan_data_path)
loans = pd.merge(loans, demographics, on="customerid", how="left")
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
    df=loans,
)
logging.info(
    f"Splitting data into train and validation datasets & Features Generation: {datetime.now() - t}"
)
