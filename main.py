import pandas as pd

from config import settings

# db utils

loans = pd.read_csv(settings.loan_data_path)

prevloans = pd.read_csv(settings.prev_loan_data_path)
demographics = pd.read_csv(settings.demographics_loan_data_path)
