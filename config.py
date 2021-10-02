from pathlib import Path

from dynaconf import Dynaconf

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
)
settings.data_folder = str(Path(__file__).resolve().parent / "data")
settings.loan_data_path = settings.data_folder + "/perf.csv"
settings.prev_loan_data_path = settings.data_folder + "/prevloans.csv"
settings.demographics_loan_data_path = settings.data_folder + "/demographics.csv"
