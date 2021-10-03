import os

import pandas as pd
from pandas.util.testing import assert_frame_equal

from config import settings
from getting_data_utils import read_and_prepare_csv


class TestClass:
    @classmethod
    def setup_class(self):
        print("setup_class called once for the class")
        self.test_data_file = f"{settings.tests_data_folder}/input_df.csv"
        self.input = {
            "customerid": {
                0: "8a2a81a74ce8c05d014cfb32a0da1049",
                1: "8a85886e54beabf90154c0a29ae757c0",
                2: "8a8588f35438fe12015444567666018e",
            },
            "systemloanid": {0: 301994762, 1: 301965204, 2: 301966580},
            "loannumber": {0: 12, 1: 2, 2: 7},
            "approveddate": {
                0: "2017-07-25 08:22:56.000000",
                1: "2017-07-05 17:04:41.000000",
                2: "2017-07-06 14:52:57.000000",
            },
            "creationdate": {
                0: "2017-07-25 07:22:47.000000",
                1: "2017-07-05 16:04:18.000000",
                2: "2017-07-06 13:52:51.000000",
            },
            "loanamount": {0: 30000.0, 1: 15000.0, 2: 20000.0},
            "totaldue": {0: 34500.0, 1: 17250.0, 2: 22250.0},
            "termdays": {0: 30, 1: 30, 2: 15},
            "referredby": {0: "AAAA", 1: None, 2: None},
            "good_bad_flag": {0: "Good", 1: "Good", 2: "Good"},
        }
        # save input as csv
        pd.DataFrame(self.input).to_csv(
            f"{settings.tests_data_folder}/input_df.csv", index=False
        )
        self.expected = {
            "customerid": {
                0: "8a2a81a74ce8c05d014cfb32a0da1049",
                1: "8a85886e54beabf90154c0a29ae757c0",
                2: "8a8588f35438fe12015444567666018e",
            },
            "systemloanid": {0: 301994762, 1: 301965204, 2: 301966580},
            "loannumber": {0: 12, 1: 2, 2: 7},
            "approveddate": {
                0: pd.to_datetime("2017-07-25 08:22:56.000000"),
                1: pd.to_datetime("2017-07-05 17:04:41.000000"),
                2: pd.to_datetime("2017-07-06 14:52:57.000000"),
            },
            "creationdate": {
                0: pd.to_datetime("2017-07-25 07:22:47.000000"),
                1: pd.to_datetime("2017-07-05 16:04:18.000000"),
                2: pd.to_datetime("2017-07-06 13:52:51.000000"),
            },
            "loanamount": {0: 30000.0, 1: 15000.0, 2: 20000.0},
            "totaldue": {0: 34500.0, 1: 17250.0, 2: 22250.0},
            "termdays": {0: 30, 1: 30, 2: 15},
            "referredby": {0: "aaaa", 1: None, 2: None},
            "target": {0: False, 1: False, 2: False},
        }
        self.expected_df = pd.DataFrame(self.expected)

    @staticmethod
    def teardown_class():
        print("teardown_class called once for the class")
        listdir = os.listdir(settings.tests_data_folder)
        for item in listdir:
            if item.endswith(".csv"):
                os.remove(os.path.join(settings.tests_data_folder, item))

    def test_read_and_prepare_csv(self):
        df = read_and_prepare_csv(self.test_data_file)
        assert_frame_equal(self.expected_df, df)
