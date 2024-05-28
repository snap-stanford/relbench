from datetime import datetime

import pandas as pd
import pytz

from relbench.data import Table


def test_table():
    table = Table(df=pd.DataFrame(), fkey_col_to_pkey_table={})
    assert len(table) == 0


def test_table_timezone():
    table = Table(
        df=pd.DataFrame(
            {"timestamp": [datetime(2021, 1, 1, tzinfo=pytz.utc) for _ in range(5)]}
        ),
        fkey_col_to_pkey_table={},
    )

    table.upto(pd.Timestamp("2022-01-01"))

    table = Table(
        df=pd.DataFrame(
            {"timestamp": [datetime(2021, 1, 1, tzinfo=None) for _ in range(5)]}
        ),
        fkey_col_to_pkey_table={},
    )

    table.upto(pd.Timestamp("2022-01-01", tzinfo=pytz.utc))

    table = Table(
        df=pd.DataFrame(
            {"timestamp": [datetime(2021, 1, 1, tzinfo=pytz.utc) for _ in range(5)]}
        ),
        fkey_col_to_pkey_table={},
    )
    table.upto(pd.Timestamp("2022-01-01", tzinfo=pytz.timezone("America/Los_Angeles")))
