import pandas as pd
from rtb.data import Table


def test_table():
    table = Table(df=pd.DataFrame(), fkeys={})
    assert len(table) == 0
