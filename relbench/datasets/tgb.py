from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from relbench.base import Database, Dataset


@dataclass(frozen=True)
class TGBCutoffs:
    val_timestamp_s: int
    test_timestamp_s: int


_TGB_CUTOFFS: dict[str, TGBCutoffs] = {
    # Dynamic link property prediction (tgbl-*)
    # NOTE: These cutoffs match the pre-built RelBench exports used in our
    # conversion pipeline. They are expected to be consistent with the
    # official TGB temporal split (70/15/15).
    "tgbl-wiki": TGBCutoffs(val_timestamp_s=1862653, test_timestamp_s=2218300),
    "tgbl-wiki-v2": TGBCutoffs(val_timestamp_s=1862653, test_timestamp_s=2218300),
    "tgbl-review": TGBCutoffs(val_timestamp_s=1464912000, test_timestamp_s=1488844800),
    "tgbl-review-v2": TGBCutoffs(val_timestamp_s=1464912000, test_timestamp_s=1488844800),
    "tgbl-coin": TGBCutoffs(val_timestamp_s=1662096249, test_timestamp_s=1664482319),
    "tgbl-comment": TGBCutoffs(val_timestamp_s=1282869285, test_timestamp_s=1288838725),
    "tgbl-flight": TGBCutoffs(val_timestamp_s=1638162000, test_timestamp_s=1653796800),
    # Temporal heterogeneous link prediction (thgl-*)
    "thgl-software": TGBCutoffs(val_timestamp_s=1706003880, test_timestamp_s=1706315669),
    "thgl-forum": TGBCutoffs(val_timestamp_s=1390426563, test_timestamp_s=1390838358),
    "thgl-github": TGBCutoffs(val_timestamp_s=1711075987, test_timestamp_s=1711482874),
    "thgl-myket": TGBCutoffs(val_timestamp_s=1603724860, test_timestamp_s=1606341312),
    # Dynamic node property prediction (tgbn-*)
    # Nodeprop timestamps may be stored as years in the raw sources; our export
    # normalizes timestamps to UNIX seconds.
    "tgbn-trade": TGBCutoffs(val_timestamp_s=1262304000, test_timestamp_s=1388534400),
    "tgbn-genre": TGBCutoffs(val_timestamp_s=1216427762, test_timestamp_s=1230448684),
    "tgbn-reddit": TGBCutoffs(val_timestamp_s=1279485233, test_timestamp_s=1286653871),
    "tgbn-token": TGBCutoffs(val_timestamp_s=1522889022, test_timestamp_s=1525386888),
}


class TGBDataset(Dataset):
    r"""Community dataset family: Temporal Graph Benchmark (TGB) exports.

    This dataset class expects a pre-built RelBench database at:
    `cache_dir/db/*.parquet`.

    The recommended workflow for contributions (see CONTRIBUTING.md):
    1) Materialize `db/` (Parquet) for each dataset.
    2) Zip it as `db.zip` and publish it.
    3) Add the sha256 hash to `relbench/datasets/hashes.json`.

    Notes:
    - Primary/foreign keys are stored as int64 (DBML: `bigint`) to avoid 32-bit
      overflow and to match PyArrow's default integer type.
    - We store only the relational schema + timestamps; "splits" are derived
      from cutoffs (val/test timestamps).
    """

    url = "https://tgb.complexdatalab.com/"

    def __init__(self, *, tgb_name: str, cache_dir: Optional[str] = None) -> None:
        if tgb_name not in _TGB_CUTOFFS:
            raise ValueError(
                f"Unknown tgb_name='{tgb_name}'. Known keys: {sorted(_TGB_CUTOFFS.keys())}"
            )
        self.tgb_name = str(tgb_name)

        cutoffs = _TGB_CUTOFFS[self.tgb_name]
        # TGB exports store timestamps as timezone-aware UTC (timestamp[ns, UTC]).
        # Keep dataset cutoffs in UTC as well to avoid tz-mismatch in task builders.
        self.val_timestamp = pd.to_datetime(int(cutoffs.val_timestamp_s), unit="s", utc=True)
        self.test_timestamp = pd.to_datetime(int(cutoffs.test_timestamp_s), unit="s", utc=True)

        super().__init__(cache_dir=cache_dir)

    def make_db(self) -> Database:
        if self.cache_dir is None:
            raise RuntimeError("TGBDataset requires cache_dir to locate the cached db.")

        db_dir = Path(self.cache_dir) / "db"
        if db_dir.exists() and any(db_dir.glob("*.parquet")):
            return Database.load(db_dir)

        raise RuntimeError(
            f"TGB dataset '{self.tgb_name}' not found at {db_dir}. "
            "This dataset is distributed as a pre-built RelBench database (db.zip). "
            "Please run `download_dataset(...)` or place `db/*.parquet` in the cache directory."
        )

