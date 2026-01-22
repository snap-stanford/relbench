from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from relbench.base import Database, RecommendationTask, Table, TaskType
from relbench.metrics import (
    link_prediction_map,
    link_prediction_ndcg,
    link_prediction_precision,
    link_prediction_recall,
)


def _unique_list(values: pd.Series) -> list[int]:
    # Preserve deterministic ordering per group as first-appearance order.
    # Pandas' `unique()` preserves order.
    return [int(x) for x in values.unique().tolist()]


@dataclass(frozen=True)
class TGBAggregatedEventsSpec:
    src_entity_table: str
    dst_entity_table: str
    event_tables: tuple[str, ...] = ()
    event_time_col: str = "event_ts"
    src_col: str = "src_id"
    dst_col: str = "dst_id"


class TGBSrcDstNextTask(RecommendationTask):
    r"""Recommendation task for TGB-style edge streams.

    For each anchor timestamp `t`, predict the set of destination entities that
    connect to a source entity within the horizon `(t, t + timedelta]`.

    This task is intentionally generic to support:
    - homogeneous `tgbl-*` exports: `nodes` + `events`
    - bipartite `tgbl-wiki*` exports: `src_nodes`/`dst_nodes` + `events`
    - heterogeneous `thgl-*` exports: multiple `events_edge_type_*` tables
      aggregated by source/destination node types
    """

    task_type = TaskType.LINK_PREDICTION
    metrics = [
        link_prediction_recall,
        link_prediction_precision,
        link_prediction_map,
        link_prediction_ndcg,
    ]

    # Convention: keep these stable across all TGB tasks.
    src_entity_col = "src_id"
    dst_entity_col = "dst_id"
    time_col = "timestamp"

    def __init__(
        self,
        dataset,
        *,
        spec: TGBAggregatedEventsSpec,
        timedelta: pd.Timedelta,
        eval_k: int = 10,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.spec = spec
        self.timedelta = pd.Timedelta(timedelta)
        self.eval_k = int(eval_k)
        self.src_entity_table = spec.src_entity_table
        self.dst_entity_table = spec.dst_entity_table
        super().__init__(dataset, cache_dir=cache_dir)

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        event_tables = (
            self.spec.event_tables
            if self.spec.event_tables
            else tuple(name for name in db.table_dict.keys() if name.startswith("events"))
        )
        rows: list[pd.DataFrame] = []
        for t in timestamps:
            start = pd.Timestamp(t)
            end = start + self.timedelta

            window_frames: list[pd.DataFrame] = []
            for table_name in event_tables:
                table = db.table_dict[table_name]
                # For heterogeneous exports, keep only event tables whose FK schema
                # matches the intended src/dst entity tables.
                if table.fkey_col_to_pkey_table.get(self.spec.src_col) != self.src_entity_table:
                    continue
                if table.fkey_col_to_pkey_table.get(self.spec.dst_col) != self.dst_entity_table:
                    continue

                ev = table.df
                mask = (ev[self.spec.event_time_col] > start) & (ev[self.spec.event_time_col] <= end)
                if mask.any():
                    window_frames.append(ev.loc[mask, [self.spec.src_col, self.spec.dst_col]])

            if not window_frames:
                continue

            win = pd.concat(window_frames, axis=0, ignore_index=True)
            grouped = (
                win.groupby(self.spec.src_col, sort=False)[self.spec.dst_col]
                .agg(_unique_list)
                .reset_index()
                .rename(columns={self.spec.src_col: self.src_entity_col, self.spec.dst_col: self.dst_entity_col})
            )
            grouped[self.time_col] = start
            rows.append(grouped[[self.time_col, self.src_entity_col, self.dst_entity_col]])

        if rows:
            out = pd.concat(rows, axis=0, ignore_index=True)
        else:
            out = pd.DataFrame({self.time_col: [], self.src_entity_col: [], self.dst_entity_col: []})

        return Table(
            df=out,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


class TGBNodeLabelNextTask(RecommendationTask):
    r"""Node property prediction as a recommendation task.

    For each anchor timestamp `t`, predict the set of labels associated with a
    node within the horizon `(t, t + timedelta]`.

    This task assumes the standard `tgbn-*` export schema:
    - `nodes(node_id)`
    - `labels(label_id)`
    - `label_events(label_event_id, src_id, label_ts)`
    - `label_event_items(item_id, label_event_id, label_id, label_weight)`

    Note: This task treats labels as unweighted positives for evaluation
    (binary relevance at top-k). If graded relevance is desired, a custom task
    evaluator would be required.
    """

    task_type = TaskType.LINK_PREDICTION
    metrics = [
        link_prediction_recall,
        link_prediction_precision,
        link_prediction_map,
        link_prediction_ndcg,
    ]

    src_entity_col = "src_id"
    dst_entity_col = "label_id"
    time_col = "timestamp"

    def __init__(
        self,
        dataset,
        *,
        timedelta: pd.Timedelta,
        eval_k: int = 10,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.timedelta = pd.Timedelta(timedelta)
        self.eval_k = int(eval_k)
        self.src_entity_table = "nodes"
        self.dst_entity_table = "labels"
        super().__init__(dataset, cache_dir=cache_dir)

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        le = db.table_dict["label_events"].df
        items = db.table_dict["label_event_items"].df

        rows: list[pd.DataFrame] = []
        for t in timestamps:
            start = pd.Timestamp(t)
            end = start + self.timedelta
            mask = (le["label_ts"] > start) & (le["label_ts"] <= end)
            if not mask.any():
                continue

            le_win = le.loc[mask, ["label_event_id", "src_id"]]
            joined = le_win.merge(items[["label_event_id", "label_id"]], on="label_event_id", how="inner")
            grouped = (
                joined.groupby("src_id", sort=False)["label_id"]
                .agg(_unique_list)
                .reset_index()
                .rename(columns={"label_id": self.dst_entity_col, "src_id": self.src_entity_col})
            )
            grouped[self.time_col] = start
            rows.append(grouped[[self.time_col, self.src_entity_col, self.dst_entity_col]])

        if rows:
            out = pd.concat(rows, axis=0, ignore_index=True)
        else:
            out = pd.DataFrame({self.time_col: [], self.src_entity_col: [], self.dst_entity_col: []})

        return Table(
            df=out,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )
