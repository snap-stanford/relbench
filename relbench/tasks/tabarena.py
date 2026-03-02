from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from relbench.base import EntityTask, Table, TaskType
from relbench.datasets.tabarena import TabArenaDataset

_SPLIT_TIMESTAMPS = {
    "train": pd.Timestamp("2000-01-01"),
    "val": pd.Timestamp("2000-01-02"),
    "test": pd.Timestamp("2000-01-03"),
}


def _binary_metric_error(true: np.ndarray, pred: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim > 1:
        pred = pred.reshape(pred.shape[0], -1)
        if pred.shape[1] == 1:
            pred = pred[:, 0]
        else:
            pred = pred[:, 1]
    if pred.min() < 0.0 or pred.max() > 1.0:
        pred = np.clip(pred, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-pred))
    score = roc_auc_score(np.asarray(true, dtype=np.int64), pred)
    return float(1.0 - score)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _multiclass_metric_error(true: np.ndarray, pred: np.ndarray) -> float:
    return _multiclass_metric_error_with_num_classes(true, pred, num_classes=None)


def _multiclass_metric_error_with_num_classes(
    true: np.ndarray,
    pred: np.ndarray,
    num_classes: Optional[int],
) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.int64)

    if num_classes is not None:
        inferred_num_classes = int(num_classes)
    elif pred.ndim == 2:
        inferred_num_classes = int(pred.shape[1])
    else:
        inferred_num_classes = int(
            max(true_arr.max(initial=0), pred.max(initial=0)) + 1
        )
    if inferred_num_classes <= 1:
        inferred_num_classes = 2

    if pred.ndim == 1:
        pred_labels = pred.astype(np.int64, copy=False)
        pred_labels = np.clip(pred_labels, 0, inferred_num_classes - 1)
        eps = 1e-7
        probs = np.full(
            (len(pred_labels), inferred_num_classes),
            fill_value=eps / max(inferred_num_classes - 1, 1),
            dtype=np.float64,
        )
        probs[np.arange(len(pred_labels), dtype=np.int64), pred_labels] = 1.0 - eps
    elif pred.ndim == 2:
        probs = pred
        if probs.shape[1] < inferred_num_classes:
            padding = np.full(
                (probs.shape[0], inferred_num_classes - probs.shape[1]),
                fill_value=0.0,
                dtype=np.float64,
            )
            probs = np.hstack([probs, padding])
        elif probs.shape[1] > inferred_num_classes:
            probs = probs[:, :inferred_num_classes]

        row_sums = probs.sum(axis=1)
        if np.all(probs >= 0.0) and np.allclose(row_sums, 1.0, atol=1e-4):
            pass
        else:
            probs = _softmax(probs)
    else:
        raise ValueError(
            "Expected multiclass predictions with shape (N,) or (N, num_classes). "
            f"Got shape {pred.shape}."
        )

    labels = np.arange(inferred_num_classes, dtype=np.int64)
    return float(
        sklearn_log_loss(
            true_arr,
            probs,
            labels=labels,
        )
    )


def _regression_metric_error(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


_binary_metric_error.__name__ = "metric_error"
_multiclass_metric_error.__name__ = "metric_error"
_regression_metric_error.__name__ = "metric_error"


class TabArenaFoldEntityTask(EntityTask):
    r"""Single-table TabArena task for a specific OpenML fold index."""

    entity_col = "record_id"
    entity_table = "records"
    time_col = "timestamp"
    target_col = "target"
    timedelta = pd.Timedelta(days=1)
    num_eval_timestamps = 1

    def __init__(
        self,
        dataset,
        *,
        fold: int,
        val_frac: float = 0.2,
        random_state: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        if not isinstance(dataset, TabArenaDataset):
            raise TypeError(
                "TabArenaFoldEntityTask expects a TabArenaDataset instance. "
                f"Got {type(dataset)}"
            )

        self.fold = int(fold)
        self.val_frac = float(val_frac)
        if not (0.0 < self.val_frac < 1.0):
            raise ValueError(f"val_frac must be in (0, 1), got {self.val_frac}")
        self.random_state = self.fold if random_state is None else int(random_state)

        if self.fold not in dataset.available_folds:
            raise ValueError(
                f"Fold={self.fold} is unavailable for {dataset.name}. "
                f"Available folds: {dataset.available_folds}"
            )

        self.problem_type = dataset.problem_type
        if self.problem_type == "regression":
            self.task_type = TaskType.REGRESSION
            self.metrics = [_regression_metric_error]
        elif self.problem_type == "binary":
            self.task_type = TaskType.BINARY_CLASSIFICATION
            self.metrics = [_binary_metric_error]
        elif self.problem_type == "multiclass":
            self.task_type = TaskType.MULTICLASS_CLASSIFICATION
            self.num_classes = int(dataset.num_classes)
            self.metrics = [
                _make_multiclass_metric_error_with_num_classes(self.num_classes)
            ]
        else:
            raise ValueError(f"Unsupported problem_type={self.problem_type}")

        super().__init__(dataset, cache_dir=cache_dir)

    def make_table(self, db, timestamps):  # pragma: no cover
        raise RuntimeError(
            "TabArenaFoldEntityTask uses precomputed OpenML fold indices and overrides _get_table()."
        )

    @lru_cache(maxsize=None)
    def _split_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_idx, test_idx = self.dataset.get_openml_fold_indices(self.fold)
        y = self.dataset.get_target_array()

        stratify = (
            y[train_idx] if self.problem_type in {"binary", "multiclass"} else None
        )
        try:
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=self.val_frac,
                random_state=self.random_state,
                shuffle=True,
                stratify=stratify,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=self.val_frac,
                random_state=self.random_state,
                shuffle=True,
                stratify=None,
            )

        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )

    def _get_table(self, split: str) -> Table:
        if split not in _SPLIT_TIMESTAMPS:
            raise ValueError(
                f"Unknown split={split!r}. Expected one of {sorted(_SPLIT_TIMESTAMPS.keys())}."
            )

        train_idx, val_idx, test_idx = self._split_indices()
        if split == "train":
            idx = train_idx
        elif split == "val":
            idx = val_idx
        else:
            idx = test_idx

        y = self.dataset.get_target_array()
        target = y[idx]

        df = pd.DataFrame(
            {
                self.time_col: _SPLIT_TIMESTAMPS[split],
                self.entity_col: idx.astype(np.int64, copy=False),
                self.target_col: target,
            }
        )

        if self.task_type != TaskType.REGRESSION:
            df[self.target_col] = df[self.target_col].astype(np.int64, copy=False)

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


def _make_multiclass_metric_error_with_num_classes(
    num_classes: int,
):
    def _metric(true: np.ndarray, pred: np.ndarray) -> float:
        return _multiclass_metric_error_with_num_classes(
            true,
            pred,
            num_classes=num_classes,
        )

    _metric.__name__ = "metric_error"
    return _metric
