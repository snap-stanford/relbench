import numpy as np
import pandas as pd

from relbench.datasets.tabarena import TabArenaDataset
from relbench.tasks.tabarena import TabArenaFoldEntityTask


class _FakeOpenMLDataset:
    def __init__(self, X_df: pd.DataFrame, y_ser: pd.Series, target_name: str):
        self._X_df = X_df
        self._y_ser = y_ser
        self._target_name = str(target_name)

    def get_data(self, *, target: str, dataset_format: str):
        assert target == self._target_name
        assert dataset_format == "dataframe"
        categorical_indicator = [False] * len(self._X_df.columns)
        attribute_names = list(self._X_df.columns)
        return self._X_df, self._y_ser, categorical_indicator, attribute_names


class _FakeOpenMLTask:
    def __init__(
        self,
        *,
        target_name: str,
        X_df: pd.DataFrame,
        y_ser: pd.Series,
        n_repeats: int,
        n_folds: int,
    ):
        self.target_name = str(target_name)
        self._dataset = _FakeOpenMLDataset(
            X_df=X_df, y_ser=y_ser, target_name=target_name
        )
        self._n_repeats = int(n_repeats)
        self._n_folds = int(n_folds)
        self._n_samples = int(len(X_df))

    def get_dataset(self):
        return self._dataset

    def get_split_dimensions(self):
        return self._n_repeats, self._n_folds, self._n_samples

    def get_train_test_split_indices(self, *, repeat: int, fold: int, sample: int):
        assert int(sample) == 0
        repeat = int(repeat)
        fold = int(fold)
        if repeat < 0 or repeat >= self._n_repeats:
            raise ValueError(
                f"repeat={repeat} out of range for n_repeats={self._n_repeats}"
            )
        if fold < 0 or fold >= self._n_folds:
            raise ValueError(f"fold={fold} out of range for n_folds={self._n_folds}")

        # Simple deterministic CV split: each fold takes indices i where i % n_folds == fold.
        idx = np.arange(self._n_samples, dtype=np.int64)
        test_idx = idx[idx % self._n_folds == fold]
        train_idx = idx[idx % self._n_folds != fold]
        return train_idx, test_idx


def _install_fake_openml(monkeypatch):
    def _fake_load_task_with_retry(
        self: TabArenaDataset, task_id: int, retries: int = 4
    ):
        _ = task_id
        _ = retries
        n_samples = 90
        n_folds = int(self.spec.fold_count)
        target_name = self.spec.target

        X_df = pd.DataFrame(
            {
                "feat_num": np.arange(n_samples, dtype=np.int64),
                "feat_mod3": np.arange(n_samples, dtype=np.int64) % 3,
            }
        )
        if self.problem_type == "regression":
            y_ser = pd.Series(np.linspace(0.0, 1.0, n_samples), name=target_name)
        elif self.problem_type == "binary":
            y_ser = pd.Series(["no", "yes"] * (n_samples // 2), name=target_name)
        else:
            classes = [f"class_{i}" for i in range(int(self.spec.num_classes))]
            y_ser = pd.Series(
                [classes[i % len(classes)] for i in range(n_samples)], name=target_name
            )

        return _FakeOpenMLTask(
            target_name=target_name,
            X_df=X_df,
            y_ser=y_ser,
            n_repeats=1,
            n_folds=n_folds,
        )

    monkeypatch.setattr(
        TabArenaDataset, "_load_task_with_retry", _fake_load_task_with_retry
    )


def test_tabarena_dataset_and_task_binary(monkeypatch):
    _install_fake_openml(monkeypatch)

    dataset = TabArenaDataset(dataset_slug="apsfailure", cache_dir=None)
    db = dataset.get_db()
    records = db.table_dict["records"]
    assert records.pkey_col == "record_id"
    assert records.time_col is None
    assert len(records) == 90

    train_idx, test_idx = dataset.get_openml_fold_indices(0)
    assert train_idx.dtype == np.int64
    assert test_idx.dtype == np.int64
    assert set(train_idx).isdisjoint(set(test_idx))

    task = TabArenaFoldEntityTask(dataset, fold=0, cache_dir=None)
    train_table = task.get_table("train")
    assert set(train_table.df.columns) == {"timestamp", "record_id", "target"}
    test_table = task.get_table("test")
    assert set(test_table.df.columns) == {"timestamp", "record_id"}

    # Perfect predictions yield AUC=1.0 => metric_error=0.0.
    full_test = task.get_table("test", mask_input_cols=False)
    y_true = full_test.df["target"].to_numpy()
    metrics = task.evaluate(y_true, target_table=full_test)
    assert metrics["metric_error"] == 0.0


def test_tabarena_dataset_and_task_regression(monkeypatch):
    _install_fake_openml(monkeypatch)

    dataset = TabArenaDataset(dataset_slug="diamonds", cache_dir=None)
    task = TabArenaFoldEntityTask(dataset, fold=0, cache_dir=None)

    full_test = task.get_table("test", mask_input_cols=False)
    y_true = full_test.df["target"].to_numpy()
    metrics = task.evaluate(y_true, target_table=full_test)
    assert metrics["metric_error"] == 0.0


def test_tabarena_task_multiclass(monkeypatch):
    _install_fake_openml(monkeypatch)

    dataset = TabArenaDataset(dataset_slug="splice", cache_dir=None)
    task = TabArenaFoldEntityTask(dataset, fold=0, cache_dir=None)

    full_test = task.get_table("test", mask_input_cols=False)
    y_true = full_test.df["target"].to_numpy()
    num_classes = int(dataset.num_classes)

    # Use a uniform distribution: log loss is finite and should be > 0.
    probs = np.full(
        (len(y_true), num_classes), fill_value=1.0 / num_classes, dtype=np.float64
    )
    metrics = task.evaluate(probs, target_table=full_test)
    assert metrics["metric_error"] > 0.0
