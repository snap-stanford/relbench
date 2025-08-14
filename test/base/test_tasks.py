import pandas as pd

from relbench.base import AutoCompleteTask, TaskType
from relbench.datasets import get_dataset
from relbench.datasets.fake import FakeDataset
from relbench.tasks import get_task


def test_autocomplete_task():
    dataset = FakeDataset()
    assert dataset is not None
    task = AutoCompleteTask(
        dataset=dataset,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        entity_table="review",
        target_col="rating",
        cache_dir=None,
        remove_columns=[],
    )
    db = dataset.get_db(upto_test_timestamp=False)
    pks = db.table_dict["review"].df.get("primary_key")
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")
    assert test_table.df.primary_key.max() == pks.max()
    task_table_full = pd.concat(
        [train_table.df, val_table.df, test_table.df], ignore_index=True
    )
    assert task_table_full.primary_key.isin(pks).all()

    # ensure get_task can be called multiple times on the same database
    dataset = get_dataset("rel-f1")
    db = dataset.get_db()
    results_columns = db.table_dict["results"].df.columns.tolist()

    task = get_task("rel-f1", "results-position")
    table = task.get_table("train")

    # ensure columns are removed correctly
    for table, column in task.dataset.remove_columns:
        assert column not in task.dataset.get_db().table_dict[table].df.columns

    task = get_task("rel-f1", "qualifying-position")
    # ensure the results table contains all the correct columns
    assert (
        task.dataset.get_db().table_dict["results"].df.columns.tolist()
        == results_columns
    )
