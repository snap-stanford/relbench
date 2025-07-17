import pandas as pd

from relbench.base import AutoCompleteTask, TaskType
from relbench.datasets.fake import FakeDataset


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
