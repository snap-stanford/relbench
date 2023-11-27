from relbench.datasets import FakeDataset


def test_fake_reviews_dataset():
    dataset = FakeDataset()
    assert str(dataset) == "FakeDataset()"
    assert dataset.task_names == ["rel-amazon-churn", "rel-amazon-ltv"]

    task = dataset.get_task("rel-amazon-churn", process=True)
    assert str(task) == "ChurnTask(dataset=FakeDataset())"

    train_table = task.train_table
    val_table = task.val_table
    for table in [train_table, val_table]:
        assert set(table.df.columns) >= {
            "timestamp",
            "customer_id",
            "churn",
        }

    test_table = task.test_table
    assert set(test_table.df.columns) == {
        "timestamp",
        "customer_id",
    }
