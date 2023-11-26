from rtb.datasets import FakeReviewsDataset


def test_fake_reviews_dataset():
    dataset = FakeReviewsDataset()
    assert str(dataset) == "FakeReviewsDataset()"
    assert dataset.task_names == ["customer_churn", "customer_ltv"]

    task = dataset.get_task("customer_churn", process=True)
    assert str(task) == "CustomerChurnTask(dataset=FakeReviewsDataset())"

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
