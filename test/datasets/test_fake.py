from rtb.datasets import FakeReviewsDataset


def test_fake_reviews_dataset():
    dataset = FakeReviewsDataset()
    assert str(dataset) == "FakeReviewsDataset()"
    assert dataset.task_names == ["customer_churn", "customer_ltv"]

    task = dataset.get_task("customer_churn")
    assert str(task) == "CustomerChurnTask(dataset=FakeReviewsDataset())"

    train_table = task.make_default_train_table()
    val_table = task.make_default_val_table()
    for table in [train_table, val_table]:
        assert set(table.df.columns) >= {
            "timestamp",
            "customer_id",
            "churn",
        }

    input_test_table = task.make_input_test_table()
    assert set(input_test_table.df.columns) == {
        "timestamp",
        "customer_id",
    }

    task = dataset.get_task("customer_ltv")
    assert str(task) == "CustomerLTVTask(dataset=FakeReviewsDataset())"

    train_table = task.make_default_train_table()
    val_table = task.make_default_val_table()
    for table in [train_table, val_table]:
        assert set(table.df.columns) >= {
            "timestamp",
            "customer_id",
            "ltv",
        }

    input_test_table = task.make_input_test_table()
    assert set(input_test_table.df.columns) == {
        "timestamp",
        "customer_id",
    }
