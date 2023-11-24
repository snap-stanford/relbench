from rtb.datasets import FakeReviewsDataset


def test_fake_reviews_dataset():
    dataset = FakeReviewsDataset()
    assert str(dataset) == "FakeProductDataset()"
    assert dataset.task_names == ["customer_churn", "customer_ltv"]

    train_table = dataset.make_train_table("ltv")
    val_table = dataset.make_val_table("ltv")
    test_table = dataset.make_test_table("ltv")
    for table in [train_table, val_table]:
        assert set(table.df.columns) > {
            "window_min_time",
            "window_max_time",
            "customer_id",
            "ltv",
        }
    assert set(test_table.df.columns) == {
        "window_min_time",
        "window_max_time",
        "customer_id",
    }
