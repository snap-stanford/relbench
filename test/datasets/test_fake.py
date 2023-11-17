from rtb.datasets import FakeProductDataset


def test_fake_product_dataset(tmp_path):
    dataset = FakeProductDataset(root=tmp_path, process=True)
    assert (
        str(dataset)
        == """FakeProductDataset(
  tables=['customer', 'product', 'review'],
  tasks=['churn', 'ltv'],
  min_time=1970-01-01 00:00:00,
  max_time=1983-08-31 00:00:00,
  train_max_time=1980-12-06 00:00:00,
  val_max_time=1982-04-19 00:00:00,
)"""
    )

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
