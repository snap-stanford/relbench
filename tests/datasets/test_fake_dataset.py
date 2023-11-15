from rtb.datasets import FakeEcommerceDataset
import tempfile


def test_fake_ecommerce_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = FakeEcommerceDataset(root=temp_dir)
    assert (
        str(dataset) == "FakeEcommerceDataset(\n"
        "  tables=['product', 'transaction', 'customer'],\n"
        "  tasks=['ltv'],\n"
        "  min_time=1970-01-01 00:00:00,\n"
        "  max_time=1983-08-31 00:00:00,\n"
        "  train_max_time=1980-12-06 00:00:00,\n"
        "  val_max_time=1982-04-19 00:00:00,\n"
        ")"
    )
    window_size = dataset.tasks["ltv"].test_time_window_sizes[0]
    train_table = dataset.make_train_table("ltv", window_size)
    val_table = dataset.make_val_table("ltv", window_size)
    test_table = dataset.make_test_table("ltv", window_size)
    for table in [train_table, val_table]:
        assert set(table.df.columns) == {
            "window_max_time",
            "ltv",
            "customer_id",
            "window_min_time",
        }
    assert set(test_table.df.columns) == {
        "window_max_time",
        "customer_id",
        "window_min_time",
    }
