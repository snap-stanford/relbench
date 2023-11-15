from rtb.datasets import FakeEcommerceDataset
from rtb.utils import make_pkey_fkey_graph
import tempfile


def test_make_pkey_fkey_graph():
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = FakeEcommerceDataset(root=temp_dir)
    hetero_data, col_stats_dict = make_pkey_fkey_graph(
        dataset.db_train, dataset.col_to_stype_dict
    )
