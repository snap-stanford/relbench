from rtb.datasets import FakeEcommerceDataset
from rtb.external.pyg import make_pkey_fkey_graph


def test_make_pkey_fkey_graph(tmp_path):
    dataset = FakeEcommerceDataset(root=tmp_path)

    hetero_data, col_stats_dict = make_pkey_fkey_graph(
        dataset.db_train,
        dataset.get_stype_proposal(),
    )
    # TODO Add some tests.
