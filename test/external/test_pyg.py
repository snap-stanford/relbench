from rtb.datasets import FakeEcommerceDataset
from rtb.external.pyg import make_pkey_fkey_graph
from torch_frame import TensorFrame


def test_make_pkey_fkey_graph(tmp_path):
    dataset = FakeEcommerceDataset(root=tmp_path)

    data = make_pkey_fkey_graph(
        dataset.db_train,
        dataset.get_stype_proposal(),
    )
    assert set(data.node_types) == {"customer", "transaction", "product"}

    data.validate()

    assert data["customer"].num_nodes == 100
    assert isinstance(data["customer"].tf, TensorFrame)

    assert data["transaction"].num_nodes <= 500
    assert isinstance(data["transaction"].tf, TensorFrame)

    assert data["product"].num_nodes == 30
    assert isinstance(data["product"].tf, TensorFrame)

    assert len(data.edge_types) == 4
    for edge_type in data.edge_types:
        src, _, dst = edge_type

        edge_index = data[edge_type].edge_index
        assert edge_index.size(0) == 2
        assert edge_index.size(1) <= data["transaction"].num_nodes
        assert edge_index[0].max() <= data[src].num_nodes
        assert edge_index[1].max() <= data[dst].num_nodes
