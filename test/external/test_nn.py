from rtb.datasets import FakeEcommerceDataset
from rtb.external.graph import make_pkey_fkey_graph
from rtb.external.nn import GraphSAGE, HeteroEncoder


def test_train_fake_ecommerce_dataset(tmp_path):
    dataset = FakeEcommerceDataset(root=tmp_path)

    data = make_pkey_fkey_graph(
        dataset.db_train,
        dataset.get_stype_proposal(),
    )
    col_names_dict = {  # TODO Expose as method in `HeteroData`.
        node_type: data[node_type].tf.col_names_dict for node_type in data.node_types
    }

    encoder = HeteroEncoder(64, col_names_dict, data.col_stats_dict)
    gnn = GraphSAGE(data.node_types, data.edge_types, 64)

    x_dict = encoder(data.tf_dict)
    x_dict = gnn(x_dict, data.edge_index_dict)

    assert len(x_dict) == 3
    assert x_dict["customer"].size() == (100, 64)
    assert x_dict["transaction"].size() == (400, 64)
    assert x_dict["product"].size() == (30, 64)
