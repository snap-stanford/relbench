from rtb.datasets import FakeProductDataset
from rtb.external.graph import make_pkey_fkey_graph
from rtb.external.nn import GraphSAGE, HeteroEncoder
from torch_frame.config import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder


def test_train_fake_product_dataset(tmp_path):
    dataset = FakeProductDataset(root=tmp_path, process=True)

    data = make_pkey_fkey_graph(dataset.db_train,
                                dataset.get_stype_proposal(),
                                text_embedder_cfg=TextEmbedderConfig(
                                    text_embedder=HashTextEmbedder(8),
                                    batch_size=None))
    node_to_col_names_dict = {  # TODO Expose as method in `HeteroData`.
        node_type: data[node_type].tf.col_names_dict
        for node_type in data.node_types
    }

    encoder = HeteroEncoder(64, node_to_col_names_dict, data.col_stats_dict)
    gnn = GraphSAGE(data.node_types, data.edge_types, 64)

    x_dict = encoder(data.tf_dict)
    x_dict = gnn(x_dict, data.edge_index_dict)

    assert len(x_dict) == 3
    assert x_dict["customer"].size() == (100, 64)
    assert x_dict["review"].size() == (400, 64)
    assert x_dict["product"].size() == (30, 64)
