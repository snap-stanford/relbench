from rtb.datasets import FakeProductDataset
from rtb.external.graph import make_pkey_fkey_graph
from torch_frame import TensorFrame

from torch_frame.config import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder


def test_make_pkey_fkey_graph(tmp_path):
    dataset = FakeProductDataset(root=tmp_path, process=True)

    data = make_pkey_fkey_graph(dataset.db_train,
                                dataset.get_stype_proposal(),
                                text_embedder_cfg=TextEmbedderConfig(
                                    HashTextEmbedder(16),
                                    batch_size=None,
                                ))
    assert set(data.node_types) == {"customer", "review", "product"}

    data.validate()

    assert data["customer"].num_nodes == 100
    assert isinstance(data["customer"].tf, TensorFrame)

    assert data["review"].num_nodes <= 500
    assert isinstance(data["review"].tf, TensorFrame)

    assert data["product"].num_nodes == 30
    assert isinstance(data["product"].tf, TensorFrame)

    assert len(data.edge_types) == 4
    for edge_type in data.edge_types:
        src, _, dst = edge_type

        edge_index = data[edge_type].edge_index
        assert edge_index.size(0) == 2
        assert edge_index.size(1) <= data["review"].num_nodes
        assert edge_index[0].max() <= data[src].num_nodes
        assert edge_index[1].max() <= data[dst].num_nodes
