import torch
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler import NeighborSampler

from rtb.datasets import FakeProductDataset
from rtb.external.graph import get_train_table_input, make_pkey_fkey_graph
from rtb.external.nn import GraphSAGE, HeteroEncoder


def test_train_fake_product_dataset(tmp_path):
    dataset = FakeProductDataset(root=tmp_path, process=True)

    data = make_pkey_fkey_graph(
        dataset.db,
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
    assert x_dict["review"].size() == (500, 64)
    assert x_dict["product"].size() == (30, 64)

    sampler = NeighborSampler(
        data,
        num_neighbors=[-1, -1],
        time_attr="time",
    )

    train_table = dataset.make_train_table("churn")
    val_table = dataset.make_val_table("churn")
    test_table = dataset.make_test_table("churn")

    for i, table in enumerate([train_table, val_table, test_table]):
        train_table_input = get_train_table_input(
            train_table=table,
            target_col="churn",
            target_dtype=torch.float,  # Binary classification.
        )

        loader = NodeLoader(
            data,
            node_sampler=sampler,
            input_nodes=train_table_input.nodes,
            input_time=train_table_input.time,
            transform=train_table_input.transform,
            batch_size=32,
        )

        batch = next(iter(loader))
        assert batch["customer"].batch_size == 32
        assert batch["customer"].seed_time.size() == (32,)
        if i < 2:
            assert batch["customer"].y.size() == (32,)
