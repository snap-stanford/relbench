import torch
import torch.nn.functional as F
from torch_geometric.loader import NodeLoader
from torch_geometric.nn import MLP
from torch_geometric.sampler import NeighborSampler

from rtb.datasets import FakeProductDataset
from rtb.external.graph import get_train_table_input, make_pkey_fkey_graph
from rtb.external.nn import HeteroEncoder, HeteroGraphSAGE


def test_train_fake_product_dataset(tmp_path):
    dataset = FakeProductDataset(root=tmp_path, process=True)

    data = make_pkey_fkey_graph(
        dataset.db,
        dataset.get_stype_proposal(),
    )

    col_names_dict = {  # TODO Expose as method in `HeteroData`.
        node_type: data[node_type].tf.col_names_dict for node_type in data.node_types
    }

    # Ensure that full-batch model works as expected ##########################

    encoder = HeteroEncoder(64, col_names_dict, data.col_stats_dict)
    gnn = HeteroGraphSAGE(data.node_types, data.edge_types, 64)
    head = MLP(64, out_channels=1, num_layers=1)

    x_dict = encoder(data.tf_dict)
    x_dict = gnn(x_dict, data.edge_index_dict)
    x = head(x_dict["customer"])

    assert len(x_dict) == 3
    assert x_dict["customer"].size() == (100, 64)
    assert x_dict["review"].size() == (500, 64)
    assert x_dict["product"].size() == (30, 64)
    assert x.size() == (100, 1)

    # Ensure that neighbor sampling works on train/val/test splits ############

    sampler = NeighborSampler(
        data,
        num_neighbors=[-1, -1],
        time_attr="time",
    )

    train_table = dataset.make_train_table("churn")
    val_table = dataset.make_val_table("churn")
    test_table = dataset.make_test_table("churn")

    for i, table in enumerate([train_table, val_table, test_table]):
        train_table_input = get_train_table_input(table, target_col="churn")

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

    # Ensure that mini-batch training works ###################################

    train_table_input = get_train_table_input(
        train_table=train_table,
        target_col="churn",
        target_dtype=torch.float,  # Binary classification.
    )

    train_loader = NodeLoader(
        data,
        node_sampler=sampler,
        input_nodes=train_table_input.nodes,
        input_time=train_table_input.time,
        transform=train_table_input.transform,
        batch_size=32,
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(gnn.parameters()) + list(head.parameters()),
        lr=0.01,
    )

    for batch in train_loader:
        optimizer.zero_grad()

        x_dict = encoder(batch.tf_dict)
        x_dict = gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        x = head(x_dict["customer"]).squeeze(-1)

        loss = F.binary_cross_entropy_with_logits(x, batch["customer"].y)
        loss.backward()

        optimizer.step()
