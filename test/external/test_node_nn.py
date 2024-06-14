from typing import Dict

import torch
import torch.nn.functional as F
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP

from relbench.data.task_base import TaskType
from relbench.datasets import FakeDataset
from relbench.external.graph import (
    get_node_train_table_input,
    get_stype_proposal,
    make_pkey_fkey_graph,
)
from relbench.external.nn import HeteroEncoder, HeteroGraphSAGE


def test_node_train_fake_product_dataset(tmp_path):
    dataset = FakeDataset()

    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.db,
        get_stype_proposal(dataset.db),
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=HashTextEmbedder(8), batch_size=None
        ),
        cache_dir=tmp_path,
    )
    node_to_col_names_dict = {  # TODO Expose as method in `HeteroData`.
        node_type: data[node_type].tf.col_names_dict for node_type in data.node_types
    }

    # Ensure that full-batch model works as expected ##########################

    encoder = HeteroEncoder(64, node_to_col_names_dict, col_stats_dict)
    gnn = HeteroGraphSAGE(data.node_types, data.edge_types, 64)
    head = MLP(64, out_channels=1, num_layers=1)

    x_dict = encoder(data.tf_dict)
    x_dict = gnn(x_dict, data.edge_index_dict)
    x = head(x_dict["customer"])

    assert len(x_dict) == 4
    assert x_dict["customer"].size() == (100, 64)
    assert x_dict["review"].size() == (540, 64)
    assert x_dict["product"].size() == (30, 64)
    assert x.size() == (100, 1)

    # Ensure that neighbor loading works on train/val/test splits ############
    task = dataset.get_task("user-churn", process=True)
    assert task.task_type == TaskType.BINARY_CLASSIFICATION

    stats = task.stats()
    assert len(stats) == 3
    assert len(stats["train"]) == 11
    assert len(next(iter(stats["train"].values()))) == 4
    assert len(stats["val"]) == 2
    assert len(next(iter(stats["val"].values()))) == 4
    assert len(stats["total"].values()) == 2

    loader_dict: Dict[str, NeighborLoader] = {}
    for split, table in [
        ("train", task.train_table),
        ("val", task.val_table),
        ("test", task.test_table),
    ]:
        table_input = get_node_train_table_input(table=table, task=task)
        loader = NeighborLoader(
            data,
            num_neighbors=[-1, -1],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=32,
            # Only shuffle during training.
            shuffle=split == "train",
        )
        loader_dict[split] = loader

        batch = next(iter(loader))
        batch_size = batch["customer"].batch_size
        assert batch_size <= 32
        assert batch["customer"].seed_time.size() == (batch_size,)
        if split != "test":
            assert batch["customer"].y.size() == (batch_size,)

    # Ensure that mini-batch training works ###################################

    train_table_input = get_node_train_table_input(task.train_table, task=task)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(gnn.parameters()) + list(head.parameters()),
        lr=0.01,
    )
    entity_table = train_table_input.nodes[0]

    # Training
    encoder.train()
    gnn.train()
    head.train()
    for batch in loader_dict["train"]:
        x_dict = encoder(batch.tf_dict)
        x_dict = gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        pred = head(x_dict[entity_table]).squeeze(-1)

        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(pred, batch[entity_table].y)
        loss.backward()

        optimizer.step()

    # Validation
    encoder.eval()
    gnn.eval()
    head.eval()

    for split in ["val", "test"]:
        pred_list = []
        target_list = []
        for batch in loader_dict[split]:
            with torch.no_grad():
                x_dict = encoder(batch.tf_dict)
                x_dict = gnn(
                    x_dict,
                    batch.edge_index_dict,
                    batch.num_sampled_nodes_dict,
                    batch.num_sampled_edges_dict,
                )
                pred = head(x_dict[entity_table]).squeeze(-1).sigmoid()

            pred_list.append(pred.detach().cpu())
            if split == "val":
                target_list.append(batch[entity_table].y.cpu())
        if split == "val":
            target = torch.cat(target_list)
            assert torch.allclose(
                target,
                torch.from_numpy(task.val_table.df[task.target_col].values).to(
                    target.dtype
                ),
            )
        pred = torch.cat(pred_list, dim=0).numpy()
        if split == "val":
            task.evaluate(pred, task.val_table)
        else:
            task.evaluate(pred)


def test_node_train_empty_graph(tmp_path):
    # Make a very sparse graph
    num_customers = 50
    dataset = FakeDataset(num_customers=num_customers, num_reviews=1)

    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.db,
        get_stype_proposal(dataset.db),
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=HashTextEmbedder(8), batch_size=None
        ),
        cache_dir=tmp_path,
    )
    node_to_col_names_dict = {  # TODO Expose as method in `HeteroData`.
        node_type: data[node_type].tf.col_names_dict for node_type in data.node_types
    }
    loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        time_attr="time",
        input_nodes=("customer", torch.arange(num_customers)),
        input_time=torch.zeros(num_customers, dtype=torch.long),
        batch_size=5,
        # Only shuffle during training.
        shuffle=True,
    )
    encoder = HeteroEncoder(64, node_to_col_names_dict, col_stats_dict)
    gnn = HeteroGraphSAGE(data.node_types, data.edge_types, 64)
    head = MLP(64, out_channels=1, num_layers=1)

    for batch in loader:
        x_dict = encoder(batch.tf_dict)
        x_dict = gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        head(x_dict["customer"])
