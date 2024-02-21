from typing import Dict, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MIPSKNNIndex
from torch_geometric.typing import NodeType

from relbench.data import LinkTask
from relbench.data.task_base import TaskType
from relbench.datasets import FakeDataset
from relbench.external.graph import (
    get_link_train_table_input,
    get_stype_proposal,
    make_pkey_fkey_graph,
)
from relbench.external.loader import LinkNeighborLoader
from relbench.external.nn import HeteroEncoder, HeteroGraphSAGE


@pytest.mark.parametrize(
    "share_same_time",
    [True, False],
)
def test_link_train_fake_product_dataset(tmp_path, share_same_time):
    dataset = FakeDataset()

    data = make_pkey_fkey_graph(
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
    encoder = HeteroEncoder(64, node_to_col_names_dict, data.col_stats_dict)
    gnn = HeteroGraphSAGE(data.node_types, data.edge_types, 64)

    # Ensure that neighbor loading works on train/val/test splits ############
    task: LinkTask = dataset.get_task("rel-amazon-rec", process=True)
    assert task.task_type == TaskType.LINK_PREDICTION

    train_table_input = get_link_train_table_input(task.train_table, task)
    batch_size = 16
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[-1, -1],
        time_attr="time",
        src_nodes=train_table_input.src_nodes,
        src_to_dst_nodes=train_table_input.src_to_dst_nodes,
        num_dst_nodes=train_table_input.num_dst_nodes,
        src_time=train_table_input.src_time,
        share_same_time=share_same_time,
        batch_size=batch_size,
        # if share_same_time is True, we use sampler, so shuffle must be set False
        shuffle=not share_same_time,
        drop_last=not share_same_time,
    )

    batch = next(iter(train_loader))
    src_batch, batch_pos_dst, batch_neg_dst = batch
    src_seed_time = src_batch[task.src_entity_table].seed_time
    pos_dst_seed_time = batch_pos_dst[task.dst_entity_table].seed_time
    neg_dst_seed_time = batch_neg_dst[task.dst_entity_table].seed_time
    assert len(src_seed_time) == batch_size
    assert len(pos_dst_seed_time) == batch_size
    assert len(neg_dst_seed_time) == batch_size
    if share_same_time:
        shared_time = src_seed_time[0]
        assert (shared_time == src_seed_time).all()
        assert (shared_time == pos_dst_seed_time).all()
        assert (shared_time == neg_dst_seed_time).all()

    eval_loaders_dict: Dict[str, Tuple[NeighborLoader, NeighborLoader]] = {}
    for split in ["val", "test"]:
        seed_time = task.val_seed_time if split == "val" else task.test_seed_time
        target_table = task.val_table if split == "val" else task.test_table
        src_node_indices = torch.from_numpy(target_table.df[task.src_entity_col].values)
        src_loader = NeighborLoader(
            data,
            num_neighbors=[-1, -1],
            time_attr="time",
            input_nodes=(task.src_entity_table, src_node_indices),
            input_time=torch.full(
                size=(len(src_node_indices),), fill_value=seed_time, dtype=torch.long
            ),
            batch_size=32,
            shuffle=False,
        )
        dst_loader = NeighborLoader(
            data,
            num_neighbors=[-1, -1],
            time_attr="time",
            input_nodes=(task.dst_entity_table, torch.arange(task.num_dst_nodes)),
            input_time=torch.full(
                size=(task.num_dst_nodes,), fill_value=seed_time, dtype=torch.long
            ),
            batch_size=32,
            shuffle=False,
        )
        eval_loaders_dict[split] = (src_loader, dst_loader)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(gnn.parameters()),
        lr=0.01,
    )

    def model_forward(batch: HeteroData, node_type: NodeType):
        x_dict = encoder(batch.tf_dict)
        x_dict = gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        batch_size = batch[node_type].batch_size
        return x_dict[node_type][:batch_size]

    # Training
    encoder.train()
    gnn.train()
    for batch in train_loader:
        src_batch, batch_pos_dst, batch_neg_dst = batch
        # [batch_size, emb_dim]
        x_src = model_forward(src_batch, task.src_entity_table)
        x_pos_dst = model_forward(batch_pos_dst, task.dst_entity_table)
        x_neg_dst = model_forward(batch_neg_dst, task.dst_entity_table)

        # [batch_size, ]
        pos_score = torch.sum(x_src * x_pos_dst, dim=1)
        if share_same_time:
            # [batch_size, batch_size]
            neg_score = x_src @ x_neg_dst.t()
        else:
            # [batch_size, ]
            neg_score = torch.sum(x_src * x_neg_dst, dim=1)
        diff_score = pos_score - neg_score
        # BPR loss
        optimizer.zero_grad()
        loss = F.softplus(-diff_score).mean()
        loss.backward()
        optimizer.step()

    # Validation
    encoder.eval()
    gnn.eval()

    for split in ["val", "test"]:
        dst_embs = []
        src_loader, dst_loader = eval_loaders_dict[split]
        for batch in dst_loader:
            emb = model_forward(batch, task.dst_entity_table)
            dst_embs.append(emb)
        dst_emb = torch.cat(dst_embs, dim=0)
        del dst_embs

        mips = MIPSKNNIndex(dst_emb)

        pred_index_mat_list = []
        for batch in src_loader:
            emb = model_forward(batch, task.src_entity_table)
            _, pred_index_mat = mips.search(emb, k=task.eval_k)
            pred_index_mat_list.append(pred_index_mat)
        pred = torch.cat(pred_index_mat_list, dim=0).numpy()

        if split == "val":
            task.evaluate(pred, task.val_table)
        else:
            task.evaluate(pred)
