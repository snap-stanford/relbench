from typing import Dict

import pytest
import torch
import torch.nn.functional as F
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP

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

    # Ensure that neighbor loading works on train/val/test splits ############
    task = dataset.get_task("rel-amazon-rec", process=True)
    assert task.task_type == TaskType.LINK_PREDICTION

    train_table_input = get_link_train_table_input(task.train_table, task)
    batch_size = 16
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[-1, -1],
        time_attr="time",
        src_nodes=train_table_input.src_nodes,
        dst_nodes_list=train_table_input.dst_nodes_list,
        num_dst_nodes=train_table_input.num_dst_nodes,
        src_time=train_table_input.src_time,
        share_same_time=share_same_time,
        batch_size=batch_size,
        # if share_same_time is True, we use sampler, so shuffle must be set False
        shuffle=not share_same_time,
    )

    for batch in train_loader:
        src_batch, pos_dst_batch, neg_dst_batch = batch
        src_seed_time = src_batch[task.src_entity_table].seed_time
        pos_dst_seed_time = pos_dst_batch[task.dst_entity_table].seed_time
        neg_dst_seed_time = neg_dst_batch[task.dst_entity_table].seed_time
        assert len(src_seed_time) <= batch_size
        assert len(pos_dst_seed_time) <= batch_size
        assert len(neg_dst_seed_time) <= batch_size
        if share_same_time:
            shared_time = src_seed_time[0]
            assert (shared_time == src_seed_time).all()
            assert (shared_time == pos_dst_seed_time).all()
            assert (shared_time == neg_dst_seed_time).all()

    eval_loader_dict: Dict[str, NeighborLoader] = {}
    for split, table in [
        ("val", task.val_table),
        ("test", task.test_table),
    ]:
        table_input = get_link_train_table_input(table, task)
