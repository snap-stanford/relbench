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
    get_link_train_table_input,
    get_stype_proposal,
    make_pkey_fkey_graph,
)
from relbench.external.nn import HeteroEncoder, HeteroGraphSAGE


def test_link_train_fake_product_dataset(tmp_path):
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

    loader_dict: Dict[str, NeighborLoader] = {}
    for split, table in [
        ("train", task.train_table),
        ("val", task.val_table),
        ("test", task.test_table),
    ]:
        table_input = get_link_train_table_input(table, task)
