from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch_frame import TensorFrame
from torch_frame.config import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder

from relbench.data import LinkTask
from relbench.datasets import FakeDataset
from relbench.external.graph import (
    get_link_train_table_input,
    get_stype_proposal,
    make_pkey_fkey_graph,
)
from relbench.metrics import link_prediction_map


def test_make_pkey_fkey_graph():
    dataset = FakeDataset()

    data, _ = make_pkey_fkey_graph(
        dataset.db,
        get_stype_proposal(dataset.db),
        text_embedder_cfg=TextEmbedderConfig(
            HashTextEmbedder(16),
            batch_size=None,
        ),
    )
    assert set(data.node_types) == {"customer", "review", "product", "relations"}

    data.validate()

    assert data["customer"].num_nodes == 100
    assert isinstance(data["customer"].tf, TensorFrame)

    assert data["review"].num_nodes <= 600
    assert isinstance(data["review"].tf, TensorFrame)

    assert data["product"].num_nodes == 30
    assert isinstance(data["product"].tf, TensorFrame)

    assert isinstance(data["relations"].tf, TensorFrame)

    assert len(data.edge_types) == 8
    for edge_type in data.edge_types:
        src, _, dst = edge_type

        edge_index = data[edge_type].edge_index
        assert edge_index.size(0) == 2
        assert edge_index.size(1) <= data["review"].num_nodes
        assert edge_index[0].max() <= data[src].num_nodes
        assert edge_index[1].max() <= data[dst].num_nodes
