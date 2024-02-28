import random

import torch
from torch import Tensor

from relbench.external.loader import CustomLinkDataset


def get_lp_setup(
    num_src_nodes, num_dst_nodes, num_timestamps
) -> tuple[Tensor, Tensor, Tensor]:
    src_node_indices = []
    src_time = []
    edge_indices = []
    idx = 0
    for timestamp in range(num_timestamps):
        for src_node_idx in range(num_src_nodes):
            src_node_indices.append(src_node_idx)
            src_time.append(timestamp)
            samples = random.sample(range(num_dst_nodes), random.randint(1, 10))
            edge_indices.extend([[idx, sample] for sample in samples])
            idx += 1
    src_node_indices = torch.tensor(src_node_indices)
    src_time = torch.tensor(src_time)
    edge_indices = torch.tensor(edge_indices).transpose(0, 1)
    sparse_coo = torch.sparse_coo_tensor(
        edge_indices,
        torch.ones(edge_indices.size(1), dtype=bool),
        (len(src_node_indices), num_dst_nodes),
    )
    dst_node_indices = sparse_coo.to_sparse_csr()

    return src_node_indices, dst_node_indices, src_time


def test_custom_link_dataset():
    num_src_nodes = 100
    num_dst_nodes = 50
    num_timestamps = 3
    src_node_indices, dst_node_indices, src_time = get_lp_setup(
        num_src_nodes, num_dst_nodes, num_timestamps
    )
    dataset = CustomLinkDataset(
        src_node_indices, dst_node_indices, num_dst_nodes, src_time
    )
    for i in range(len(dataset)):
        src, pos, time = dataset[i]
        assert src == src_node_indices[i]
        assert time == src_time[i]
        assert pos in dst_node_indices[i].indices()
