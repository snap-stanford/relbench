import random

import torch

from relbench.external.loader import CustomLinkDataset


def get_lp_setup(num_src_nodes, num_dst_nodes):
    src_node_indices = torch.arange(num_src_nodes)
    edge_indices = []
    for src_idx in range(num_src_nodes):
        edge_indices.extend(
            [src_idx, dst_idx]
            for dst_idx in random.sample(range(num_dst_nodes), random.randint(1, 10))
        )
    edge_indices = torch.tensor(edge_indices).transpose(0, 1)
    sparse_coo = torch.sparse_coo_tensor(
        edge_indices,
        torch.ones(edge_indices.size(1), dtype=bool),
        (num_src_nodes, num_dst_nodes),
    )
    dst_node_indices = sparse_coo.to_sparse_csr()
    src_time = torch.randint(0, 4, (num_src_nodes,))
    return src_node_indices, dst_node_indices, src_time


def test_custom_link_dataset():
    num_src_nodes = 100
    num_dst_nodes = 50
    src_node_indices, dst_node_indices, src_time = get_lp_setup(
        num_src_nodes, num_dst_nodes
    )
    dataset = CustomLinkDataset(
        src_node_indices, dst_node_indices, num_dst_nodes, src_time
    )
    for i in range(len(dataset)):
        src, pos, time = dataset[i]
        assert src == src_node_indices[i]
        assert time == src_time[i]
        assert pos in dst_node_indices[i].indices()
