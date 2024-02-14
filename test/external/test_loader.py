import random

import torch

from relbench.external.loader import CustomLinkDataset


def get_lp_setup(num_src_nodes, num_dst_nodes):
    src_node_indices = torch.arange(num_src_nodes)
    dst_node_indices_list = [
        random.sample(range(num_dst_nodes), random.randint(1, 10))
        for _ in range(num_src_nodes)
    ]
    src_time = torch.randint(0, 4, (num_src_nodes,))
    return src_node_indices, dst_node_indices_list, src_time


def test_custom_link_dataset():
    num_src_nodes = 100
    num_dst_nodes = 50
    src_node_indices, dst_node_indices_list, src_time = get_lp_setup(
        num_src_nodes, num_dst_nodes
    )
    dataset = CustomLinkDataset(
        src_node_indices, dst_node_indices_list, num_dst_nodes, src_time
    )
    for i in range(len(dataset)):
        src, pos, neg, time = dataset[i]
        assert src == src_node_indices[i]
        assert time == src_time[i]
        assert neg not in dst_node_indices_list[i]
        assert pos in dst_node_indices_list[i]
