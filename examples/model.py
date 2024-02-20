from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch_frame.data import TensorFrame
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import EdgeType, NodeType

from relbench.external.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder


class Model(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=data.col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

    def forward(
        self,
        entity_table: NodeType,
        tf_dict: Dict[NodeType, TensorFrame],
        edge_index_dict: Dict[EdgeType, Tensor],
        seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Dict[NodeType, List[int]],
        num_sampled_edges_dict: Dict[EdgeType, List[int]],
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
    ) -> Tensor:
        x_dict = self.encoder(tf_dict)

        rel_time_dict = self.temporal_encoder(seed_time, time_dict, batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            edge_index_dict,
            num_sampled_nodes_dict,
            num_sampled_edges_dict,
        )

        out = self.head(x_dict[entity_table][: seed_time.size(0)])
        if (not self.training) and (clamp_min is not None) and (clamp_max is not None):
            out = torch.clamp(out, clamp_min, clamp_max)
        return out
