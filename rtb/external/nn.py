from typing import Any, Dict, List, Optional

import torch
import torch_frame
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_geometric.nn import HeteroConv, LayerNorm, SAGEConv
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import trim_to_layer


class HeteroEncoder(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in col_names_dict.keys():
            encoders = {
                torch_frame.categorical: torch_frame.nn.EmbeddingEncoder(),
                torch_frame.numerical: torch_frame.nn.LinearEncoder(),
            }
            encoders = {
                stype: encoders[stype] for stype in col_names_dict[node_type].keys()
            }

            self.encoders[node_type] = torch_frame.nn.StypeWiseFeatureEncoder(
                out_channels=channels,
                col_stats=col_stats_dict[node_type],
                col_names_dict=col_names_dict[node_type],
                stype_encoder_dict=encoders,
            )

            # TODO Add conv+decoder
            # TODO Add relative time

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {
            node_type: self.encoders[node_type](tf)[0]
            for node_type, tf in tf_dict.items()
        }
        x_dict = {node_type: x.mean(dim=1) for node_type, x in x_dict.items()}
        return x_dict


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        num_layers: int = 2,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            # Trim graph and features to only hold required data per layer:
            if num_sampled_nodes_dict is not None:
                assert num_sampled_edges_dict is not None
                x_dict, edge_index_dict, _ = trim_to_layer(
                    layer=i,
                    num_sampled_nodes_per_hop=num_sampled_nodes_dict,
                    num_sampled_edges_per_hop=num_sampled_edges_dict,
                    x=x_dict,
                    edge_index=edge_index_dict,
                )

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict
