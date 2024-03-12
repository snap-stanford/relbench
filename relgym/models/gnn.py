from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATConv, HeteroConv, LayerNorm, SAGEConv
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import trim_to_layer

conv_name_to_func = {
    "sage": SAGEConv,
    "gat": partial(GATConv, add_self_loops=False),
}


class HybridConv(nn.Module):
    def __init__(self, conv_func, aggr, channels):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in aggr:
            _conv = conv_func((channels, channels), channels, aggr=_)
            self.convs.append(_conv)

    def forward(self, *args, **kwargs):
        return sum([conv(*args, **kwargs) for conv in self.convs])


def parse_conv_func(conv, aggr, channels):
    conv_func = conv_name_to_func[conv]
    if type(aggr) is str and aggr != "hybrid":
        return conv_func((channels, channels), channels, aggr=aggr)
    else:
        if aggr == "hybrid":
            aggr = ["sum", "mean"]
        return HybridConv(conv_func, aggr, channels)


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        conv: str,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        hetero_aggr: str = "sum",
        num_layers: int = 2,
        feature_dropout: float = 0.0,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            _conv = HeteroConv(
                {
                    edge_type: parse_conv_func(conv, aggr, channels)
                    for edge_type in edge_types
                },
                aggr=hetero_aggr,
            )
            self.convs.append(_conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

        self.feature_dropout = feature_dropout

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

        # Apply dropout to the input features
        x_dict = {
            key: nn.functional.dropout(
                x, p=self.feature_dropout, training=self.training
            )
            for key, x in x_dict.items()
        }

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
