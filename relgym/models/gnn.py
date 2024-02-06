from typing import Dict, List, Optional
from functools import partial

import torch
from torch import Tensor
from torch_geometric.nn import HeteroConv, LayerNorm, SAGEConv, GATConv
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import trim_to_layer
from torch_geometric.nn import MLP
from torch_scatter import scatter
from torch.nn.functional import cosine_similarity


conv_name_to_func = {
    'sage': SAGEConv,
    'gat': partial(GATConv, add_self_loops=False),
}


class SelfJoinLayer(torch.nn.Module):
    def __init__(self, node_types, channels, node_type_considered=None):
        super().__init__()
        if node_type_considered is None:
            node_type_considered = []
        elif node_type_considered == 'all':
            node_type_considered = node_types
        else:
            assert type(node_type_considered) is list
        self.score_type = 'mlp'
        self.msg_dict = torch.nn.ModuleDict()
        self.upd_dict = torch.nn.ModuleDict()

        self.node_type_considered = node_type_considered
        for node_type in node_types:
            if node_type not in self.node_type_considered:
                continue
            self.msg_dict[node_type] = MLP(channel_list=[channels * 2, channels, channels])
            self.upd_dict[node_type] = MLP(channel_list=[channels * 2, channels, channels])

    def forward(self, x_dict: Dict):
        upd_x_dict = {}
        for node_type, feature in x_dict.items():
            if node_type not in self.node_type_considered:
                upd_x_dict[node_type] = feature
                continue
            # Aggregate feature with similarity weighting
            sim_score = cosine_similarity(feature[:, None, :], feature[None, :, :], dim=-1) + 1  # [N, N]
            # sort
            index_sampled = torch.topk(sim_score, k=20, dim=1).indices  # [N, K]
            edge_index_i = torch.arange(index_sampled.size(0)
                                        ).to(sim_score.device).unsqueeze(-1).repeat(1, index_sampled.size(1)).view(-1)
            # [NK]
            edge_index_j = index_sampled.view(-1)  # [NK]
            edge_index = torch.stack((edge_index_i, edge_index_j), dim=0)  # [2, NK]

            h_i, h_j = feature[edge_index[0]], feature[edge_index[1]]  # [M, H], M = N * K
            # Compute score
            if self.score_type == 'mlp':
                score = self.msg_dict[node_type](torch.cat((h_i, h_j), dim=-1))  # [M, H]
            elif self.score_type == 'cos':
                score = (h_i * h_j).sum(dim=-1, keepdim=True) / (
                            torch.norm(h_i, dim=-1, p=2, keepdim=True) * torch.norm(h_j, dim=-1, p=2, keepdim=True)
                )  # [M, 1]
            else:
                raise NotImplementedError(self.score_type)
            # Aggregate
            if self.score_type == 'mlp':
                h_agg = scatter(score, edge_index[0], dim=0, reduce='sum')  # [N, H]
            elif self.score_type == 'cos':
                h_agg = scatter(score * h_j, edge_index[0], dim=0, reduce='sum')  # [N, H]
            else:
                raise NotImplementedError(self.score_type)
            feature = feature + self.upd_dict[node_type](torch.cat((feature, h_agg), dim=-1))  # [N, H]
            upd_x_dict[node_type] = feature
        return upd_x_dict


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        conv: str,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
        use_self_join: bool = False,
        **kwargs,
    ):
        super().__init__()
        conv_func = conv_name_to_func[conv]

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: conv_func((channels, channels), channels, aggr=aggr)
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

        self.use_self_join = use_self_join
        self.self_joins = torch.nn.ModuleList()
        if use_self_join:
            for _ in range(num_layers):
                self.self_joins.append(SelfJoinLayer(node_types, channels, **kwargs))

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

            if self.use_self_join:
                x_dict = self.self_joins[i](x_dict)

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict
