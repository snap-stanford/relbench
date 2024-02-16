from typing import Dict, List, Optional
from functools import partial

import torch
import torch.nn as nn
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
    def __init__(self, node_types, channels,
                 node_type_considered=None, num_filtered=20, sim_score_type='cos',
                 aggr_scheme='mpnn', normalize_score=True, selfjoin_aggr='sum'):
        super().__init__()
        # trick
        if sim_score_type is None:
            node_type_considered = None

        if node_type_considered is None:
            node_type_considered = []
        elif node_type_considered == 'all':
            node_type_considered = node_types
        elif type(node_type_considered) is str:
            node_type_considered = [node_type_considered]  # If only one type as str, treat it as a list
        else:
            assert type(node_type_considered) is list
        self.node_type_considered = node_type_considered
        self.sim_score_type = sim_score_type
        self.num_filtered = num_filtered
        self.aggr_scheme = aggr_scheme
        self.normalize_score = normalize_score
        self.selfjoin_aggr = selfjoin_aggr

        # Initialize the message fusion module
        if self.aggr_scheme == 'gat':
            self.conv_dict = torch.nn.ModuleDict()
            for node_type in self.node_type_considered:
                # self.conv_dict[node_type] = SAGEConv(channels, channels, aggr='sum')
                self.conv_dict[node_type] = GATConv(channels, channels, edge_dim=1)
        elif self.aggr_scheme == 'mpnn':
            self.msg_dict = torch.nn.ModuleDict()
            self.upd_dict = torch.nn.ModuleDict()
            for node_type in self.node_type_considered:
                self.msg_dict[node_type] = MLP(channel_list=[channels * 2, channels, channels])
                self.upd_dict[node_type] = MLP(channel_list=[channels * 2, channels, channels])
        else:
            raise NotImplementedError(self.aggr_scheme)

        # Initialize the similarity score computation module
        self.query_dict = torch.nn.ModuleDict()
        self.key_dict = torch.nn.ModuleDict()
        if self.sim_score_type == 'cos':
            pass
        elif self.sim_score_type == 'L2':
            for node_type in self.node_type_considered:
                self.key_dict[node_type] = nn.Linear(channels, channels)
        elif self.sim_score_type == 'attention':
            for node_type in self.node_type_considered:
                self.query_dict[node_type] = nn.Linear(channels, channels)
                self.key_dict[node_type] = nn.Linear(channels, channels)
        elif self.sim_score_type is None:
            pass
        else:
            raise NotImplementedError(self.sim_score_type)

    def forward(self, x_dict: Dict):
        upd_x_dict = {}
        for node_type, feature in x_dict.items():
            if node_type not in self.node_type_considered:
                upd_x_dict[node_type] = feature
                continue

            # Compute similarity score
            if self.sim_score_type == 'cos':
                sim_score = cosine_similarity(feature[:, None, :], feature[None, :, :], dim=-1)  # [N, N]
            elif self.sim_score_type == 'L2':
                feature = self.key_dict[node_type](feature)  # [N, H]
                sim_score = - torch.norm(feature[:, None, :] - feature[None, :, :], p=2, dim=-1) ** 2  # [N, N]
            elif self.sim_score_type == 'attention':
                q = self.query_dict[node_type](feature)  # [N, H]
                k = self.key_dict[node_type](feature)  # [N, H]
                sim_score = torch.matmul(k, q.transpose(0, 1))  # [N, N]
            else:
                raise NotImplementedError(self.sim_score_type)

            # Select Top K
            sim_score, index_sampled, = torch.topk(sim_score, k=self.num_filtered, dim=1)  # [N, K], [N, K]

            # Normalize
            if self.normalize_score:
                sim_score = torch.softmax(sim_score, dim=-1)  # [N, K]

            # Construct the graph over the retrieved entries
            edge_index_i = torch.arange(index_sampled.size(0)
                                        ).to(sim_score.device).unsqueeze(-1).repeat(1, index_sampled.size(1)).view(-1)
            # [NK]
            edge_index_j = index_sampled.view(-1)  # [NK]
            edge_index = torch.stack((edge_index_i, edge_index_j), dim=0)  # [2, NK]

            if self.aggr_scheme == 'gat':
                feature_out = self.conv_dict[node_type](feature, edge_index, sim_score.view(-1, 1))  # [N, H]
            elif self.aggr_scheme == 'mpnn':
                h_i, h_j = feature[edge_index[0]], feature[edge_index[1]]  # [M, H], M = N * K
                score = self.msg_dict[node_type](torch.cat((h_i, h_j), dim=-1)) * sim_score.view(-1, 1)  # [M, H]
                h_agg = scatter(score, edge_index[0], dim=0, reduce=self.selfjoin_aggr)  # [N, H]
                feature_out = feature + self.upd_dict[node_type](torch.cat((feature, h_agg), dim=-1))  # [N, H]
            else:
                raise NotImplementedError(self.aggr_scheme)

            upd_x_dict[node_type] = feature_out

        return upd_x_dict


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
                aggr=hetero_aggr,
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
