from functools import partial
from typing import Dict, List, Optional
from relbench.data.task_base import TaskType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torch_geometric.nn import MLP, GATConv, HeteroConv, LayerNorm, SAGEConv
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import trim_to_layer
from torch_scatter import scatter

conv_name_to_func = {
    "sage": SAGEConv,
    "gat": partial(GATConv, add_self_loops=False),
}


class SelfJoinLayer(torch.nn.Module):
    def __init__(
        self,
        node_types,
        channels,
        node_type_considered=None,
        num_filtered=20,
        sim_score_type="cos",
        aggr_scheme="mpnn",
        normalize_score=True,
        selfjoin_aggr="sum",
    ):
        super().__init__()
        # trick
        if sim_score_type is None:
            node_type_considered = None

        if node_type_considered is None:
            node_type_considered = []
        elif node_type_considered == "all":
            node_type_considered = node_types
        elif type(node_type_considered) is str:
            node_type_considered = [
                node_type_considered
            ]  # If only one type as str, treat it as a list
        else:
            assert type(node_type_considered) is list
        self.node_type_considered = node_type_considered
        self.sim_score_type = sim_score_type
        self.num_filtered = num_filtered
        self.aggr_scheme = aggr_scheme
        self.normalize_score = normalize_score
        self.selfjoin_aggr = selfjoin_aggr

        # Initialize the message fusion module
        if self.aggr_scheme == "gat":
            self.conv_dict = torch.nn.ModuleDict()
            for node_type in self.node_type_considered:
                # self.conv_dict[node_type] = SAGEConv(channels, channels, aggr='sum')
                self.conv_dict[node_type] = GATConv(channels, channels, edge_dim=1)
        elif self.aggr_scheme == "mpnn":
            self.msg_dict = torch.nn.ModuleDict()
            self.upd_dict = torch.nn.ModuleDict()
            for node_type in self.node_type_considered:
                self.msg_dict[node_type] = MLP(
                    channel_list=[channels * 2, channels, channels]
                )
                self.upd_dict[node_type] = MLP(
                    channel_list=[channels * 2, channels, channels]
                )
        else:
            raise NotImplementedError(self.aggr_scheme)

        # Initialize the similarity score computation module
        self.query_dict = torch.nn.ModuleDict()
        self.key_dict = torch.nn.ModuleDict()
        if self.sim_score_type == "cos":
            pass
        elif self.sim_score_type == "L2":
            for node_type in self.node_type_considered:
                self.key_dict[node_type] = nn.Linear(channels, channels)
        elif self.sim_score_type == "attention":
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
            if self.sim_score_type == "cos":
                sim_score = cosine_similarity(
                    feature[:, None, :], feature[None, :, :], dim=-1
                )  # [N, N]
            elif self.sim_score_type == "L2":
                feature = self.key_dict[node_type](feature)  # [N, H]
                sim_score = (
                    -torch.norm(feature[:, None, :] - feature[None, :, :], p=2, dim=-1)
                    ** 2
                )  # [N, N]
            elif self.sim_score_type == "attention":
                q = self.query_dict[node_type](feature)  # [N, H]
                k = self.key_dict[node_type](feature)  # [N, H]
                sim_score = torch.matmul(k, q.transpose(0, 1))  # [N, N]
            else:
                raise NotImplementedError(self.sim_score_type)

            # Select Top K
            sim_score, index_sampled = torch.topk(
                sim_score, k=min(self.num_filtered, sim_score.shape[1]), dim=1
            )  # [N, K], [N, K]

            # Normalize
            if self.normalize_score:
                sim_score = torch.softmax(sim_score, dim=-1)  # [N, K]

            # Construct the graph over the retrieved entries
            edge_index_i = (
                torch.arange(index_sampled.size(0))
                .to(sim_score.device)
                .unsqueeze(-1)
                .repeat(1, index_sampled.size(1))
                .view(-1)
            )
            # [NK]
            edge_index_j = index_sampled.view(-1)  # [NK]
            edge_index = torch.stack((edge_index_i, edge_index_j), dim=0)  # [2, NK]

            if self.aggr_scheme == "gat":
                feature_out = self.conv_dict[node_type](
                    feature, edge_index, sim_score.view(-1, 1)
                )  # [N, H]
            elif self.aggr_scheme == "mpnn":
                h_i, h_j = (
                    feature[edge_index[0]],
                    feature[edge_index[1]],
                )  # [M, H], M = N * K
                score = self.msg_dict[node_type](
                    torch.cat((h_i, h_j), dim=-1)
                ) * sim_score.view(
                    -1, 1
                )  # [M, H]
                h_agg = scatter(
                    score, edge_index[0], dim=0, reduce=self.selfjoin_aggr
                )  # [N, H]
                feature_out = feature + self.upd_dict[node_type](
                    torch.cat((feature, h_agg), dim=-1)
                )  # [N, H]
            else:
                raise NotImplementedError(self.aggr_scheme)

            upd_x_dict[node_type] = feature_out

        return upd_x_dict


class SelfJoinLayerWithRetrieval(torch.nn.Module):
    def __init__(
        self,
        node_types,
        channels,
        batch_size,
        node_type_considered=None,
        num_filtered=20,
        sim_score_type="cos",
        aggr_scheme="mpnn",
        normalize_score=True,
        selfjoin_aggr="sum",
        selfjoin_dropout=0.0,
        memory_bank_size=4096,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
    ):
        super().__init__()
        # trick
        if sim_score_type is None:
            node_type_considered = None

        if node_type_considered is None:
            node_type_considered = []
        elif node_type_considered == "all":
            node_type_considered = node_types
        elif type(node_type_considered) is str:
            node_type_considered = [
                node_type_considered
            ]  # If only one type as str, treat it as a list
        else:
            assert type(node_type_considered) is list
        self.node_type_considered = node_type_considered
        self.sim_score_type = sim_score_type
        self.num_filtered = num_filtered
        self.aggr_scheme = aggr_scheme
        self.normalize_score = normalize_score
        self.selfjoin_aggr = selfjoin_aggr
        self.selfjoin_dropout = selfjoin_dropout
        self.expected_batch_size = batch_size

        # Initialize the message fusion module
        if self.aggr_scheme == "gat":
            self.conv_dict = torch.nn.ModuleDict()
            for node_type in self.node_type_considered:
                # self.conv_dict[node_type] = SAGEConv(channels, channels, aggr='sum')
                self.conv_dict[node_type] = GATConv(channels, channels, edge_dim=1)
        elif self.aggr_scheme == "mpnn":
            self.msg_dict = torch.nn.ModuleDict()
            self.upd_dict = torch.nn.ModuleDict()
            for node_type in self.node_type_considered:
                self.msg_dict[node_type] = nn.Linear(channels * 3, channels)
                self.upd_dict[node_type] = MLP(
                    channel_list=[channels * 2, channels, channels],
                    dropout=selfjoin_dropout,
                )
        else:
            raise NotImplementedError(self.aggr_scheme)

        # Initialize the similarity score computation module
        self.query_dict = torch.nn.ModuleDict()
        self.key_dict = torch.nn.ModuleDict()
        if self.sim_score_type == "cos":
            pass
        elif self.sim_score_type == "L2":
            for node_type in self.node_type_considered:
                self.key_dict[node_type] = nn.Linear(channels, channels)
        elif self.sim_score_type == "attention":
            for node_type in self.node_type_considered:
                # self.query_dict[node_type] = nn.Linear(channels, channels)
                self.key_dict[node_type] = nn.Linear(channels, channels)
        elif self.sim_score_type is None:
            pass
        else:
            raise NotImplementedError(self.sim_score_type)

        # initialize memory bank for the self-join layer
        self.bank_size = memory_bank_size
        self.memory_bank = {
            "x": torch.randn(self.bank_size, channels),
            "y": torch.zeros(self.bank_size),
            "seed_time": torch.full((self.bank_size,), float("-inf")),
        }

        self.task_type = task_type

        if task_type == TaskType.BINARY_CLASSIFICATION:
            self.y_emb = torch.nn.Embedding(
                2, channels
            )  
        elif task_type == TaskType.REGRESSION:
            self.y_emb = torch.nn.Linear(1, channels)
        else:
            raise NotImplementedError(task_type)
        self.pointer = 0  # memory bank pointer

    def update_memory_bank(self, x_dict: Dict, y: Tensor, seed_time: Tensor):
        for node_type, feature in x_dict.items():
            if node_type not in self.node_type_considered:
                continue

            if (
                feature.shape[0] != self.expected_batch_size
                or y.shape[0] != self.expected_batch_size
            ):
                continue  # skip the update if the batch size is not as expected - handles the last batch

            # update memory bank
            self.memory_bank["x"][
                self.pointer : self.pointer + feature.size(0)
            ] = feature.clone().detach()
            self.memory_bank["y"][
                self.pointer : self.pointer + feature.size(0)
            ] = y.clone().detach()
            self.memory_bank["seed_time"][
                self.pointer : self.pointer + feature.size(0)
            ] = seed_time.clone().detach()
            self.pointer = (self.pointer + feature.size(0)) % self.bank_size

    def forward(self, x_dict: Dict, y: Tensor = None, seed_time: Tensor = None):
        upd_x_dict = {}
        for node_type, feature in x_dict.items():
            if node_type not in self.node_type_considered:
                upd_x_dict[node_type] = feature
                continue

            # Compute similarity score
            if self.sim_score_type == "attention":
                q = self.key_dict[node_type](
                    F.normalize(feature, p=2, dim=-1)
                )  # [N, H]

                # retrieve memory bank
                memory_feature = self.memory_bank["x"].to(feature.device)  # [N_bank, H]
                memory_feature = F.normalize(memory_feature, p=2, dim=-1)
                k = self.key_dict[node_type](memory_feature)  # [N_bank, H]
                sim_score = torch.matmul(q, k.transpose(0, 1))  # [N, N_bank]
            elif self.sim_score_type == "L2":
                feat = F.normalize(feature, p=2, dim=-1)
                feat = self.key_dict[node_type](feat)  # [N, H]
                memory_feature = self.memory_bank["x"].to(feature.device)  # [N_bank, H]
                memory_feature = F.normalize(memory_feature, p=2, dim=-1)
                mem_feat = self.key_dict[node_type](memory_feature)  # [N_bank, H]
                sim_score = (
                    -torch.norm(feat[:, None, :] - mem_feat[None, :, :], p=2, dim=-1)
                    ** 2
                )  # [N, N]
            else:
                raise NotImplementedError(self.sim_score_type)

            # Mask nodes whose seed time is greater than the current node to avoid time leakage
            mask = seed_time[:, None] < self.memory_bank["seed_time"][None, :].to(
                sim_score.device
            )  # [N, bank_size]
            nonzero_indices = (~mask).int().nonzero()  # [N, 2]
            edge_index = torch.t(nonzero_indices).contiguous()  # [2, |E|]

            sim_score = sim_score.masked_fill(mask, -float("inf"))  # [N, bank_size]

            if self.normalize_score:
                sim_score = torch.softmax(sim_score, dim=-1)  # [N, K]

            # retrieve memory bank labels
            memory_y = self.memory_bank["y"].to(feature.device)  # [N_ban]
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                memory_y = self.y_emb(memory_y.long())  # [N, K, H]
            elif self.task_type == TaskType.REGRESSION:
                memory_y = self.y_emb(memory_y.view(-1, 1)) 
            memory_y = memory_y.view(-1, memory_y.size(-1))  # [NK, H]

            if self.aggr_scheme == "gat":
                feature_out = self.conv_dict[node_type](
                    feature, edge_index, sim_score.view(-1, 1)
                )  # [N, H]
            elif self.aggr_scheme == "mpnn":
                h_i, h_j = (
                    feature[edge_index[0]],
                    memory_feature[edge_index[1]],
                )  # [M, H], M = N * K
                memory_y = memory_y[edge_index[1]]
                score = self.msg_dict[node_type](
                    torch.cat((h_i, h_j, memory_y), dim=-1)
                ) * sim_score[~mask].view(
                    -1, 1
                )  # [M, H]
                h_agg = scatter(
                    score, edge_index[0], dim=0, reduce=self.selfjoin_aggr
                )  # [N, H]
                feature_out = feature + self.upd_dict[node_type](
                    torch.cat((feature, h_agg), dim=-1)
                )  # [N, H]
            else:
                raise NotImplementedError(self.aggr_scheme)

            upd_x_dict[node_type] = feature_out

        if self.training:
            self.update_memory_bank(upd_x_dict, y, seed_time)

        return upd_x_dict


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        conv: str,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        batch_size: int,
        aggr: str = "mean",
        hetero_aggr: str = "sum",
        num_layers: int = 2,
        use_self_join: bool = False,
        use_self_join_with_retrieval: bool = False,
        feature_dropout: float = 0.0,
        memory_bank_size: int = 4096,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
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

        assert not (
            use_self_join and use_self_join_with_retrieval
        )  # only one of them can be True
        self.use_self_join = use_self_join
        self.use_self_join_with_retrieval = use_self_join_with_retrieval
        self.feature_dropout = feature_dropout

        self.self_joins = torch.nn.ModuleList()

        assert not (
            use_self_join and use_self_join_with_retrieval
        )  # only one of them can be True
        if use_self_join:
            for _ in range(num_layers):
                self.self_joins.append(SelfJoinLayer(node_types, channels, **kwargs))
        elif use_self_join_with_retrieval:
            for _ in range(num_layers):
                self.self_joins.append(
                    SelfJoinLayerWithRetrieval(
                        node_types,
                        channels,
                        batch_size,
                        memory_bank_size=memory_bank_size,
                        task_type=task_type,
                        **kwargs,
                    )
                )

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
        seed_time: Tensor = None,
        y: Tensor = None,
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

            if self.use_self_join:
                x_dict = self.self_joins[i](x_dict)

            elif self.use_self_join_with_retrieval:
                x_dict = self.self_joins[i](x_dict, y, seed_time)

            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict
