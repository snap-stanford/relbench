import torch
from torch import Tensor
from typing import Dict, List
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.nn import MLP
from torch_frame import TensorFrame

from relgym.config import cfg
from relgym.models.feature_encoder import HeteroEncoder, HeteroTemporalEncoder
from relgym.models.gnn import HeteroGNN


def create_model(data, entity_table, to_device=True):
    r"""
    Create model for graph machine learning

    Args:
        to_device (string): The devide that the model will be transferred to
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = HeteroEncoder(
                channels=cfg.model.channels,
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
                channels=cfg.model.channels,
            )
            self.gnn = HeteroGNN(
                conv=cfg.model.conv,
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=cfg.model.channels,
                aggr=cfg.model.aggr,
            )
            self.head = MLP(
                cfg.model.channels,
                out_channels=1,  # TODO: hard coding here, need to polish
                num_layers=1,
            )

        def forward(
                self,
                tf_dict: Dict[NodeType, TensorFrame],
                edge_index_dict: Dict[EdgeType, Tensor],
                seed_time: Tensor,
                time_dict: Dict[NodeType, Tensor],
                batch_dict: Dict[NodeType, Tensor],
                num_sampled_nodes_dict: Dict[NodeType, List[int]],
                num_sampled_edges_dict: Dict[EdgeType, List[int]],
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

            return self.head(x_dict[entity_table][: seed_time.size(0)])

    model = Model()
    if to_device:
        model.to(torch.device(cfg.device))
    return model
