import argparse
import copy
import math
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch_frame
from text_embedder import GloveTextEmbedding
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import TensorFrame
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import MLP
from torch_geometric.seed import seed_everything
from torch_geometric.typing import EdgeType, NodeType
import torch_geometric.transforms as T

from tqdm import tqdm

from relbench.data import RelBenchDataset
from relbench.data.task_base import TaskType
from relbench.datasets import get_dataset
from relbench.external.graph import (
    get_stype_proposal,
    get_link_train_table_input,
    make_pkey_fkey_graph,
)
from relbench.external.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder

# Stores the informative text columns to retain for each table:
dataset_to_informative_text_cols = {}
dataset_to_informative_text_cols["rel-stackex"] = {
    "postHistory": ["Text"],
    "users": ["AboutMe"],
    "posts": ["Body", "Title", "Tags"],
    "comments": ["Text"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stackex")
parser.add_argument("--task", type=str, default="rel-stackex-comment-on-post")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--neg_sampling_ratio", type=float, default=2.0)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

root_dir = "./data"

# TODO: remove process=True once correct data/task is uploaded.
dataset: RelBenchDataset = get_dataset(name=args.dataset, process=True)
task = dataset.get_task(args.task, process=True)

col_to_stype_dict = get_stype_proposal(dataset.db)
informative_text_cols: Dict = dataset_to_informative_text_cols[args.dataset]
for table_name, stype_dict in col_to_stype_dict.items():
    for col_name, stype in list(stype_dict.items()):
        # Remove text columns except for the informative ones:
        if stype == torch_frame.text_embedded:
            if col_name not in informative_text_cols.get(table_name, []):
                del stype_dict[col_name]

data: HeteroData = make_pkey_fkey_graph(
    dataset.db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=os.path.join(root_dir, f"{args.dataset}_materialized_cache"),
)


loader_dict: Dict[str, LinkNeighborLoader] = {}
for split, table in [
    ("train", task.train_table),
    ("val", task.val_table),
    ("test", task.test_table),
]:
    table_input = get_link_train_table_input(table=table, task=task)
    
    if split == "train": # TODO (joshrob) remove once fixed val/test indexing bug
        # add training links to data
        edge_type = table_input.edge_label_index[0]
        data[edge_type].edge_index = table_input.edge_label_index[1]
        #data[train_edge_type].edge_label_index = table_input.edge_label_index[1]
        data[edge_type].edge_label = torch.ones(table_input.edge_label_index[1].shape[1])


    loader_dict[split] = LinkNeighborLoader(
        data,
        num_neighbors=[args.num_neighbors, args.num_neighbors],
        neg_sampling_ratio=args.neg_sampling_ratio, # negatives are sampled on the fly at ratio 2:1
        time_attr="time",
        edge_label=data[edge_type].edge_label,
        edge_label_index=(edge_type, data[edge_type].edge_index),
        edge_label_time=table_input.time,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

if task.task_type == TaskType.LINK_PREDICTION:
    out_channels = 1
    tune_metric = "mrr"
    higher_is_better = True


class LinkPredictor(torch.nn.Module):
    # borrowed from https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ddi/gnn.py
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.5):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=args.channels,
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
            channels=args.channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=args.channels,
            aggr=args.aggr,
        )
        self.head = LinkPredictor(
            in_channels=args.channels,
            hidden_channels=args.channels,
            out_channels=out_channels,
            num_layers=2,
        )

    def forward(
        self,
        tf_dict: Dict[NodeType, TensorFrame],
        edge_index_dict: Dict[EdgeType, Tensor],
        edge_label_index: Tensor,
        #seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Dict[NodeType, List[int]],
        num_sampled_edges_dict: Dict[EdgeType, List[int]],
    ) -> Tensor:
        x_dict = self.encoder(tf_dict)


        """
        # TODO (joshrob) fix seed time to enable relative time encoding      

        rel_time_dict = self.temporal_encoder(seed_time, time_dict, batch_dict)

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        """
        breakpoint()
        x_dict = self.gnn(
            x_dict,
            edge_index_dict,
            num_sampled_nodes_dict,
            num_sampled_edges_dict,
        )
        breakpoint()

        x_src = x_dict[task.source_entity_table][edge_label_index[0]]
        x_dst = x_dict[task.destination_entity_table][edge_label_index[1]]
        return self.head(x_src, x_dst)


model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> Dict[str, float]:
    model.train()

    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        batch = batch.to(device)

        optimizer.zero_grad()
        edge_label = batch[(task.source_entity_table, "train_link",task.destination_entity_table)].edge_label
        edge_label_index = batch[(task.source_entity_table, "train_link",task.destination_entity_table)].edge_label_index

        out = model(
            batch.tf_dict,
            batch.edge_index_dict,
            edge_label_index,
            #batch[entity_table].seed_time,
            batch.time_dict,
            batch.batch_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        ).squeeze()

        loss = F.binary_cross_entropy_with_logits(out, edge_label)

        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * out.size(0)
        count_accum += out.size(0)

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: LinkNeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
        batch = batch.to(device)

        edge_label_index = batch[(task.source_entity_table, "train_link",task.destination_entity_table)].edge_label_index

        pred = model(
            batch.tf_dict,
            batch.edge_index_dict,
            edge_label_index,
            #batch[entity_table].seed_time,
            batch.time_dict,
            batch.batch_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        ).squeeze()

        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


state_dict = None
best_val_metric = 0 if higher_is_better else math.inf
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.val_table, args.neg_sampling_ratio)
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

    if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
        not higher_is_better and val_metrics[tune_metric] < best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())

model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.val_table, args.neg_sampling_ratio)
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred, task.test_table, args.neg_sampling_ratio)
print(f"Best test metrics: {test_metrics}")
