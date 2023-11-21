import argparse
import math
from typing import Dict, List, Tuple

import torch
import torch_frame
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_geometric.data import HeteroData
from torch_geometric.loader import NodeLoader
from torch_geometric.nn import MLP
from torch_geometric.sampler import NeighborSampler
from torch_geometric.typing import EdgeType, NodeType
from torchmetrics import AUROC, MeanAbsoluteError
from tqdm import tqdm

from rtb.data.task import TaskType
from rtb.datasets import get_dataset
from rtb.external.graph import (get_stype_proposal, get_train_table_input,
                                make_pkey_fkey_graph)
from rtb.external.nn import HeteroEncoder, HeteroGraphSAGE

# Stores the informative text columns to retain for each table:
dataset_to_informative_text_cols = {}
dataset_to_informative_text_cols["rtb-forum"] = {
    "postHistory": ["Text"],
    "users": ["AboutMe"],
    "posts": ["Body", "Title", "Tags"],
    "comments": ["Text"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rtb-forum")
parser.add_argument("--task", type=str, default="UserSumCommentScoresTask")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--num_neighbors", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=6)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = get_dataset(name=args.dataset, root="./data")
if args.task not in dataset.tasks:
    raise ValueError(
        f"'{args.dataset}' does not support the given task {args.task}. "
        f"Please choose the task from {list(dataset.tasks.keys())}."
    )

task = dataset.tasks[args.task]
train_table = dataset.make_train_table(args.task)
val_table = dataset.make_val_table(args.task)
test_table = dataset.make_test_table(args.task)

col_to_stype_dict = get_stype_proposal(dataset.db)
# Fix semantic type proposal:
if args.dataset == "rtb-forum":
    col_to_stype_dict["postHistory"]["PostHistoryTypeId"] = torch_frame.categorical
    col_to_stype_dict["posts"]["PostTypeId"] = torch_frame.categorical
# Drop text columns for now:
for stype_dict in col_to_stype_dict.values():
    for col_name, stype in list(stype_dict.items()):
        if stype == torch_frame.text_embedded:
            del stype_dict[col_name]

# TODO Add table materialization/saving logic.
data: HeteroData = make_pkey_fkey_graph(
    dataset.db,
    col_to_stype_dict=col_to_stype_dict,
)

sampler = NeighborSampler(  # Initialize sampler only once:
    data,
    num_neighbors=[args.num_neighbors, args.num_neighbors],
    time_attr="time",
)

loader_dict: Dict[str, NodeLoader] = {}
for split, table in [
    ("train", train_table),
    ("val", val_table),
    ("test", test_table),
]:
    table_input = get_train_table_input(train_table=table, task=task)
    entity_table = table_input.nodes[0]
    loader_dict[split] = NodeLoader(
        data,
        node_sampler=sampler,
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    metric_name = "AUROC"
    metric = AUROC(task="binary").to(device)
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    metric_name = "MAE"
    metric = MeanAbsoluteError(squared=False).to(device)
    higher_is_better = False


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
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=args.channels,
        )
        self.head = MLP(
            args.channels,
            out_channels=out_channels,
            num_layers=1,
        )

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Dict[NodeType, List[int]],
        num_sampled_edges_dict: Dict[EdgeType, List[int]],
    ) -> Tensor:
        x_dict = self.encoder(tf_dict)
        x_dict = self.gnn(
            x_dict,
            edge_index_dict,
            num_sampled_nodes_dict,
            num_sampled_edges_dict,
        )
        return self.head(x_dict[entity_table])


model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> Tuple[float, float]:
    model.train()

    metric.reset()
    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch.tf_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        loss = loss_fn(pred, batch[entity_table].y)
        loss.backward()
        optimizer.step()

        loss_accum += float(loss) * pred.size(0)
        count_accum += pred.size(0)
        metric.update(pred, batch[entity_table].y)

    return loss_accum / count_accum, float(metric.compute())


@torch.no_grad()
def test(loader: NodeLoader) -> float:
    model.eval()

    metric.reset()
    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = model(
            batch.tf_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        metric.update(pred, batch[entity_table].y)

    return float(metric.compute())


state_dict = None
best_val_metric = 0 if higher_is_better else math.inf
for epoch in range(1, args.epochs + 1):
    loss, train_metric = train()
    val_metric = test(loader_dict["val"])
    print(
        f"Epoch: {epoch:02d}, "
        f"Loss: {loss:.4f}, "
        f"Train {metric_name}: {train_metric:.4f}, "
        f"Val {metric_name}: {val_metric:.4f}"
    )

    if (higher_is_better and val_metric > best_val_metric) or (
        not higher_is_better and val_metric < best_val_metric
    ):
        best_val_metric = val_metric
        state_dict = model.state_dict()

model.load_state_dict(state_dict)
test_metric = test(loader_dict["test"])
print(f"Test {metric_name}: {test_metric:.4f}")
