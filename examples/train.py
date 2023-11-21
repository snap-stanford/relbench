import argparse
import math

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from torch_frame import stype
from torch_geometric.loader import NodeLoader
from torch_geometric.nn import MLP
from torch_geometric.sampler import NeighborSampler
from torchmetrics import AUROC, Accuracy, MeanAbsoluteError
from tqdm import tqdm

from rtb.data.task import TaskType
from rtb.datasets import get_dataset
from rtb.external.graph import (get_stype_proposal, get_train_table_input,
                                make_pkey_fkey_graph)
from rtb.external.nn import HeteroEncoder, HeteroGraphSAGE

# Stores the informative text columns to retain for each table.
_dataset_to_informative_text_cols = {}
_dataset_to_informative_text_cols["rtb-forum"] = {
    "postHistory": ["Text"],
    "users": ["AboutMe"],
    "posts": ["Body", "Title", "Tags"],
    "comments": ["Text"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default="rtb-forum",
                    choices=["rtb-forum"])
parser.add_argument("--task", type=str, default="UserSumCommentScoresTask")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--num_neighbors", type=int, default=64)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = get_dataset(name=args.dataset, root="/home/weihua/code/rtb/data")
if args.task not in dataset.tasks:
    raise ValueError(
        f"{args.dataset} does not support the given task {args.task}. Please "
        f"choose the task from {list(dataset.tasks.keys())}")

inferred_col_to_stype_dict = get_stype_proposal(dataset.db)

# Drop text columns for now.
# TODO: Re-include _dataset_to_informative_text_cols for each dataset and
# and support text columns.
for table_name, col_to_stype in inferred_col_to_stype_dict.items():
    filtered_col_to_stype = {
        key: value
        for key, value in col_to_stype.items() if value != stype.text_embedded
    }
    inferred_col_to_stype_dict[table_name] = filtered_col_to_stype

# TODO: Add table materialization/saving logic so that we don't need to
# re-compute text embeddings every time. Pass :obj:`path`.
data = make_pkey_fkey_graph(
    dataset.db,
    col_to_stype_dict=inferred_col_to_stype_dict,
)

task_type = dataset.tasks[args.task].task_type

if task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fun = BCEWithLogitsLoss()
    metric_computer = AUROC(task="binary").to(device)
    higher_is_better = True
elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
    # TODO Expose num_classes in dataset object.
    num_classes = 10
    out_channels = num_classes
    loss_fun = CrossEntropyLoss()
    metric_computer = Accuracy(task="multiclass",
                               num_classes=num_classes).to(device)
    higher_is_better = True
elif task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fun = L1Loss()
    metric_computer = MeanAbsoluteError(squared=False).to(device)
    higher_is_better = False

node_to_col_names_dict = {
    node_type: data[node_type].tf.col_names_dict
    for node_type in data.node_types
}

encoder = HeteroEncoder(args.channels, node_to_col_names_dict,
                        data.col_stats_dict).to(device)
gnn = HeteroGraphSAGE(data.node_types, data.edge_types,
                      args.channels).to(device)
head = MLP(args.channels, out_channels=out_channels, num_layers=1).to(device)

sampler = NeighborSampler(
    data,
    num_neighbors=[args.num_neighbors, args.num_neighbors],
    time_attr="time",
)

train_table = dataset.make_train_table(args.task)
val_table = dataset.make_val_table(args.task)
test_table = dataset.make_test_table(args.task)

# Ensure that mini-batch training works ###################################

loader_dict = {}
for split_name, label_table in [
    ("train", train_table),
    ("val", val_table),
    ("test", test_table),
]:
    label_table_input = get_train_table_input(
        train_table=label_table,
        task=dataset.tasks[args.task],
    )
    shuffle = True if split_name == "train" else False
    loader = NodeLoader(
        data,
        node_sampler=sampler,
        input_nodes=label_table_input.nodes,
        input_time=label_table_input.time,
        transform=label_table_input.transform,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    loader_dict[split_name] = loader

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(gnn.parameters()) +
    list(head.parameters()),
    lr=args.lr,
)

entity_node = label_table_input.nodes[0]


def train() -> float:
    encoder.train()
    gnn.train()
    head.train()

    loss_accum = 0.0
    count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        optimizer.zero_grad()
        batch = batch.to(device)
        x_dict = encoder(batch.tf_dict)
        x_dict = gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        pred = head(x_dict[entity_node])
        if pred.size(1) == 1:
            pred = pred.view(-1, )

        optimizer.zero_grad()
        loss = loss_fun(pred, batch[entity_node].y)
        loss.backward()
        loss_accum += loss.item() * len(pred)
        count_accum += len(pred)

        optimizer.step()
    return loss_accum / count_accum


def eval(loader: NodeLoader) -> float:
    encoder.eval()
    gnn.eval()
    head.eval()

    metric_computer.reset()
    for batch in tqdm(loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        x_dict = encoder(batch.tf_dict)
        x_dict = gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        pred = head(x_dict[entity_node])
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(dim=-1)
        elif task_type == TaskType.REGRESSION:
            pred = pred.view(-1, )
        metric_computer.update(pred, batch[entity_node].y)
    return metric_computer.compute().item()


if higher_is_better:
    best_val_metric = 0
else:
    best_val_metric = math.inf

for epoch in range(args.epochs):
    print(f"===Epoch {epoch}")
    train_loss = train()
    print(f"Train Loss: {train_loss:.4f}")
    train_metric = eval(loader_dict["train"])
    val_metric = eval(loader_dict["val"])

    if higher_is_better:
        if val_metric > best_val_metric:
            best_val_metric = val_metric
    else:
        if val_metric < best_val_metric:
            best_val_metric = val_metric

    print(f"Train metric: {train_metric:.4f}, Val metric: {val_metric:.4f}")

print(f"Best val metric: {best_val_metric:.4f}")
