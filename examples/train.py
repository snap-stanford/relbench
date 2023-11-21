import argparse

import torch
import torch.nn.functional as F
from torch_frame import stype
from torch_geometric.loader import NodeLoader
from torch_geometric.nn import MLP
from torch_geometric.sampler import NeighborSampler
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
parser.add_argument("--dataset", type=str, default="rtb-forum", choices=["rtb-forum"])
parser.add_argument("--task", type=str, default="UserSumCommentScoresTask")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--channels", type=int, default=64)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = get_dataset(name=args.dataset, root="/home/weihua/code/rtb/data")
if args.task not in dataset.tasks:
    raise ValueError(
        f"{args.dataset} does not support the given task {args.task}. Please "
        f"choose the task from {list(dataset.tasks.keys())}"
    )

inferred_col_to_stype_dict = get_stype_proposal(dataset.db)

# Drop text columns for now.
for table_name, col_to_stype in inferred_col_to_stype_dict.items():
    filtered_col_to_stype = {
        key: value
        for key, value in col_to_stype.items()
        if value != stype.text_embedded
    }
    inferred_col_to_stype_dict[table_name] = filtered_col_to_stype

data = make_pkey_fkey_graph(
    dataset.db,
    col_to_stype_dict=inferred_col_to_stype_dict,
)

node_to_col_names_dict = {
    node_type: data[node_type].tf.col_names_dict for node_type in data.node_types
}

encoder = HeteroEncoder(args.channels, node_to_col_names_dict, data.col_stats_dict).to(
    device
)
gnn = HeteroGraphSAGE(data.node_types, data.edge_types, args.channels).to(device)
head = MLP(args.channels, out_channels=1, num_layers=1).to(device)

sampler = NeighborSampler(
    data,
    num_neighbors=[64, 64],
    time_attr="time",
)

train_table = dataset.make_train_table(args.task)
# val_table = dataset.make_val_table(args.task)
# test_table = dataset.make_test_table(args.task)

# Ensure that mini-batch training works ###################################

train_table_input = get_train_table_input(
    train_table=train_table,
    task=dataset.tasks[args.task],
)

train_loader = NodeLoader(
    data,
    node_sampler=sampler,
    input_nodes=train_table_input.nodes,
    input_time=train_table_input.time,
    transform=train_table_input.transform,
    batch_size=args.batch_size,
)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(gnn.parameters()) + list(head.parameters()),
    lr=args.lr,
)

entity_node = train_table_input.nodes[0]
task_type = dataset.tasks[args.task].task_type


def train() -> float:
    loss_accum = 0.0
    count_accum = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        x_dict = encoder(batch.tf_dict)
        x_dict = gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        pred = head(x_dict[entity_node]).squeeze(-1)

        optimizer.zero_grad()
        if task_type == TaskType.BINARY_CLASSIFICATION:
            loss = F.binary_cross_entropy_with_logits(pred, batch[entity_node].y)
        elif task_type == TaskType.REGRESSION:
            loss = F.l1_loss(pred, batch[entity_node].y)
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = F.cross_entropy(pred, batch[entity_node].y)

        loss.backward()
        loss_accum += loss.item() * len(pred)
        count_accum += len(pred)

        optimizer.step()
    return loss_accum / count_accum


for epoch in range(args.epochs):
    print(f"===Epoch {epoch}")
    train_loss = train()
    print("Train loss: ", train_loss)
