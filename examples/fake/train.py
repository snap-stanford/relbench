import argparse

import torch
import torch.nn.functional as F
from torch_geometric.loader import NodeLoader
from torch_geometric.nn import MLP
from torch_geometric.sampler import NeighborSampler

from rtb.datasets import FakeProductDataset
from rtb.external.graph import get_train_table_input, make_pkey_fkey_graph
from rtb.external.nn import HeteroEncoder, HeteroGraphSAGE

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="churn", choices=["churn", "ltv"])
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset and create training labels #####################################

dataset = FakeProductDataset(root="/tmp/rtb-data", process=True)
train_table = dataset.make_train_table(args.task)
val_table = dataset.make_val_table(args.task)
test_table = dataset.make_test_table(args.task)

# Convert the dataset into a PyG heterogeneous graphs #########################

data = make_pkey_fkey_graph(
    dataset.db,
    dataset.get_stype_proposal(),
)

# Create data loaders (while sharing the sampler) #############################

sampler = NeighborSampler(
    data,
    num_neighbors=[-1, -1],
    time_attr="time",
)

loader_kwargs = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    persistent_workers=args.num_workers > 0,
    node_sampler=sampler,
)
train_table_input = get_train_table_input(train_table, "churn", torch.float)
train_loader = NodeLoader(
    data,
    input_nodes=train_table_input.nodes,
    input_time=train_table_input.time,
    transform=train_table_input.transform,
    **loader_kwargs,
)
val_table_input = get_train_table_input(val_table, "churn", torch.float)
val_loader = NodeLoader(
    data,
    input_nodes=val_table_input.nodes,
    input_time=val_table_input.time,
    transform=val_table_input.transform,
    **loader_kwargs,
)
test_table_input = get_train_table_input(test_table)
test_loader = NodeLoader(
    data,
    input_nodes=test_table_input.nodes,
    input_time=test_table_input.time,
    **loader_kwargs,
)

# Create model ################################################################

col_names_dict = {
    node_type: data[node_type].tf.col_names_dict for node_type in data.node_types
}
encoder = HeteroEncoder(
    args.channels,
    col_names_dict,
    data.col_stats_dict,
).to(device)
gnn = HeteroGraphSAGE(
    data.node_types,
    data.edge_types,
    args.channels,
    num_layers=2,
).to(device)
head = MLP(
    args.channels,
    out_channels=1,
    num_layers=1,
).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(gnn.parameters()) + list(head.parameters()),
    lr=0.01,
)

# Training and evaluation loops ###############################################


def train() -> float:
    encoder.train()
    gnn.train()
    head.train()

    total_loss = total_examples = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        x_dict = encoder(batch.tf_dict)
        x_dict = gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        x = head(x_dict[train_table_input.nodes[0]]).squeeze(-1)
        y = batch[train_table_input.nodes[0]].y

        if args.task == "churn":
            loss = F.binary_cross_entropy_with_logits(x, y)
        else:
            loss = F.mse_loss(x, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.size(0)
        total_examples += y.size(0)

    return total_loss / total_examples


for epoch in range(1, args.epochs + 1):
    loss = train()
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")
