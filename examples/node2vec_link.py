import argparse
import copy
import os

import torch
from inferred_stypes import dataset2inferred_stypes
from text_embedder import GloveTextEmbedding
from torch.utils.tensorboard import SummaryWriter
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.nn import Node2Vec
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_undirected

from relbench.data import LinkTask, RelBenchDataset
from relbench.data.table import Table
from relbench.data.task_base import TaskType
from relbench.datasets import get_dataset
from relbench.external.graph import make_pkey_fkey_graph

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-trial")
parser.add_argument("--task", type=str, default="rel-trial-sponsor-condition")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--embedding_dim", type=int, default=128)
parser.add_argument("--walk_length", type=int, default=10)
parser.add_argument("--context_size", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--log_dir", type=str, default="results")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(42)

root_dir = "./data"

# TODO: remove process=True once correct data/task is uploaded.
dataset: RelBenchDataset = get_dataset(name=args.dataset, process=True)
task: LinkTask = dataset.get_task(args.task, process=True)
tune_metric = "link_prediction_map"
assert task.task_type == TaskType.LINK_PREDICTION

col_to_stype_dict = dataset2inferred_stypes[args.dataset]

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.db,
    col_to_stype_dict=col_to_stype_dict,
    cache_dir=os.path.join(root_dir, f"{args.dataset}_materialized_cache"),
)


num_src_nodes = task.num_src_nodes
df = task.train_table.df.explode(task.dst_entity_col)
# edge from src to dst
edge_index = torch.stack(
    [
        torch.from_numpy(df[task.src_entity_col].astype(int).values),
        (torch.from_numpy(df[task.dst_entity_col].astype(int).values + num_src_nodes)),
    ]
)


model = Node2Vec(
    edge_index=to_undirected(edge_index),
    embedding_dim=args.embedding_dim,
    walk_length=args.walk_length,
    context_size=args.context_size,
    num_negative_samples=1,
    p=1.0,
    q=1.0,
    sparse=True,
).to(device)

loader = model.loader(
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def get_valid_test_dst_entities():
    dst_entity_table = task.dataset.db.table_dict[task.dst_entity_table]
    # Might be good to filter out entities based on time col in the future
    return dst_entity_table.df[task.dst_entity_col].values + num_src_nodes


@torch.no_grad()
def test(table: Table, k):
    model.eval()
    z = model()
    lhs = z[table.df[task.src_entity_col].values]
    rhs = z[get_valid_test_dst_entities()]
    _, indices = torch.topk(lhs @ rhs.T, k)
    dst_ids = df[task.dst_entity_col].astype(int).values
    mapped_tensor = torch.take(torch.from_numpy(dst_ids), indices)
    return mapped_tensor


writer = SummaryWriter(log_dir=args.log_dir)

state_dict = None
best_val_metric = 0

for epoch in range(1, args.epochs + 1):
    train_loss = train()
    val_pred = test(task.val_table, task.eval_k)
    val_metrics = task.evaluate(val_pred, task.val_table)
    print(
        f"Epoch: {epoch:02d}, Train loss: {train_loss}, " f"Val metrics: {val_metrics}"
    )

    if val_metrics[tune_metric] >= best_val_metric:
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())

    writer.add_scalar("train/loss", train_loss, epoch)
    for name, metric in val_metrics.items():
        writer.add_scalar(f"val/{name}", metric, epoch)

model.load_state_dict(state_dict)
val_pred = test(task.val_table, task.eval_k)
val_metrics = task.evaluate(val_pred, task.val_table)
print(f"Best Val metrics: {val_metrics}")

test_pred = test(task.test_table, task.eval_k)
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")

for name, metric in test_metrics.items():
    writer.add_scalar(f"test/{name}", metric, 0)

writer.flush()
writer.close()
