import argparse
import copy
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from inferred_stypes import dataset2inferred_stypes
from model import Model
from text_embedder import GloveTextEmbedding
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType
from tqdm import tqdm

from relbench.data import LinkTask, RelBenchDataset
from relbench.data.task_base import TaskType
from relbench.datasets import get_dataset
from relbench.external.graph import get_link_train_table_input, make_pkey_fkey_graph
from relbench.external.loader import SparseTensor

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-hm")
parser.add_argument("--task", type=str, default="rel-hm-rec")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=512)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
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
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=os.path.join(root_dir, f"{args.dataset}_materialized_cache"),
)

num_neighbors = [int(args.num_neighbors // 2**i) for i in range(args.num_layers)]

loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
for split, table in [
    ("train", task.train_table),
    ("val", task.val_table),
    ("test", task.test_table),
]:
    table_input = get_link_train_table_input(table, task)
    dst_nodes_dict[split] = table_input.dst_nodes
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=table_input.src_nodes,
        input_time=table_input.src_time,
        subgraph_type="bidirectional",
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=1,
    aggr=args.aggr,
    norm="layer_norm",
    id_awareness=True,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)


def train() -> Dict[str, float]:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)
        out = model.forward_dst_readout(
            batch, task.src_entity_table, task.dst_entity_table
        ).flatten()

        batch_size = batch[task.src_entity_table].batch_size

        # Get ground-truth
        input_id = batch[task.src_entity_table].input_id
        src_batch, dst_index = train_sparse_tensor[input_id]

        # Get target label
        target = torch.isin(
            batch[task.dst_entity_table].batch
            + batch_size * batch[task.dst_entity_table].n_id,
            src_batch + batch_size * dst_index,
        ).float()

        # Optimization
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(out, target)
        loss.backward()

        optimizer.step()

        loss_accum += float(loss) * out.numel()
        count_accum += out.numel()

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list: list[Tensor] = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        out = (
            model.forward_dst_readout(
                batch, task.src_entity_table, task.dst_entity_table
            )
            .detach()
            .flatten()
        )
        batch_size = batch[task.src_entity_table].batch_size
        scores = torch.zeros(batch_size, task.num_dst_nodes, device=out.device)
        scores[
            batch[task.dst_entity_table].batch, batch[task.dst_entity_table].n_id
        ] = torch.sigmoid(out)
        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred


writer = SummaryWriter(log_dir=args.log_dir)

state_dict = None
best_val_metric = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    if epoch % args.eval_epochs_interval == 0:
        val_pred = test(loader_dict["val"])
        val_metrics = task.evaluate(val_pred, task.val_table)
        print(
            f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
            f"Val metrics: {val_metrics}"
        )

        if val_metrics[tune_metric] > best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())

        writer.add_scalar("train/loss", train_loss, epoch)
        for name, metric in val_metrics.items():
            writer.add_scalar(f"val/{name}", metric, epoch)

model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.val_table)
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")

for name, metric in test_metrics.items():
    writer.add_scalar(f"test/{name}", metric, 0)

writer.flush()
writer.close()
