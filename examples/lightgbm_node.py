import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch_frame
from text_embedder import GloveTextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.gbdt import LightGBM
from torch_frame.typing import Metric
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey
from relbench.tasks import get_task

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stack")
parser.add_argument("--task", type=str, default="user-engage")
parser.add_argument("--num_trials", type=int, default=10)
parser.add_argument(
    "--sample_size",
    type=int,
    default=50_000,
    help="Subsample the specified number of training data to train lightgbm model.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")


dfs: Dict[str, pd.DataFrame] = {}
entity_table = dataset.get_db().table_dict[task.entity_table]
entity_df = entity_table.df

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

col_to_stype = col_to_stype_dict[task.entity_table]
remove_pkey_fkey(col_to_stype, entity_table)

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    col_to_stype[task.target_col] = torch_frame.categorical
elif task.task_type == TaskType.REGRESSION:
    col_to_stype[task.target_col] = torch_frame.numerical
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    col_to_stype[task.target_col] = torch_frame.embedding
else:
    raise ValueError(f"Unsupported task type called {task.task_type}")

# randomly subsample in case training data size is too large.
if args.sample_size > 0 and args.sample_size < len(train_table):
    sampled_idx = np.random.permutation(len(train_table))[: args.sample_size]
    train_table.df = train_table.df.iloc[sampled_idx]

for split, table in [
    ("train", train_table),
    ("val", val_table),
    ("test", test_table),
]:
    left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
    entity_df = entity_df.astype({entity_table.pkey_col: table.df[left_entity].dtype})
    dfs[split] = table.df.merge(
        entity_df,
        how="left",
        left_on=left_entity,
        right_on=entity_table.pkey_col,
    )

train_dataset = torch_frame.data.Dataset(
    df=dfs["train"],
    col_to_stype=col_to_stype,
    target_col=task.target_col,
    col_to_text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=256,
    ),
)
path = Path(
    f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/materialized/node_train.pt"
)
path.parent.mkdir(parents=True, exist_ok=True)
train_dataset = train_dataset.materialize(path=path)

tf_train = train_dataset.tensor_frame
tf_val = train_dataset.convert_to_tensor_frame(dfs["val"])
tf_test = train_dataset.convert_to_tensor_frame(dfs["test"])

if task.task_type in [
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
]:
    tune_metric = Metric.ROCAUC
elif task.task_type == TaskType.REGRESSION:
    tune_metric = Metric.MAE
else:
    raise ValueError(f"Task task type is unsupported {task.task_type}")

if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION]:
    model = LightGBM(task_type=train_dataset.task_type, metric=tune_metric)
    model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=args.num_trials)

    pred = model.predict(tf_test=tf_train).numpy()
    train_metrics = task.evaluate(pred, train_table)

    pred = model.predict(tf_test=tf_val).numpy()
    val_metrics = task.evaluate(pred, val_table)

    pred = model.predict(tf_test=tf_test).numpy()
    test_metrics = task.evaluate(pred)

elif TaskType.MULTILABEL_CLASSIFICATION:
    y_train = tf_train.y.values.to(torch.long)
    y_val = tf_val.y.values.to(torch.long)
    pred_train_list = []
    pred_val_list = []
    pred_test_list = []
    # Per-label evaluation
    for i in tqdm(range(task.num_labels)):
        model = LightGBM(
            task_type=torch_frame.TaskType.BINARY_CLASSIFICATION, metric=tune_metric
        )
        tf_train.y = y_train[:, i]
        tf_val.y = y_val[:, i]
        model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=10)
        pred_train_list.append(model.predict(tf_test=tf_train).numpy())
        pred_val_list.append(model.predict(tf_test=tf_val).numpy())
        pred_test_list.append(model.predict(tf_test=tf_test).numpy())

    pred_train = np.stack(pred_train_list).transpose()
    train_metrics = task.evaluate(pred_train, train_table)

    pred_val = np.stack(pred_val_list).transpose()
    val_metrics = task.evaluate(pred_val, val_table)

    pred_test = np.stack(pred_test_list).transpose()
    test_metrics = task.evaluate(pred_test)

print(f"Train: {train_metrics}")
print(f"Val: {val_metrics}")
print(f"Test: {test_metrics}")
