import argparse
from typing import Dict

import pandas as pd
import torch
import torch_frame
from text_embedder import GloveTextEmbedding
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.gbdt import XGBoost
from torch_frame.typing import Metric

from rtb.data.task import TaskType
from rtb.datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rtb-forum")
parser.add_argument("--task", type=str, default="UserSumCommentScoresTask")
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

dfs: Dict[str, pd.DataFrame] = {}
if args.dataset in {"rtb-forum", "relbench-forum"}:
    if args.dataset == "rtb-forum":
        col_to_stype = {
            "Reputation": torch_frame.numerical,
            "AboutMe": torch_frame.text_embedded,
            "Age": torch_frame.numerical,
        }
    elif args.dataset == "relbench-forum":
        col_to_stype = {
            "AboutMe": torch_frame.text_embedded,
        }
    user_table = dataset.db.tables["users"]
    user_df = user_table.df[[user_table.pkey_col, *col_to_stype.keys()]]

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        col_to_stype[task.target_col] = torch_frame.categorical
    elif task.task_type == TaskType.REGRESSION:
        col_to_stype[task.target_col] = torch_frame.numerical

    for split, table in [
        ("train", train_table),
        ("val", val_table),
    ]:
        # TODO Feature-engineer from neighboring tables.

        dfs[split] = table.df.merge(
            user_df,
            how="left",
            left_on=list(table.fkey_col_to_pkey_table.keys())[0],
            right_on=user_table.pkey_col,
        )

train_dataset = Dataset(
    df=dfs["train"],
    col_to_stype=col_to_stype,
    target_col=task.target_col,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=256,
    ),
).materialize()

tf_train = train_dataset.tensor_frame
tf_val = dataset.convert_to_tensor_frame(dfs["val"])

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    metric = Metric.ROCAUC
else:
    metric = Metric.MAE

model = XGBoost(task_type=train_dataset.task_type, metric=metric)

model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=20)

pred = model.predict(tf_test=tf_train)
score = model.compute_metric(tf_train.y, pred)
print(f"Train {model.metric}: {score:.4f}")

pred = model.predict(tf_test=tf_val)
score = model.compute_metric(tf_val.y, pred)
print(f"Val {model.metric}: {score:.4f}")
