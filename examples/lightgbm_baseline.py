import argparse
from typing import Dict

import pandas as pd
import torch
import torch_frame
from inferred_stypes import dataset2inferred_stypes
from text_embedder import GloveTextEmbedding
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.gbdt import LightGBM
from torch_frame.typing import Metric
from torch_frame.utils import infer_df_stype

from relbench.data import RelBenchDataset
from relbench.data.task_base import TaskType
from relbench.datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stackex")
parser.add_argument("--task", type=str, default="rel-stackex-engage")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: remove process=True once correct data/task is uploaded.
dataset: RelBenchDataset = get_dataset(name=args.dataset, process=True)
task = dataset.get_task(args.task, process=True)

train_table = task.train_table
val_table = task.val_table
test_table = task.test_table

dfs: Dict[str, pd.DataFrame] = {}
entity_table = dataset.db.table_dict[task.entity_table]
entity_df = entity_table.df

col_to_stype = dataset2inferred_stypes[args.dataset][task.entity_table]

if entity_table.pkey_col is not None:
    del col_to_stype[entity_table.pkey_col]
for fkey_col in entity_table.fkey_col_to_pkey_table.keys():
    del col_to_stype[fkey_col]

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    col_to_stype[task.target_col] = torch_frame.categorical
elif task.task_type == TaskType.REGRESSION:
    col_to_stype[task.target_col] = torch_frame.numerical

for split, table in [
    ("train", train_table),
    ("val", val_table),
    ("test", test_table),
]:
    dfs[split] = table.df.merge(
        entity_df,
        how="left",
        left_on=list(table.fkey_col_to_pkey_table.keys())[0],
        right_on=entity_table.pkey_col,
    )

train_dataset = Dataset(
    df=dfs["train"],
    col_to_stype=col_to_stype,
    target_col=task.target_col,
    col_to_text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=256,
    ),
).materialize()

tf_train = train_dataset.tensor_frame
tf_val = train_dataset.convert_to_tensor_frame(dfs["val"])
tf_test = train_dataset.convert_to_tensor_frame(dfs["test"])

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    tune_metric = Metric.ROCAUC
else:
    tune_metric = Metric.MAE

model = LightGBM(task_type=train_dataset.task_type, metric=tune_metric)

model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=10)

pred = model.predict(tf_test=tf_train).numpy()
print(f"Train: {task.evaluate(pred, train_table)}")

pred = model.predict(tf_test=tf_val).numpy()
print(f"Val: {task.evaluate(pred, val_table)}")

pred = model.predict(tf_test=tf_test).numpy()
print(f"Test: {task.evaluate(pred)}")
