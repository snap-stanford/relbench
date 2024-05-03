import argparse
from typing import Dict

import numpy as np
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
from tqdm import tqdm

from relbench.data import RelBenchDataset, RelBenchNodeTask
from relbench.data.task_base import TaskType
from relbench.datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stackex")
parser.add_argument("--task", type=str, default="rel-stackex-engage")
# Use auto-regressive label as hand-crafted feature as input to LightGBM
parser.add_argument("--use_ar_label", action="store_true")
parser.add_argument(
    "--sample_size",
    type=int,
    default=50000,
    help="Subsample the specified number of training data to train lightgbm model.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: remove process=True once correct data/task is uploaded.
dataset: RelBenchDataset = get_dataset(name=args.dataset, process=True)
task: RelBenchNodeTask = dataset.get_task(args.task, process=True)

train_table = task.train_table
val_table = task.val_table
test_table = task.test_table

ar_label_cols = []

if args.use_ar_label:
    ### Adding AR labels into train/val/test_table
    whole_df = pd.concat([train_table.df, val_table.df, test_table.df], axis=0)
    num_ar_labels = max(train_table.df[train_table.time_col].nunique() - 2, 1)

    sorted_unique_times = np.sort(whole_df[train_table.time_col].unique())
    timedelta = sorted_unique_times[1:] - sorted_unique_times[:-1]
    if (timedelta / timedelta[0] - 1).max() > 0.1:
        raise RuntimeError(
            "Timestamps are not equally spaced, making it inappropriate for "
            "AR labels to be used."
        )
    TIME_IDX_COL = "time_idx"
    time_df = pd.DataFrame(
        {
            task.time_col: sorted_unique_times,
            "time_idx": np.arange(len(sorted_unique_times)),
        }
    )

    whole_df = whole_df.merge(time_df, how="left", on=task.time_col)
    whole_df.drop(task.time_col, axis=1, inplace=True)
    # Shift timestamp of whole_df iteratively and join it with train/val/test_table
    for i in range(1, num_ar_labels + 1):
        whole_df_shifted = whole_df.copy(deep=True)
        # Shift time index by i
        whole_df_shifted[TIME_IDX_COL] += i
        # Map time index back to datetime timestamp
        whole_df_shifted = whole_df_shifted.merge(time_df, how="inner", on=TIME_IDX_COL)
        whole_df_shifted.drop(TIME_IDX_COL, axis=1, inplace=True)
        ar_label = f"AR_{i}"
        ar_label_cols.append(ar_label)
        whole_df_shifted.rename(columns={task.target_col: ar_label}, inplace=True)

        for table in [train_table, val_table, test_table]:
            table.df = table.df.merge(
                whole_df_shifted, how="left", on=(task.entity_col, task.time_col)
            )

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
    for ar_label in ar_label_cols:
        col_to_stype[ar_label] = torch_frame.categorical
elif task.task_type == TaskType.REGRESSION:
    col_to_stype[task.target_col] = torch_frame.numerical
    for ar_label in ar_label_cols:
        col_to_stype[ar_label] = torch_frame.numerical
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    col_to_stype[task.target_col] = torch_frame.embedding
    for ar_label in ar_label_cols:
        col_to_stype[ar_label] = torch_frame.embedding
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

if task.task_type in [
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
]:
    tune_metric = Metric.ROCAUC
elif task.task_type == TaskType.REGRESSION:
    tune_metric = Metric.MAE

if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION]:
    model = LightGBM(task_type=train_dataset.task_type, metric=tune_metric)
    model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=10)

    pred = model.predict(tf_test=tf_train).numpy()
    print(f"Train: {task.evaluate(pred, train_table)}")

    pred = model.predict(tf_test=tf_val).numpy()
    print(f"Val: {task.evaluate(pred, val_table)}")

    pred = model.predict(tf_test=tf_test).numpy()
    print(f"Test: {task.evaluate(pred)}")
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
    print(f"Train: {task.evaluate(pred_train, train_table)}")
    pred_val = np.stack(pred_val_list).transpose()
    print(f"Val: {task.evaluate(pred_val, val_table)}")
    pred_test = np.stack(pred_test_list).transpose()
    print(f"Test: {task.evaluate(pred_test)}")
