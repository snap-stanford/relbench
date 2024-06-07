import argparse
import copy

# <<<
import os
from collections import Counter

# >>>
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

# <<<
from torch_geometric.seed import seed_everything

from relbench.data import RelBenchDataset, RelBenchLinkTask, Table
from relbench.datasets import get_dataset
from relbench.external.utils import remove_pkey_fkey

# >>>


LINK_PRED_BASELINE_TARGET_COL_NAME = "link_pred_baseline_target_column_name"
PRED_SCORE_COL_NAME = "pred_score_col_name"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stack")
parser.add_argument("--task", type=str, default="user-post-comment")
# <<<
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--roach_project",
    type=str,
    default=None,
    help="This is for internal use only.",
)
parser.add_argument(
    "--sample_size",
    type=int,
    default=50000,
    help="Subsample the specified number of training data to train lightgbm model.",
)
args = parser.parse_args()

if args.roach_project:
    import roach

    roach.init(args.roach_project)
    roach.store["args"] = args.__dict__

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

root_dir = "./data"

dataset: RelBenchDataset = get_dataset(name=args.dataset, process=False)
# >>>
task: RelBenchLinkTask = dataset.get_task(args.task, process=True)
target_col_name: str = LINK_PRED_BASELINE_TARGET_COL_NAME

train_table = task.train_table
val_table = task.val_table
test_table = task.test_table

# We plan to merge train table with entity and target table to include both
# entity and target table features during lightGBM training.
dfs: Dict[str, pd.DataFrame] = {}
target_dfs: Dict[str, pd.DateOffset] = {}
src_entity_table = dataset.db.table_dict[task.src_entity_table]
src_entity_df = src_entity_table.df
dst_entity_table = dataset.db.table_dict[task.dst_entity_table]
dst_entity_df = dst_entity_table.df

# Prepare col_to_stype dictioanry mapping between column names and stypes
# for torch_frame Dataset initialization.
col_to_stype = {}
src_entity_table_col_to_stype = dataset2inferred_stypes[args.dataset][
    task.src_entity_table
]
dst_entity_table_col_to_stype = dataset2inferred_stypes[args.dataset][
    task.dst_entity_table
]

remove_pkey_fkey(src_entity_table_col_to_stype, src_entity_table)
remove_pkey_fkey(dst_entity_table_col_to_stype, dst_entity_table)

# Rename the column to stype column names appearing in both `src_entity_table`
# and `dst_entity_table` with `_x` and `_y` suffix respectively since they
# will automatically be renamed this way after train/val/test table join with
# both of them in torch frame data preparation.
src_dst_intersection_column_names = set(src_entity_table_col_to_stype.keys()) & set(
    dst_entity_table_col_to_stype.keys()
)
for column_name in src_dst_intersection_column_names:
    src_entity_table_col_to_stype[f"{column_name}_x"] = src_entity_table_col_to_stype[
        column_name
    ]
    del src_entity_table_col_to_stype[column_name]
    dst_entity_table_col_to_stype[f"{column_name}_y"] = dst_entity_table_col_to_stype[
        column_name
    ]
    del dst_entity_table_col_to_stype[column_name]
col_to_stype.update(src_entity_table_col_to_stype)
col_to_stype.update(dst_entity_table_col_to_stype)
col_to_stype[target_col_name] = torch_frame.categorical

# randomly subsample in case training data size is too large.
sampled_train_table = copy.deepcopy(train_table)
if args.sample_size > 0 and args.sample_size < len(sampled_train_table):
    sampled_idx = np.random.permutation(len(sampled_train_table))[: args.sample_size]
    sampled_train_table.df = sampled_train_table.df.iloc[sampled_idx]

# Prepare train/val dataset for lightGBM model training. For each src
# entity, their corresponding dst entities are used as positive label.
# The same number of random dst entities are sampled as negative label.
# lightGBM will train and eval on this binary classification task.
left_entity = list(train_table.fkey_col_to_pkey_table.keys())[0]
right_entity = list(train_table.fkey_col_to_pkey_table.keys())[1]
for split, table in [
    ("train", sampled_train_table),
    ("val", val_table),
]:
    src_entity_df = src_entity_df.astype(
        {src_entity_table.pkey_col: table.df[left_entity].dtype}
    )

    dst_entity_df = dst_entity_df.astype(
        {dst_entity_table.pkey_col: table.df[right_entity].dtype}
    )

    # Left join train table and entity table
    df = table.df.merge(
        src_entity_df,
        how="left",
        left_on=left_entity,
        right_on=src_entity_table.pkey_col,
    )

    # Transform the mapping between one src entity with a list of dst entities
    # to src entity, dst entity pairs
    df = df.explode(right_entity)

    # Add a target col indicating there is a link between src and dst entities
    df[target_col_name] = 1

    # Create a negative sampling df, containing src and dst entities pairs,
    # such that there are no links between them.
    negative_sample_df_columns = list(df.columns)
    negative_sample_df_columns.remove(right_entity)
    negative_samples_df = df[negative_sample_df_columns]
    negative_samples_df[right_entity] = np.random.choice(
        dst_entity_df[dst_entity_table.pkey_col], size=len(negative_samples_df)
    )
    negative_samples_df[target_col_name] = 0

    # Constructing a dataframe containing the same number of positive and
    # negative links and
    df = pd.concat([df, negative_samples_df], ignore_index=True)
    df = pd.merge(
        df,
        dst_entity_df,
        how="left",
        left_on=right_entity,
        right_on=dst_entity_table.pkey_col,
    )
    dfs[split] = df

# Prepare test dataset for lightGBM model evalution
test_df_column_names = list(test_table.df.columns)
test_df_column_names.remove(right_entity)
test_df = test_table.df[test_df_column_names]

# Per each src entity, collect all past linked dst entities
trainval_table_df = pd.concat([train_table.df, val_table.df], axis=0)
trainval_table_df.drop(columns=[train_table.time_col], inplace=True)


def dst_entities_aggr(dst_entities):
    r"concatenate and deduplicate dst entities"
    concatenated = [item for sublist in dst_entities for item in sublist]
    return list(set(concatenated))


grouped_deduped_trainval = (
    trainval_table_df.groupby(left_entity)[right_entity]
    .apply(dst_entities_aggr)
    .reset_index()
)
test_df = pd.merge(test_df, grouped_deduped_trainval, how="left", on=left_entity)

# collect the most popular dst entities
all_dst_entities = [
    entity for sublist in trainval_table_df[right_entity] for entity in sublist
]
dst_entity_counter = Counter(all_dst_entities)
top_dst_entities = [
    entity for entity, _ in dst_entity_counter.most_common(task.eval_k * 2)
]
test_df[right_entity] = test_df[right_entity].apply(
    lambda x: x + top_dst_entities if isinstance(x, list) else top_dst_entities
)

# For each src entity, keep at most `task.eval_k * 2` dst entity candidates
test_df[right_entity] = test_df[right_entity].apply(
    lambda x: (
        x[: task.eval_k * 2] if isinstance(x, list) and len(x) > task.eval_k * 2 else x
    )
)

# Include src and dst entity table features for `test_df`
test_df = pd.merge(
    test_df,
    src_entity_df,
    how="left",
    left_on=left_entity,
    right_on=src_entity_table.pkey_col,
)
test_df = test_df.explode(right_entity)
test_df = pd.merge(
    test_df,
    dst_entity_df,
    how="left",
    left_on=right_entity,
    right_on=dst_entity_table.pkey_col,
)
dfs["test"] = test_df

train_dataset = Dataset(
    df=dfs["train"],
    col_to_stype=col_to_stype,
    target_col=target_col_name,
    col_to_text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
        batch_size=256,
    ),
)
# <<<
train_dataset = train_dataset.materialize(
    path=os.path.join(
        root_dir, f"{args.dataset}_{args.task}_materialized_cache_lightgbm_link.pt"
    )
)
# >>>

tf_train = train_dataset.tensor_frame
tf_val = train_dataset.convert_to_tensor_frame(dfs["val"])
tf_test = train_dataset.convert_to_tensor_frame(dfs["test"])

# tune metric for binary classification problem
tune_metric = Metric.ROCAUC
model = LightGBM(task_type=train_dataset.task_type, metric=tune_metric)
model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=10)


def evaluate(
    lightgbm_output: pd.DataFrame,
    src_entity_name: str,
    dst_entity_name: str,
    timestamp_col_name: str,
    eval_k: int,
    pred_score: float,
    train_table: Table,
    task: RelBenchLinkTask,
) -> Dict[str, float]:
    """Given the input dataframe used for lightGBM binary link classification
    and its output prediction scores and true labels, generate link prediction
    evaluation metrics.

    Args:
        lightgbm_output (pd.DataFrame): The lightGBM input dataframe merged
            with output prediction scores.
        src_entity_name (str): The src entity name.
        dst_entity_name (str): The dst entity name
        timestamp_col (str): The name of the time column.
        eval_k (int): Pre-defined eval k parameter for link pred metric
            evaluation.
        pred_score (float): The binary classification prediction scores.
        train_table (Table): The train table.
        task (RelBenchLinkTask): The task.

    Returns:
        Dict[str, float]: The link pred metrics
    """

    def adjust_past_dst_entities(values):
        if len(values) < eval_k:
            return values + [-1] * (eval_k - len(values))
        else:
            return values[:eval_k]

    grouped_df = (
        lightgbm_output.sort_values(pred_score, ascending=False)
        .groupby([src_entity_name, timestamp_col_name])[dst_entity_name]
        .apply(list)
        .reset_index()
    )
    grouped_df = train_table.df[[src_entity_name, timestamp_col_name]].merge(
        grouped_df, on=[src_entity_name, timestamp_col_name], how="left"
    )
    dst_entity_array = (
        grouped_df[dst_entity_name].apply(adjust_past_dst_entities).tolist()
    )
    dst_entity_array = np.array(dst_entity_array, dtype=int)
    metrics = task.evaluate(dst_entity_array, train_table)
    return metrics


# NOTE: train/val metrics will be artifically high since all true links are
# included in the candidate set
pred = model.predict(tf_test=tf_train).numpy()
lightgbm_output = dfs["train"]
lightgbm_output[PRED_SCORE_COL_NAME] = pred
train_metrics = evaluate(
    lightgbm_output,
    left_entity,
    right_entity,
    task.train_table.time_col,
    task.eval_k,
    PRED_SCORE_COL_NAME,
    sampled_train_table,
    task,
)
print(f"Train: {train_metrics}")

pred = model.predict(tf_test=tf_val).numpy()
lightgbm_output = dfs["val"]
lightgbm_output[PRED_SCORE_COL_NAME] = pred
val_metrics = evaluate(
    lightgbm_output,
    left_entity,
    right_entity,
    task.train_table.time_col,
    task.eval_k,
    PRED_SCORE_COL_NAME,
    val_table,
    task,
)
print(f"Val: {val_metrics}")


pred = model.predict(tf_test=tf_test).numpy()
lightgbm_output = dfs["test"]
lightgbm_output[PRED_SCORE_COL_NAME] = pred
test_metrics = evaluate(
    lightgbm_output,
    left_entity,
    right_entity,
    task.train_table.time_col,
    task.eval_k,
    PRED_SCORE_COL_NAME,
    test_table,
    task,
)
print(f"Test: {test_metrics}")

# <<<
if args.roach_project:
    roach.store["val"] = val_metrics
    roach.store["test"] = test_metrics
    roach.finish()
# >>>
