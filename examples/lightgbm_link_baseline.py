import argparse
from collections import Counter
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

from relbench.data import LinkTask, RelBenchDataset, RelBenchLinkTask, Table
from relbench.datasets import get_dataset

LINK_PRED_BASELINE_TARGET_COL_NAME = "link_pred_baseline_target_column_name"
PRED_SCORE_COL_NAME = "pred_score_col_name"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stackex")
parser.add_argument("--task", type=str, default="rel-stackex-comment-on-post")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset: RelBenchDataset = get_dataset(name=args.dataset, process=True)
task: RelBenchLinkTask = dataset.get_task(args.task, process=True)
target_col_name: str = LINK_PRED_BASELINE_TARGET_COL_NAME

train_table = task.train_table
val_table = task.val_table
test_table = task.test_table

# We plan to merge train table with entity and target table to include both
# entity and target table features during lightGBM training.
dfs: Dict[str, pd.DataFrame] = {}
target_dfs: Dict[str, pd.DateOffset] = {}
lhs_entity_table = dataset.db.table_dict[task.src_entity_table]
lhs_entity_df = lhs_entity_table.df
rhs_entity_table = dataset.db.table_dict[task.dst_entity_table]
rhs_entity_df = rhs_entity_table.df

# Prepare col_to_stype dictioanry mapping between column names and stypes
# for torch_frame Dataset initialization.
col_to_stype = {}
lhs_entity_table_col_to_stype = dataset2inferred_stypes[args.dataset][
    task.src_entity_table
]
rhs_entity_table_col_to_stype = dataset2inferred_stypes[args.dataset][
    task.dst_entity_table
]

if lhs_entity_table.pkey_col is not None:
    del lhs_entity_table_col_to_stype[lhs_entity_table.pkey_col]
for fkey_col in lhs_entity_table.fkey_col_to_pkey_table.keys():
    del lhs_entity_table_col_to_stype[fkey_col]
if rhs_entity_table.pkey_col is not None:
    del rhs_entity_table_col_to_stype[rhs_entity_table.pkey_col]
for fkey_col in rhs_entity_table.fkey_col_to_pkey_table.keys():
    del rhs_entity_table_col_to_stype[fkey_col]

lhs_rhs_intersection_column_names = set(lhs_entity_table_col_to_stype.keys()) & set(
    rhs_entity_table_col_to_stype.keys()
)
for column_name in lhs_rhs_intersection_column_names:
    lhs_entity_table_col_to_stype[f"{column_name}_x"] = lhs_entity_table_col_to_stype[
        column_name
    ]
    del lhs_entity_table_col_to_stype[column_name]
    rhs_entity_table_col_to_stype[f"{column_name}_y"] = rhs_entity_table_col_to_stype[
        column_name
    ]
    del rhs_entity_table_col_to_stype[column_name]
col_to_stype.update(lhs_entity_table_col_to_stype)
col_to_stype.update(rhs_entity_table_col_to_stype)
col_to_stype[target_col_name] = torch_frame.categorical

# Prepare train/val dataset for lightGBM model training. For each lhs
# entity, their corresponding rhs entities are used as positive label.
# The same number of random rhs entities are sampled as negative label.
# lightGBM will train and eval on this binary classification task.
left_entity = list(train_table.fkey_col_to_pkey_table.keys())[0]
right_entity = list(train_table.fkey_col_to_pkey_table.keys())[1]
for split, table in [
    ("train", train_table),
    ("val", val_table),
]:
    lhs_entity_df = lhs_entity_df.astype(
        {lhs_entity_table.pkey_col: table.df[left_entity].dtype}
    )

    rhs_entity_df = rhs_entity_df.astype(
        {rhs_entity_table.pkey_col: table.df[right_entity].dtype}
    )

    # Left join train table and entity table
    df = table.df.merge(
        lhs_entity_df,
        how="left",
        left_on=left_entity,
        right_on=lhs_entity_table.pkey_col,
    )

    # Transform the mapping between one lhs entity with a list of rhs entities
    # to lhs entity, rhs entity pairs
    df = df.explode(right_entity)

    # Add a target col indicating there is a link between lhs and rhs entities
    df[target_col_name] = 1

    # Create a negative sampling df, containing lhs and rhs entities pairs,
    # such that there are no links between them.
    negative_sample_df_columns = list(df.columns)
    negative_sample_df_columns.remove(right_entity)
    negative_samples_df = df[negative_sample_df_columns]
    negative_samples_df[right_entity] = np.random.choice(
        rhs_entity_df[rhs_entity_table.pkey_col], size=len(negative_samples_df)
    )
    negative_samples_df[target_col_name] = 0

    # Constructing a dataframe containing the same number of positive and
    # negative links and
    df = pd.concat([df, negative_samples_df], ignore_index=True)
    df = pd.merge(
        df,
        rhs_entity_df,
        how="left",
        left_on=right_entity,
        right_on=rhs_entity_table.pkey_col,
    )
    dfs[split] = df

# Prepare test dataset for lightGBM model evalution
test_df_column_names = list(test_table.df.columns)
test_df_column_names.remove(right_entity)
test_df = test_table.df[test_df_column_names]

# Per each lhs entity, collect all past linked rhs entities
trainval_table_df = pd.concat([train_table.df, val_table.df], axis=0)
trainval_table_df.drop(columns=[train_table.time_col], inplace=True)


def rhs_entities_aggr(rhs_entities):
    r"concatenate and deduplicate rhs entities"
    concatenated = [item for sublist in rhs_entities for item in sublist]
    return list(set(concatenated))


grouped_deduped_trainval = (
    trainval_table_df.groupby(left_entity)[right_entity]
    .apply(rhs_entities_aggr)
    .reset_index()
)
test_df = pd.merge(test_df, grouped_deduped_trainval, how="left", on=left_entity)

# collect the most popular rhs entities
all_rhs_entities = [
    entity for sublist in trainval_table_df[right_entity] for entity in sublist
]
rhs_entity_counter = Counter(all_rhs_entities)
top_rhs_entities = [
    entity for entity, _ in rhs_entity_counter.most_common(task.eval_k * 2)
]
test_df[right_entity] = test_df[right_entity].apply(
    lambda x: x + top_rhs_entities if isinstance(x, list) else top_rhs_entities
)

# For each lhs entity, keep at most `task.eval_k * 2` rhs entity candidates
test_df[right_entity] = test_df[right_entity].apply(
    lambda x: (
        x[: task.eval_k * 2] if isinstance(x, list) and len(x) > task.eval_k * 2 else x
    )
)

# Include lhs and rhs entity table features for `test_df`
test_df = pd.merge(
    test_df,
    lhs_entity_df,
    how="left",
    left_on=left_entity,
    right_on=lhs_entity_table.pkey_col,
)
test_df = test_df.explode(right_entity)
test_df = pd.merge(
    test_df,
    rhs_entity_df,
    how="left",
    left_on=right_entity,
    right_on=rhs_entity_table.pkey_col,
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
).materialize()
tf_train = train_dataset.tensor_frame
tf_val = train_dataset.convert_to_tensor_frame(dfs["val"])
tf_test = train_dataset.convert_to_tensor_frame(dfs["test"])

# tune metric for binary classification problem
tune_metric = Metric.ROCAUC
model = LightGBM(task_type=train_dataset.task_type, metric=tune_metric)
model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=10)


def evaluate(
    lightgbm_output: pd.DataFrame,
    lhs_entity_name: str,
    rhs_entity_name: str,
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
        lhs_entity_name (str): The lhs entity name.
        rhs_entity_name (str): The rhs entity name
        timestamp_col (str): The name of the time column.
        eval_k (int): Pre-defined eval k parameter for link pred metric
            evaluation.
        pred_score (float): The binary classification prediction scores.
        train_table (Table): The train table.
        task (RelBenchLinkTask): The task.

    Returns:
        Dict[str, float]: The link pred metrics
    """

    def adjust_past_rhs_entities(values):
        if len(values) < eval_k:
            return values + [-1] * (eval_k - len(values))
        else:
            return values[:eval_k]

    grouped_df = (
        lightgbm_output.sort_values(pred_score, ascending=False)
        .groupby([lhs_entity_name, timestamp_col_name])[rhs_entity_name]
        .apply(list)
        .reset_index()
    )
    grouped_df = train_table.df[[lhs_entity_name, timestamp_col_name]].merge(
        grouped_df, on=[lhs_entity_name, timestamp_col_name], how="left"
    )
    rhs_entity_array = (
        grouped_df[rhs_entity_name].apply(adjust_past_rhs_entities).tolist()
    )
    rhs_entity_array = np.array(rhs_entity_array, dtype=int)
    metrics = task.evaluate(rhs_entity_array, train_table)
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
    train_table,
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
