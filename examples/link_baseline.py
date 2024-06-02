import argparse
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd

from relbench.data import LinkTask, Table
from relbench.datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stack")
parser.add_argument("--task", type=str, default="user-post-comment")
args = parser.parse_args()

dataset = get_dataset(name=args.dataset, process=True)
task: LinkTask = dataset.get_task(args.task, process=True)

train_table = task.train_table
test_table = task.test_table
val_table = task.val_table

trainval_table_df = pd.concat([train_table.df, val_table.df], axis=0)
trainval_table = Table(
    df=trainval_table_df,
    fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
    pkey_col=train_table.pkey_col,
    time_col=train_table.time_col,
)


def past_visit_aggr(x):
    lst_cat = []
    for e in list(x):
        lst_cat.extend(e)
    counter = Counter(lst_cat)
    topk = [elem for elem, _ in counter.most_common(task.eval_k)]
    # padding
    if len(topk) < task.eval_k:
        topk.extend([-1] * (task.eval_k - len(topk)))
    return topk


def evaluate(
    train_table: Table,
    pred_table: Table,
    name: str,
) -> Dict[str, float]:
    is_test = task.dst_entity_col not in pred_table.df
    if name == "past_visit":
        """Predict the most frequently-visited dst nodes per each src node."""
        df = (
            train_table.df.groupby(task.src_entity_col)[task.dst_entity_col]
            .apply(past_visit_aggr)
            .reset_index(name="__pred__")
        )
        pred_ser = pd.merge(pred_table.df, df, how="left", on=task.src_entity_col)[
            "__pred__"
        ]
        # Replace NaN with [-1, -1, ..., -1] prediction
        pred_ser = pred_ser.apply(
            lambda x: x if isinstance(x, list) else [-1] * task.eval_k
        )
        pred = np.stack(pred_ser.values)
    elif name == "global_popularity":
        """Predict the globally most visited dst nodes and predict them across
        the src nodes."""
        lst_cat = []
        for lst in train_table.df[task.dst_entity_col]:
            lst_cat.extend(lst)
        counter = Counter(lst_cat)
        topk = [elem for elem, _ in counter.most_common(task.eval_k)]
        # padding
        if len(topk) < task.eval_k:
            topk.extend([-1] * (task.eval_k - len(topk)))
        pred = np.tile(np.array(topk), (len(pred_table), 1))
    else:
        raise ValueError("Unknown eval name called {name}.")
    return task.evaluate(pred, None if is_test else pred_table)


eval_name_list = ["past_visit", "global_popularity"]
for name in eval_name_list:
    train_metrics = evaluate(train_table, train_table, name=name)
    val_metrics = evaluate(train_table, val_table, name=name)
    test_metrics = evaluate(trainval_table, test_table, name=name)
    print(f"{name}:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")
    print(f"Test: {test_metrics}")
