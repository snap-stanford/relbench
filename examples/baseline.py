import argparse
from typing import Dict

import torch
from torch import Tensor
from torchmetrics import AUROC, AveragePrecision, MeanAbsoluteError

from rtb.data import Table
from rtb.data.task import TaskType
from rtb.datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="relbench-forum")
parser.add_argument("--task", type=str, default="UserContributionTask")
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

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    metrics = {
        "AUROC": AUROC(task="binary").to(device),
        "AP": AveragePrecision(task="binary").to(device),
    }

elif task.task_type == TaskType.REGRESSION:
    metrics = {
        "MAE": MeanAbsoluteError(squared=False).to(device),
    }


def get_metrics(pred: Tensor, target: Tensor) -> Dict[str, float]:
    out_dict: Dict[str, float] = {}
    for metric_name, metric in metrics.items():
        metric.reset()
        metric.update(pred, target)
        out_dict[metric_name] = float(metric.compute())
    return out_dict


def global_zero(train_table: Table, pred_table: Table) -> Dict[str, float]:
    target = pred_table.df[task.target_col].astype(float).values
    target = torch.from_numpy(target)

    pred = torch.zeros_like(target)

    return get_metrics(pred, target)


def global_mean(train_table: Table, pred_table: Table) -> float:
    target = pred_table.df[task.target_col].astype(float).values
    target = torch.from_numpy(target)

    pred = train_table.df[task.target_col].astype(float).values
    pred = torch.from_numpy(pred)
    pred = pred.mean().expand(target.size(0))

    return get_metrics(pred, target)


def global_median(train_table: Table, pred_table: Table) -> float:
    target = pred_table.df[task.target_col].astype(float).values
    target = torch.from_numpy(target)

    pred = train_table.df[task.target_col].astype(float).values
    pred = torch.from_numpy(pred)
    pred = pred.median().expand(target.size(0))

    return get_metrics(pred, target)


def entity_mean(train_table: Table, pred_table: Table) -> float:
    fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
    df = train_table.df.groupby(fkey).agg({task.target_col: "mean"})
    df = pred_table.df.merge(df, how="left", on=fkey)

    target = df[f"{task.target_col}_x"].astype(float).values
    target = torch.from_numpy(target)

    pred = df[f"{task.target_col}_y"].fillna(0).astype(float).values
    pred = torch.from_numpy(pred)

    return get_metrics(pred, target)


def entity_median(train_table: Table, pred_table: Table) -> float:
    fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
    df = train_table.df.groupby(fkey).agg({task.target_col: "median"})
    df = pred_table.df.merge(df, how="left", on=fkey)

    target = df[f"{task.target_col}_x"].astype(float).values
    target = torch.from_numpy(target)

    pred = df[f"{task.target_col}_y"].fillna(0).astype(float).values
    pred = torch.from_numpy(pred)

    return get_metrics(pred, target)


def random(train_table: Table, pred_table: Table) -> float:
    target = pred_table.df[task.target_col].astype(int).values
    target = torch.from_numpy(target)

    pred = torch.rand(target.size())

    return get_metrics(pred, target)


def majority(train_table: Table, pred_table: Table) -> float:
    target = pred_table.df[task.target_col].astype(int).values
    target = torch.from_numpy(target)

    past_target = train_table.df[task.target_col].astype(int).values
    past_target = torch.from_numpy(past_target)

    majority_label = float(past_target.bincount().argmax())
    pred = torch.full((target.numel(),), fill_value=majority_label)

    return get_metrics(pred, target)


if task.task_type == TaskType.REGRESSION:
    train_metrics = global_zero(train_table, train_table)
    val_metrics = global_zero(train_table, val_table)
    print("Global Zero:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

    train_metrics = global_mean(train_table, train_table)
    val_metrics = global_mean(train_table, val_table)
    print("Global Mean:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

    train_metrics = global_median(train_table, train_table)
    val_metrics = global_median(train_table, val_table)
    print("Global Median:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

    train_metrics = entity_mean(train_table, train_table)
    val_metrics = entity_mean(train_table, val_table)
    print("Entity Mean:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

    train_metrics = entity_median(train_table, train_table)
    val_metrics = entity_median(train_table, val_table)
    print("Entity Median:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

elif task.task_type == TaskType.BINARY_CLASSIFICATION:
    train_metrics = random(train_table, train_table)
    val_metrics = random(train_table, val_table)
    print("Random")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")

    train_metrics = majority(train_table, train_table)
    val_metrics = majority(train_table, val_table)
    print("Majority:")
    print(f"Train: {train_metrics}")
    print(f"Val: {val_metrics}")
