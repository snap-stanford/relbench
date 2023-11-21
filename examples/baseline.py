import argparse

import torch
from torchmetrics import AUROC, MeanAbsoluteError

from rtb.data import Table
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

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    metric_name = "AUROC"
    metric = AUROC(task="binary").to(device)
    higher_is_better = True

elif task.task_type == TaskType.REGRESSION:
    metric_name = "MAE"
    metric = MeanAbsoluteError(squared=False).to(device)
    higher_is_better = False


def global_zero(train_table: Table, pred_table: Table) -> float:
    target = pred_table.df[task.target_col].astype(float).values
    target = torch.from_numpy(target)

    pred = torch.zeros_like(target)

    metric.reset()
    metric.update(pred, target)
    return float(metric.compute())


def global_mean(train_table: Table, pred_table: Table) -> float:
    target = pred_table.df[task.target_col].astype(float).values
    target = torch.from_numpy(target)

    pred = train_table.df[task.target_col].astype(float).values
    pred = torch.from_numpy(pred)
    pred = pred.mean().expand(target.size(0))

    metric.reset()
    metric.update(pred, target)
    return float(metric.compute())


def global_median(train_table: Table, pred_table: Table) -> float:
    target = pred_table.df[task.target_col].astype(float).values
    target = torch.from_numpy(target)

    pred = train_table.df[task.target_col].astype(float).values
    pred = torch.from_numpy(pred)
    pred = pred.median().expand(target.size(0))

    metric.reset()
    metric.update(pred, target)
    return float(metric.compute())


def entity_mean(train_table: Table, pred_table: Table) -> float:
    fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
    df = train_table.df.groupby(fkey).agg({task.target_col: "mean"})
    df = pred_table.df.merge(df, how="left", on=fkey)

    target = df[f"{task.target_col}_x"].astype(float).values
    target = torch.from_numpy(target)

    pred = df[f"{task.target_col}_y"].fillna(0).astype(float).values
    pred = torch.from_numpy(pred)

    metric.reset()
    metric.update(pred, target)
    return float(metric.compute())


def entity_median(train_table: Table, pred_table: Table) -> float:
    fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
    df = train_table.df.groupby(fkey).agg({task.target_col: "median"})
    df = pred_table.df.merge(df, how="left", on=fkey)

    target = df[f"{task.target_col}_x"].astype(float).values
    target = torch.from_numpy(target)

    pred = df[f"{task.target_col}_y"].fillna(0).astype(float).values
    pred = torch.from_numpy(pred)

    metric.reset()
    metric.update(pred, target)
    return float(metric.compute())


if task.task_type == TaskType.REGRESSION:
    train_metric = global_zero(train_table, train_table)
    val_metric = global_zero(train_table, val_table)
    print(
        f"Global Zero - "
        f"Train {metric_name}: {train_metric:.4f}, "
        f"Val {metric_name}: {val_metric:.4f}"
    )

    train_metric = global_mean(train_table, train_table)
    val_metric = global_mean(train_table, val_table)
    print(
        f"Global Mean - "
        f"Train {metric_name}: {train_metric:.4f}, "
        f"Val {metric_name}: {val_metric:.4f}"
    )

    train_metric = global_median(train_table, train_table)
    val_metric = global_median(train_table, val_table)
    print(
        f"Global Median - "
        f"Train {metric_name}: {train_metric:.4f}, "
        f"Val {metric_name}: {val_metric:.4f}"
    )

    train_metric = entity_mean(train_table, train_table)
    val_metric = entity_mean(train_table, val_table)
    print(
        f"Entity Mean - "
        f"Train {metric_name}: {train_metric:.4f}, "
        f"Val {metric_name}: {val_metric:.4f}"
    )

    train_metric = entity_median(train_table, train_table)
    val_metric = entity_median(train_table, val_table)
    print(
        f"Entity Median - "
        f"Train {metric_name}: {train_metric:.4f}, "
        f"Val {metric_name}: {val_metric:.4f}"
    )
