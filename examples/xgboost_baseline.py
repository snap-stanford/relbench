import argparse
from typing import Dict

import torch
import torch_frame
from rtb.data.task import TaskType
from rtb.datasets import get_dataset
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.gbdt import XGBoost
from torch_frame.typing import Metric

from text_embedder import GloveTextEmbedding

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rtb-forum")
parser.add_argument("--task", type=str, default="UserSumCommentScoresTask")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = get_dataset(name=args.dataset, root="./data")
if args.task not in dataset.tasks:
    raise ValueError(
        f"'{args.dataset}' does not support the given task {args.task}. "
        f"Please choose the task from {list(dataset.tasks.keys())}.")

task = dataset.tasks[args.task]
train_table = dataset.make_train_table(args.task)
val_table = dataset.make_val_table(args.task)
test_table = dataset.make_test_table(args.task)

if args.dataset == "rtb-forum":
    col_to_stype = {
        'Reputation': torch_frame.numerical,
        'AboutMe': torch_frame.text_embedded,
        'Age': torch_frame.numerical,
    }
    user_df = dataset.db.tables["users"].df[['Id', *col_to_stype.keys()]]

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        col_to_stype[task.target_col] = torch_frame.categorical
    elif task.task_type == TaskType.REGRESSION:
        col_to_stype[task.target_col] = torch_frame.numerical

    datasets: Dict[str, Dataset] = {}
    for split, table in [
        ("train", train_table),
        ("val", val_table),
    ]:
        # TODO Feature-engineer from neighboring tables.

        df = table.df.merge(
            user_df,
            how='left',
            left_on='UserId',
            right_on='Id',
        )

        # TODO This usage is incorrect, since we are computing separate stats.
        datasets[split] = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            target_col=task.target_col,
            text_embedder_cfg=TextEmbedderConfig(
                text_embedder=GloveTextEmbedding(device=device),
                batch_size=256,
            ),
        ).materialize()

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    metric = Metric.ROCAUC
else:
    metric = Metric.MAE

model = XGBoost(task_type=datasets['train'].task_type, metric=metric)

model.tune(
    tf_train=datasets['train'].tensor_frame,
    tf_val=datasets['val'].tensor_frame,
    num_trials=20,
)

pred = model.predict(tf_test=datasets['val'].tensor_frame)
score = model.compute_metric(datasets['val'].tensor_frame.y, pred)
print(f"{model.metric} : {score}")
