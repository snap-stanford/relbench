from relgym.config import cfg
from relbench.data.task import TaskType
from torch.nn import BCEWithLogitsLoss, L1Loss
import numpy as np


def create_loss_fn(task):
    loss_utils = {}
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "roc_auc"
        higher_is_better = True
    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = L1Loss()
        tune_metric = "mae"
        higher_is_better = False
        clamp_min, clamp_max = np.percentile(
            task.train_table.df[task.target_col].to_numpy(), [2, 98]
        )
        loss_utils['clamp_min'] = clamp_min
        loss_utils['clamp_max'] = clamp_max
    else:
        raise NotImplementedError(task.task_type)

    cfg.out_channels = out_channels
    cfg.tune_metric = tune_metric
    cfg.higher_is_better = higher_is_better

    return loss_fn, loss_utils

