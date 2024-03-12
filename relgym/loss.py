import numpy as np
from torch.nn import BCEWithLogitsLoss, L1Loss

from relbench.data.task_base import TaskType
from relgym.config import cfg


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
        loss_utils["clamp_min"] = clamp_min
        loss_utils["clamp_max"] = clamp_max
    elif task.task_type == TaskType.LINK_PREDICTION:
        out_channels = cfg.model.channels
        loss_fn = None
        tune_metric = "link_prediction_map"
        higher_is_better = True
    else:
        raise NotImplementedError(task.task_type)

    cfg.model.out_channels = out_channels
    cfg.tune_metric = tune_metric
    cfg.higher_is_better = higher_is_better

    return loss_fn, loss_utils
