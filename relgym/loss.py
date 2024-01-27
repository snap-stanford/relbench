from relgym.config import cfg
from relbench.data.task import TaskType
from torch.nn import BCEWithLogitsLoss, L1Loss


def create_loss_fn(task):
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
    else:
        raise NotImplementedError(task.task_type)

    cfg.out_channels = out_channels
    cfg.tune_metric = tune_metric
    cfg.higher_is_better = higher_is_better

    return loss_fn

