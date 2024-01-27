import numpy as np
import sklearn.metrics as skm
from numpy.typing import NDArray

###### classification metrics

### applicable to both binary and multiclass classification


def accuracy(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    if pred.ndim == 1:
        label = pred > 0.5
    else:
        label = pred.argmax(axis=1)
    return skm.accuracy_score(true, label)


def log_loss(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    if pred.ndim == 1 or pred.shape[1] == 1:
        prob = np.sigmoid(pred)
    else:
        prob = np.softmax(pred, axis=1)
    return skm.log_loss(true, prob)


### applicable to binary classification only


def f1(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim == 1 or pred.shape[1] == 1
    label = pred >= 0.5
    return skm.f1_score(true, label, average="binary")


def roc_auc(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim == 1 or pred.shape[1] == 1
    return skm.roc_auc_score(true, pred)


def average_precision(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim == 1 or pred.shape[1] == 1
    return skm.average_precision_score(true, pred)


def auprc(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim == 1 or pred.shape[1] == 1
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    return skm.auc(recall, precision)


### applicable to multiclass classification only


def macro_f1(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim > 1
    label = pred.argmax(axis=1)
    return skm.f1_score(true, label, average="macro")


def micro_f1(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim > 1
    label = pred.argmax(axis=1)
    return skm.f1_score(true, label, average="micro")


###### regression metrics


def mae(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return skm.mean_absolute_error(true, pred)


def mse(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return skm.mean_squared_error(true, pred)


def rmse(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return skm.mean_squared_error(true, pred, squared=False)


def r2(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return skm.r2_score(true, pred)


###### link prediction metrics

def hits_at_k(pred_dict, k_value):
    out, _ = hits_at_k_and_mrr(pred_dict, k_value)
    return out

def mrr(pred_dict, k_value):
    del k_value
    _, out = hits_at_k_and_mrr(pred_dict, -1)
    return out


def hits_at_k_and_mrr(pred_dict, k_value):
        r"""
        compute hist@k and mrr
        reference:
            - https://github.com/snap-stanford/ogb/blob/d5c11d91c9e1c22ed090a2e0bbda3fe357de66e7/ogb/linkproppred/evaluate.py#L214
        
        Parameters:
            pred_dict: keys y_pred_pos, y_pred_neg
            k_value: the desired 'k' value for calculating metric@k
            mrr: whether to calculate mrr or hits@k
        
        Returns:
            the computed performance metric
        """

        y_pred_pos = pred_dict["y_pred_pos"]
        y_pred_neg = pred_dict["y_pred_neg"]

        y_pred_pos = y_pred_pos.reshape(-1, 1)
        optimistic_rank = (y_pred_neg >= y_pred_pos).sum(axis=1)
        pessimistic_rank = (y_pred_neg > y_pred_pos).sum(axis=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hitsK_list = (ranking_list <= k_value).astype(np.float32)
        mrr_list = 1./ranking_list.astype(np.float32)

        return hitsK_list.mean(), mrr_list.mean()
