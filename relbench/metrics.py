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
