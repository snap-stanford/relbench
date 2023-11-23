import numpy as np
import sklearn.metrics as skm

###### classification metrics

### applicable to both binary and multiclass classification


def accuracy(true, pred):
    if pred.ndim == 1:
        label = pred > 0.5
    else:
        label = pred.argmax(axis=1)
    return skm.accuracy_score(true, label)


def log_loss(true, pred):
    if pred.ndim == 1:
        prob = np.sigmoid(pred)
    else:
        prob = np.softmax(pred, axis=1)
    return skm.log_loss(true, prob)


### applicable to binary classification only


def f1(true, pred):
    assert pred.ndim == 1
    label = pred.argmax(axis=1)
    return skm.f1_score(true, label, average="binary")


def roc_auc(true, pred):
    assert pred.ndim == 1
    return skm.roc_auc_score(true, pred)


### applicable to multiclass classification only


def macro_f1(true, pred):
    assert pred.ndim > 1
    label = pred.argmax(axis=1)
    return skm.f1_score(true, label, average="macro")


def micro_f1(true, pred):
    assert pred.ndim > 1
    label = pred.argmax(axis=1)
    return skm.f1_score(true, label, average="micro")


###### regression metrics


def mae(true, pred):
    return skm.mean_absolute_error(true, pred)


def mse(true, pred):
    return skm.mean_squared_error(true, pred)


def rmse(true, pred):
    return skm.mean_squared_error(true, pred, squared=False)


def r2(true, pred):
    return skm.r2_score(true, pred)
