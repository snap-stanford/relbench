from __future__ import annotations

# =====================
# LightGBM code - hacked to work with torch tensor inputs
# =====================
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor


import os
from abc import abstractmethod

from torch_frame import Metric, TaskType, TensorFrame

DEFAULT_METRIC = {
    TaskType.REGRESSION: Metric.RMSE,
    TaskType.BINARY_CLASSIFICATION: Metric.ROCAUC,
    TaskType.MULTICLASS_CLASSIFICATION: Metric.ACCURACY,
}


class GBDT:
    r"""Base class for GBDT (Gradient Boosting Decision Trees) models used as
    strong baseline.

    Args:
        task_type (TaskType): The task type.
        num_classes (int, optional): If the task is multiclass classification,
            an optional num_classes can be used to specify the number of
            classes. Otherwise, we infer the value from the train data.
        metric (Metric, optional): Metric to optimize for, e.g.,
            :obj:`Metric.MAE`. If :obj:`None`, it will default to
            :obj:`Metric.RMSE` for regression, :obj:`Metric.ROCAUC` for binary
            classification, and :obj:`Metric.ACCURACY` for multi-
            class classification. (default: :obj:`None`).
    """
    def __init__(
        self,
        task_type: TaskType,
        num_classes: int | None = None,
        metric: Metric | None = None,
    ):
        self.task_type = task_type
        self._is_fitted: bool = False
        self._num_classes = num_classes

        # Set up metric
        self.metric = DEFAULT_METRIC[task_type]
        if metric is not None:
            if metric.supports_task_type(task_type):
                self.metric = metric
            else:
                raise ValueError(
                    f"{task_type} does not support {metric}. Please choose "
                    f"from {task_type.supported_metrics}.")

    @abstractmethod
    def _tune(self, tf_train: TensorFrame, tf_val: TensorFrame,
              num_trials: int, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, tf_train: TensorFrame) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def _load(self, path: str) -> None:
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        r"""Whether the GBDT is already fitted."""
        return self._is_fitted

    def tune(self, data_train: Any, data_val: Any, num_trials: int,
             *args, **kwargs):
        r"""Fit the model by performing hyperparameter tuning using Optuna. The
        number of trials is specified by num_trials.

        Args:
            tf_train (TensorFrame): The train data in :class:`TensorFrame`.
            tf_val (TensorFrame): The validation data in :class:`TensorFrame`.
            num_trials (int): Number of trials to perform hyper-parameter
                search.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        self._tune(data_train, data_val, num_trials=num_trials, *args, **kwargs)
        self._is_fitted = True

    def predict(self, tf_test: TensorFrame) -> Tensor:
        r"""Predict the labels/values of the test data on the fitted model and
        returns its predictions.

        - :obj:`TaskType.REGRESSION`: Returns raw numerical values.

        - :obj:`TaskType.BINARY_CLASSIFICATION`: Returns the probability of
          being positive.

        - :obj:`TaskType.MULTICLASS_CLASSIFICATION`: Returns the class label
          predictions.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}' is not yet fitted. Please run "
                f"`tune()` first before attempting to predict.")
        pred = self._predict(tf_test)
        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            assert pred.ndim == 2
        else:
            assert pred.ndim == 1
        assert len(pred) == len(tf_test)
        return pred

    def save(self, path: str) -> None:
        r"""Save the model.

        Args:
            path (str): The path to save tuned GBDTs model.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not yet fitted. Please run "
                f"`tune()` first before attempting to save.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        r"""Load the model.

        Args:
            path (str): The path to load tuned GBDTs model.
        """
        self._load(path)
        self._is_fitted = True

    @torch.no_grad()
    def compute_metric(
        self,
        target: Tensor,
        pred: Tensor,
    ) -> float:
        r"""Compute evaluation metric given target labels :obj:`Tensor` and
        pred :obj:`Tensor`. Target contains the target values or labels; pred
        contains the prediction output from calling `predict()` function.

        Returns:
            score (float): Computed metric score.
        """
        if self.metric == Metric.RMSE:
            score = (pred - target).square().mean().sqrt().item()
        elif self.metric == Metric.MAE:
            score = (pred - target).abs().mean().item()
        elif self.metric == Metric.ROCAUC:
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(target.cpu(), pred.cpu())
        elif self.metric == Metric.ACCURACY:
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                pred = pred > 0.5
            total_correct = (target == pred).sum().item()
            test_size = len(target)
            score = total_correct / test_size
        else:
            raise ValueError(f'{self.metric} is not supported.')
        return score
    

class LightGBM(GBDT):
    r"""LightGBM implementation with hyper-parameter tuning using Optuna.

    This implementation extends GBDT and aims to find optimal hyperparameters
    by optimizing the given objective function.
    """

    def _predict_helper(
        self,
        model: Any,
        x: Any,
    ) -> np.ndarray:
        r"""A helper function that applies the lightgbm model on DataFrame
        :obj:`x`.

        Args:
            model (lightgbm.Booster): The lightgbm model.
            x (DataFrame): The input `DataFrame`.

        Returns:
            pred (numpy.ndarray): The prediction output.
        """
        pred = model.predict(x)
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(axis=1)

        return pred

    def objective(
        self,
        trial: Any,  # optuna.trial.Trial
        train_data: Any,  # lightgbm.Dataset
        eval_data: Any,  # lightgbm.Dataset
        num_boost_round: int,
    ) -> float:
        r"""Objective function to be optimized.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            train_data (lightgbm.Dataset): Train data.
            eval_data (lightgbm.Dataset): Validation data.

            num_boost_round (int): Number of boosting round.

        Returns:
            float: Best objective value. Mean absolute error for
            regression task and accuracy for classification task.
        """
        import lightgbm

        self.params = {
            "verbosity":
            -1,
            "bagging_freq":
            1,
            "max_depth":
            trial.suggest_int("max_depth", 3, 11),
            "learning_rate":
            trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves":
            trial.suggest_int("num_leaves", 2, 2**10),
            "subsample":
            trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree":
            trial.suggest_float("colsample_bytree", 0.05, 1.0),
            'lambda_l1':
            trial.suggest_float('lambda_l1', 1e-9, 10.0, log=True),
            'lambda_l2':
            trial.suggest_float('lambda_l2', 1e-9, 10.0, log=True),
            "min_data_in_leaf":
            trial.suggest_int("min_data_in_leaf", 1, 100),
        }

        if self.task_type == TaskType.REGRESSION:
            if self.metric == Metric.RMSE:
                self.params["objective"] = "regression"
                self.params["metric"] = "rmse"
            elif self.metric == Metric.MAE:
                self.params["objective"] = "regression_l1"
                self.params["metric"] = "mae"
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            self.params["objective"] = "binary"
            if self.metric == Metric.ROCAUC:
                self.params["metric"] = "auc"
            elif self.metric == Metric.ACCURACY:
                self.params["metric"] = "binary_error"
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.params["objective"] = "multiclass"
            self.params["metric"] = "multi_error"
            self.params["num_class"] = self._num_classes or len(
                np.unique(train_data.label))
        else:
            raise ValueError(f"{self.__class__.__name__} is not supported for "
                             f"{self.task_type}.")

        boost = lightgbm.train(
            self.params, train_data, num_boost_round=num_boost_round,
            valid_sets=[eval_data],
            callbacks=[
                lightgbm.early_stopping(stopping_rounds=50, verbose=False),
                lightgbm.log_evaluation(period=2000)
            ])
        pred = self._predict_helper(boost, eval_data.data)
        score = self.compute_metric(torch.from_numpy(eval_data.label),
                                    torch.from_numpy(pred))
        return score

    def _tune(
        self,
        train_data: Any,
        val_data: Any,
        num_trials: int,
        num_boost_round=2000,
    ):
        import lightgbm
        import optuna

        if self.task_type == TaskType.REGRESSION:
            study = optuna.create_study(direction="minimize")
        else:
            study = optuna.create_study(direction="maximize")

        train_x, train_y = train_data
        val_x, val_y = val_data
        assert train_y is not None
        assert val_y is not None
        train_data = lightgbm.Dataset(train_x, label=train_y,
                                      free_raw_data=False)
        eval_data = lightgbm.Dataset(val_x, label=val_y, free_raw_data=False)

        study.optimize(
            lambda trial: self.objective(trial, train_data, eval_data,
                                         num_boost_round),
            num_trials)
        self.params.update(study.best_params)

        self.model = lightgbm.train(
            self.params, train_data, num_boost_round=num_boost_round,
            valid_sets=[eval_data],
            callbacks=[
                lightgbm.early_stopping(stopping_rounds=50, verbose=False),
                lightgbm.log_evaluation(period=2000)
            ])

    def _predict(self, test_x: Any) -> Tensor:
        pred = self._predict_helper(self.model, test_x)
        return torch.from_numpy(pred)

    def _load(self, path: str) -> None:
        import lightgbm

        self.model = lightgbm.Booster(model_file=path)


