from collections import defaultdict
from functools import lru_cache

from ..data import BaseTask
from . import amazon

task_registry = defaultdict(dict)


def register_task(
    dataset_name: str,
    task_name: str,
    cls,
    *args,
    **kwargs,
):
    task_registry[dataset_name][task_name] = (cls, args, kwargs)


def get_task_names(dataset_name: str):
    return list(task_registry[dataset_name].keys())


@lru_cache(maxsize=None)
def get_task(dataset_name: str, task_name: str) -> BaseTask:
    dataset = get_dataset(dataset_name)
    cls, args, kwargs = task_registry[dataset_name][task_name]
    task = cls(dataset, *args, **kwargs)
    return task


register_task("rel-amazon", "user-churn", amazon.UserChurnTask)
register_task("rel-amazon", "user-ltv", amazon.UserLTVTask)
register_task("rel-amazon", "item-churn", amazon.ItemChurnTask)
register_task("rel-amazon", "item-ltv", amazon.ItemLTVTask)
register_task("rel-amazon", "user-item-purchase", amazon.UserItemPurchaseTask)
register_task("rel-amazon", "user-item-rate", amazon.UserItemRateTask)
register_task("rel-amazon", "user-item-review", amazon.UserItemReviewTask)
