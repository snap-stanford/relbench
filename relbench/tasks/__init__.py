import json
import pkgutil
from collections import defaultdict
from functools import lru_cache
from typing import List

import pooch

from relbench.base import AutoCompleteTask, BaseTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import amazon, avito, event, f1, hm, stack, trial

task_registry = defaultdict(dict)

hashes_str = pkgutil.get_data(__name__, "hashes.json")
hashes = json.loads(hashes_str)

DOWNLOAD_REGISTRY = pooch.create(
    path=pooch.os_cache("relbench"),
    base_url="https://relbench.stanford.edu/download/",
    registry=hashes,
)


def register_task(
    dataset_name: str,
    task_name: str,
    cls: BaseTask,
    *args,
    **kwargs,
) -> None:
    r"""Register an instantiation of a :class:`BaseTask` subclass with the given name.

    Args:
        dataset_name: The name of the dataset.
        task_name: The name of the task.
        cls: The class of the task.
        args: The arguments to instantiate the task.
        kwargs: The keyword arguments to instantiate the task.

    The name is used to enable caching and downloading functionalities.
    `cache_dir` is added to kwargs by default. If you want to override it, you
    can pass `cache_dir` as a keyword argument in `kwargs`.
    """

    cache_dir = f"{pooch.os_cache('relbench')}/{dataset_name}/tasks/{task_name}"
    kwargs = {"cache_dir": cache_dir, **kwargs}
    task_registry[dataset_name][task_name] = (cls, args, kwargs)


def get_task_names(dataset_name: str) -> List[str]:
    r"""Return a list of names of the registered tasks for the given dataset."""
    return list(task_registry[dataset_name].keys())


def download_task(dataset_name: str, task_name: str) -> None:
    r"""Download task from RelBench server into its cache directory.

    The downloaded task tables will be automatically picked up by the task object, when
    `task.get_table(split)` is called.
    """

    DOWNLOAD_REGISTRY.fetch(
        f"{dataset_name}/tasks/{task_name}.zip",
        processor=pooch.Unzip(extract_dir="."),
        progressbar=True,
    )


@lru_cache(maxsize=None)
def get_task(dataset_name: str, task_name: str, download=False) -> BaseTask:
    r"""Return a task object by name.

    Args:
        dataset_name: The name of the dataset.
        task_name: The name of the task.
        download: If True, download the task from the RelBench server.

    Returns:
        BaseTask: The task object.

    If `download` is True, the task tables (train, val, test) comprising the
    task will be downloaded into the cache from the RelBench server. If you use
    `download=False` the first time, the task tables will be computed from
    scratch using the database.

    Once the task tables are cached, either because of download or computing from
    scratch, the cache will be used. `download=True` will verify that the
    cached task tables matches the RelBench version even in this case.
    """

    if download:
        download_task(dataset_name, task_name)
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
register_task(
    "rel-amazon",
    "review-rating",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="review",
    target_col="rating",
    remove_columns=[
        ("review", "review_text"),
        ("review", "summary"),
    ],
)
register_task(
    "rel-amazon",
    "product-price",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="product",
    target_col="price",
    remove_columns=[
        ("product", "brand"),
        ("product", "title"),
        ("product", "description"),
    ],
)

register_task("rel-avito", "ad-ctr", avito.AdCTRTask)
register_task("rel-avito", "user-visits", avito.UserVisitsTask)
register_task("rel-avito", "user-clicks", avito.UserClicksTask)
register_task("rel-avito", "user-ad-visit", avito.UserAdVisitTask)
register_task(
    "rel-avito",
    "searchstream-click",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="SearchStream",
    target_col="IsClick",
)
register_task(
    "rel-avito",
    "searchinfo-isuserloggedon",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="SearchInfo",
    target_col="IsUserLoggedOn",
)

register_task("rel-event", "user-attendance", event.UserAttendanceTask)
register_task("rel-event", "user-repeat", event.UserRepeatTask)
register_task("rel-event", "user-ignore", event.UserIgnoreTask)
register_task(
    "rel-event",
    "event_interest-iterested",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="event_interest",
    target_col="interested",
    remove_columns=[
        ("event_interest", "not_interested"),
    ],
)

register_task("rel-f1", "driver-position", f1.DriverPositionTask)
register_task("rel-f1", "driver-dnf", f1.DriverDNFTask)
register_task("rel-f1", "driver-top3", f1.DriverTop3Task)
register_task(
    "rel-f1",
    "results-position",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="results",
    target_col="position",
    remove_columns=[
        ("results", "statusId"),
        ("results", "positionOrder"),
        ("results", "points"),
        ("results", "laps"),
        ("results", "milliseconds"),
        ("results", "fastestLap"),
        ("results", "rank"),
    ],
)
register_task(
    "rel-f1",
    "qualifying-position",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="qualifying",
    target_col="position",
    remove_columns=[],
)


register_task("rel-hm", "user-item-purchase", hm.UserItemPurchaseTask)
register_task("rel-hm", "user-churn", hm.UserChurnTask)
register_task("rel-hm", "item-sales", hm.ItemSalesTask)

register_task("rel-stack", "user-engagement", stack.UserEngagementTask)
register_task("rel-stack", "post-votes", stack.PostVotesTask)
register_task("rel-stack", "user-badge", stack.UserBadgeTask)
register_task("rel-stack", "user-post-comment", stack.UserPostCommentTask)
register_task("rel-stack", "post-post-related", stack.PostPostRelatedTask)
register_task(
    "rel-stack",
    "badges-class",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="badges",
    target_col="Class",
    remove_columns=[("badges", "TagBased"), ("badges", "Name")],
)

register_task("rel-trial", "study-outcome", trial.StudyOutcomeTask)
register_task("rel-trial", "study-adverse", trial.StudyAdverseTask)
register_task("rel-trial", "site-success", trial.SiteSuccessTask)
register_task("rel-trial", "condition-sponsor-run", trial.ConditionSponsorRunTask)
register_task("rel-trial", "site-sponsor-run", trial.SiteSponsorRunTask)

register_task(
    "rel-salt",
    "item-plant",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocumentitem",
    target_col="PLANT",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "item-shippoint",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocumentitem",
    target_col="SHIPPINGPOINT",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "item-incoterms",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocumentitem",
    target_col="ITEMINCOTERMSCLASSIFICATION",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
    ],
)
register_task(
    "rel-salt",
    "sales-office",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocument",
    target_col="SALESOFFICE",
    remove_columns=[
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "sales-group",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocument",
    target_col="SALESGROUP",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "sales-payterms",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocument",
    target_col="CUSTOMERPAYMENTTERMS",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "sales-shipcond",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocument",
    target_col="SHIPPINGCONDITION",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "sales-incoterms",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocument",
    target_col="HEADERINCOTERMSCLASSIFICATION",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
