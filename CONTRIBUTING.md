# Contributing to RelBench

We welcome and appreciate contributions to RelBench from the community. Please reach out if you have something in mind. We expect our handling of contributions to become more streamlined as the project matures.


## Issues, bug fixes, discussions

Please submit GitHub issues and open Pull Requests if you find some bug or other issue in RelBench.


## Contributing datasets

While the RelBench team maintains the core datasets which form the official RelBench benchmark, we envision RelBench to also serve as a repository of datasets with contributions from the community. To work with your own datasets in RelBench, please take a look at the tutorial [tutorials/custom_dataset.ipynb](tutorials/custom_dataset.ipynb), also available on [Google Colab](https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/custom_dataset.ipynb). Once you have a working dataset within RelBench, make a PR to get it added to RelBench. To help RelBench maintainers, please follow these conventions:

1. The dataset name is a single word (in singular), e.g. `amazon`.
2. The dataset class name is the dataset name followed by `Dataset` in camel case, e.g. `AmazonDataset`.
3. The dataset class is defined in `relbench/datasets/<dataset_name>.py`, e.g., `relbench/datasets/amazon.py`.
4. Import and register the dataset in `relbench/datasets/__init__.py` with name `rel-<dataset_name>`, e.g., `register_dataset("rel-amazon", amazon.AmazonDataset). (If you add args, you can use `rel-<dataset_name>-<qualifier>`, e.g., `register_dataset("rel-amazon-fashion", amazon.AmazonDataset, category="fashion")`)
5. After registering the dataset and loading it, it will be available at the location `~/.cache/relbench/rel-<dataset_name>`. Zip the database as follows:
```bash
cd ~/.cache/relbench/rel-amazon
zip -r db db
```
6. Get the SHA256 hash of `db.zip`:
```bash
cd ~/.cache/relbench/rel-amazon
sha256sum db.zip
```
7. Add the hash to `relbench/datasets/hashes.json`:
```
{
... existing hashes ...
    "rel-amazon/db.zip": "db71c7701b892a4eb7481ff04d14d25465795501dba3a5931aabee9930805efe"
}
```
8. Submit a PR with these changes, a link to `db.zip` and a description of your dataset. Be liberal with docstrings and comments to document your dataset and processing steps in the code too.


## Contributing tasks

Similar to above you can also contribute tasks to existing datasets (including community-contributed ones). To define your own tasks in RelBench, please take a look at the tutorial [tutorials/custom_task.ipynb](tutorials/custom_task.ipynb), also available on [Google Colab](https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/custom_task.ipynb). Once you have a working task within RelBench, make a PR to get it added to RelBench. To help RelBench maintainers, please follow these conventions:

1. The task name is `<entity_name>-<single_word>` for entity tasks and `<src_entity_name>-<dst_entity_name>-<single_word>` for recommendation tasks, e.g., `user-churn` and `user-item-review`.
2. The task class name is the task name followed by `Task` in camel case, e.g., `UserChurnTask` and `UserItemReviewTask`.
3. The task class is defined in `relbench/tasks/<dataset_name>.py`, e.g., `relbench/tasks/amazon.py`.
4. Import and register the task in `relbench/tasks/__init__.py`, e.g.:
```python
register_task("rel-amazon", "user-churn", amazon.UserChurnTask)
register_task("rel-amazon", "user-item-review", amazon.UserItemReviewTask)
```
5. After registering the task and loading it, it will be available at the location `~/.cache/relbench/rel-<dataset_name>/tasks`. Zip the task as follows:
```bash
cd ~/.cache/relbench/rel-amazon/tasks
zip -r user-churn user-churn
```
6. Get the SHA256 hash of `<task-name>.zip`:
```bash
cd ~/.cache/relbench/rel-amazon/tasks
sha256sum user-churn.zip
```
7. Add the hash to `relbench/tasks/hashes.json`:
```
{
... existing hashes ...
    "rel-amazon/tasks/user-item-review.zip": "f4e2f2db27bcf30b148b4d00662101f42f68dce545f8ca57b970f534aef74c36",
    "rel-amazon/tasks/user-churn.zip": "cddd511a1f609aea286cf2e20803f489ab9b3158e18cea66e0888d174b1515f0"
}
```
8. Submit a PR with these changes, a link to `<task-name>.zip` and a description of your task. Be liberal with docstrings and comments to document your task and task table construction logic in the code too.
