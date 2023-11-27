<p align="center"><img src="https://relbench.stanford.edu/img/logo.png" alt="logo" width="600px" /></p>

----

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://relbench.stanford.edu)
[![PyPI version](https://badge.fury.io/py/relbench.svg)](https://badge.fury.io/py/relbench)
[![Testing Status](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml/badge.svg)](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40RelBench)](https://twitter.com/RelBench)

[**Website**](https://relbench.stanford.edu) | [**Vision Paper**](https://relbench.stanford.edu/paper.pdf) | [**Mailing List**](https://groups.google.com/forum/#!forum/relbench/join)

# Overview

<!-- The Relational Deep Learning Benchmark (RelBench) is a collection of realistic, large-scale, and diverse benchmark datasets for machine learning on relational databases. RelBench supports deep learning framework agnostic data loading, task specification, standardized data splitting, and transforming data into graph format. RelBench also provides standardized evaluation metric computations and a leaderboard for tracking progress. -->

<!-- <p align="center"><img src="https://relbench.stanford.edu/img/relbench-fig.png" alt="pipeline" /></p> -->

Relational Deep Learning is a new approach for end-to-end representation learning on data spread across multiple tables, such as in a _relational database_ (see our [vision paper](https://relbench.stanford.edu/paper.pdf)). RelBench is the accompanying benchmark which seeks to facilitate efficient, robust and reproducible research in this direction. It comprises of a collection of realistic, large-scale, and diverse datasets structured as relational tables, along with machine learning tasks defined on them. It provides full support for data downloading, task specification and standardized evaluation in an ML-framework-agnostic manner. Additionally, there is seamless integration with [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) to load the data as a graph and train GNN models, and with [PyTorch Frame](https://github.com/pyg-team/pytorch-frame) to encode the various types of table columns. Finally, there is a leaderboard for tracking progress.

**RelBench is in its beta release stage, and we are planning to increase datasets and benchmarking in the near future. Datasets in the current version are subject to change.**


# Installation

You can install RelBench using `pip`:

```
pip install relbench
```

# Package Usage

Here we describe key functions of RelBench. RelBench provides a collection of APIs for easy access to machine-learning-ready relational databases.

To see all available datasets:
```python
from relbench.datasets import dataset_names
print(dataset_names)
```

For a concrete example, to obtain the `rel-stackex` relational database, do:

```python
from relbench.datasets import get_dataset
dataset = get_dataset(name="rel-stackex")
```

To see the tasks available for this dataset:
```python
print(dataset.task_names)
```

Next, to retrieve the `rel-stackex-votes` predictive task, which is to predict the upvotes of a post it will receive in the next 2 years, simply do:

```python
task = dataset.get_task("rel-stackex-votes")
task.train_table, task.val_table, task.test_table # training/validation/testing tables
```

The training/validation/testing tables are automatically generated using pre-defined standardized temporal split. You can then build your favorite relational deep learning model on top of it. After training and validation, you can make prediction from your model on `task.test_table`. Suppose your prediction `test_pred` is an array following the order of `task.test_table`, you can call the following to retrieve the unified evaluation metrics:

```python
task.evaluate(test_pred)
```

Additionally, you can evaluate validation (or training) predictions as such:
```python
task.evaluate(val_pred, task.val_table)
```

# Demos
List of working demos:

| Name  | Description                                             |
|-------|---------------------------------------------------------|
| [rel-stackex](examples/stackex/demo-stackex.ipynb)   | exploring `rel-stackex` dataset and tasks                           |
| [rel-amazon](examples/amazon/demo-amazon.ipynb)   | exploring `rel-amazon` dataset and tasks                           |

# Cite RelBench

If you use RelBench in your work, please cite our paper:
```
@article{relbench,
  title={Relational Deep Learning: Graph Representation Learning on Relational Tables},
  author={Matthias Fey, Weihua Hu, Kexin Huang, Jan Eric Lenssen, Rishabh Ranjan, Joshua Robinson, Rex Ying, Jiaxuan You, Jure Leskovec},
  year={2023}
}
```
