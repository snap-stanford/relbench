<p align="center"><img src="https://relbench.stanford.edu/img/logo.png" alt="logo" width="600px" /></p>

----

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://relbench.stanford.edu)
[![PyPI version](https://badge.fury.io/py/relbench.svg)](https://badge.fury.io/py/relbench)
[![Testing Status](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml/badge.svg)](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40RelBench)](https://twitter.com/RelBench)

<!-- **Get Started:** loading data &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1PAOktBqh_3QzgAKi53F4JbQxoOuBsUBY?usp=sharing), training model &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1_z0aKcs5XndEacX1eob6csDuR4DYhGQU?usp=sharing). -->


<!-- [<img align="center" src="https://relbench.stanford.edu/img/favicon.png" width="20px" />   -->
[**Website**](https://relbench.stanford.edu) | [**Position Paper**](https://proceedings.mlr.press/v235/fey24a.html) |  [**Benchmark Paper**](https://arxiv.org/abs/2407.20060) | [**Mailing List**](https://groups.google.com/forum/#!forum/relbench/join)

# Overview

<!-- The Relational Deep Learning Benchmark (RelBench) is a collection of realistic, large-scale, and diverse benchmark datasets for machine learning on relational databases. RelBench supports deep learning framework agnostic data loading, task specification, standardized data splitting, and transforming data into graph format. RelBench also provides standardized evaluation metric computations and a leaderboard for tracking progress. -->

<!-- <p align="center"><img src="https://relbench.stanford.edu/img/relbench-fig.png" alt="pipeline" /></p> -->

Relational Deep Learning is a new approach for end-to-end representation learning on data spread across multiple tables, such as in a _relational database_ (see our [position paper](https://relbench.stanford.edu/paper.pdf)). Relational databases are the world's most widely used data management system, and are used for industrial and scientific purposes across many domains. RelBench is a benchmark designed to facilitate efficient, robust and reproducible research on end-to-end deep learning over relational databases.

RelBench contains 7 realistic, large-scale, and diverse relational databases spanning domains including medical, social networks, e-commerce and sport. Each database has multiple predictive tasks (30 in total) defined, each carefully scoped to be both challenging and of domain-specific importance. It provides full support for data downloading, task specification and standardized evaluation in an ML-framework-agnostic manner.

Additionally, RelBench provides a first open-source implementation of a Graph Neural Network based approach to relational deep learning. This implementation uses [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) to load the data as a graph and train GNN models, and [PyTorch Frame](https://github.com/pyg-team/pytorch-frame) for modeling tabular data. Finally, there is an open [leaderboard](https://huggingface.co/relbench) for tracking progress.

<!---**News July 3rd 2024: RelBench v1 is now released!**-->

# Key Papers

[**RelBench: A Benchmark for Deep Learning on Relational Databases**](https://arxiv.org/abs/2407.20060)

This paper details our approach to designing the RelBench benchmark. It also includes a key user study showing that relational deep learning can produce performant models with a fraction of the manual human effort required by typical data science pipelines. This paper is useful for a detailed understanding of RelBench and our initial benchmarking results. If you just want to quickly familiarize with the data and tasks, the [**website**](https://relbench.stanford.edu) is a better place to start.
<!---Joshua Robinson*, Rishabh Ranjan*, Weihua Hu*, Kexin Huang*, Jiaqi Han, Alejandro Dobles, Matthias Fey, Jan Eric Lenssen, Yiwen Yuan, Zecheng Zhang, Xinwei He, Jure Leskovec-->

[**Position: Relational Deep Learning - Graph Representation Learning on Relational Databases (ICML 2024)**](https://proceedings.mlr.press/v235/fey24a.html)

This paper outlines our proposal for how to do end-to-end deep learning on relational databases by combining graph neural networsk with deep tabular models. We reccomend reading this paper if you want to think about new methods for end-to-end deep learning on relational databases. The paper includes a section on possible directions for future research to give a snapshot of some of the research possibilities there are in this area.

<!--- Matthias Fey*, Weihua Hu*, Kexin Huang*, Jan Eric Lenssen*, Rishabh Ranjan, Joshua Robinson*, Rex Ying, Jiaxuan You, Jure Leskovec.-->

# Design of RelBench

<p align="center"><img src="https://relbench.stanford.edu/img/relbench-fig.png" alt="logo" width="900px" /></p>

RelBench has the following main components:
1. 7 databases with a total of 30 tasks; both of these automatically downloadable for ease of use
2. Easy data loading, and graph construction from pkey-fkey links
3. Your own model, which can use any deep learning stack since RelBench is framework-agnostic. We provide a first model implementation using PyTorch Geometric and PyTorch Frame.
4. Standardized evaluators - all you need to do is produce a list of predictions for test samples, and RelBench computes metrics to ensure standardized evaluation
5. A leaderboard you can upload your results to, to track SOTA progress.


# Installation

You can install RelBench using `pip`:
```bash
pip install relbench
```

This will allow usage of the core RelBench data and task loading functionality.

To additionally use `relbench.modeling`, which requires [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) and [PyTorch Frame](https://github.com/pyg-team/pytorch-frame), install these dependencies manually or do:

```bash
pip install relbench[full]
```


For the scripts in the `examples` directory, use:
```bash
pip install relbench[example]
```

Then, to run a script:
```bash
git clone https://github.com/snap-stanford/relbench
cd relbench/examples
python gnn_node.py --dataset rel-f1 --task driver-position
```


# Package Usage

This section provides a brief overview of using the RelBench package. For a more in-depth coverage see the [Tutorials](#tutorials) section. For detailed documentations, please see the code directly.

Imports:
```python
from relbench.base import Table, Database, Dataset, EntityTask
from relbench.datasets import get_dataset
from relbench.tasks import get_task
```

Get a dataset, e.g., `rel-amazon`:
```python
dataset: Dataset = get_dataset("rel-amazon", download=True)
```

<details>
    <summary>Details on downloading and caching behavior.</summary>

RelBench datasets (and tasks) are cached to disk (usually at `~/.cache/relbench`). If not present in cache, `download=True` downloads the data, verifies it against the known hash, and caches it. If present, `download=True` performs the verification and avoids downloading if verification succeeds. This is the recommended way.

`download=False` uses the cached data without verification, if present, or processes and caches the data from scratch / raw sources otherwise.
</details>

`dataset` consists of a `Database` object and temporal splitting times `dataset.val_timestamp` and `dataset.test_timestamp`.

To get the database:
```python
db: Database = dataset.get_db()
```

<details>
    <summary>Preventing temporal leakage</summary>

By default, rows with timestamp > `dataset.test_timestamp` are excluded to prevent accidental temporal leakage. The full database can be obtained with:
```python
full_db: Database = dataset.get_db(upto_test_timestamp=False)
```
</details>

Various tasks can be defined on a dataset. For example, to get the `user-churn` task for `rel-amazon`:
```python
task: EntityTask = get_task("rel-amazon", "user-churn", download=True)
```

A task provides train/val/test tables:
```python
train_table: Table = task.get_table("train")
val_table: Table = task.get_table("val")
test_table: Table = task.get_table("test")
```

<details>
    <summary>Preventing test leakage</summary>
By default, the target labels are hidden from the test table to prevent accidental data leakage. The full test table can be obtained with:

```python
full_test_table: Table = task.get_table("test", mask_input_cols=False)
```
</details>

You can build your model on top of the database and the task tables. After training and validation, you can make prediction from your model on the test table. Suppose your prediction `test_pred` is a NumPy array following the order of `task.test_table`, you can call the following to get the evaluation metrics:

```python
task.evaluate(test_pred)
```

Additionally, you can evaluate validation (or training) predictions as such:
```python
task.evaluate(val_pred, val_table)
```

# Tutorials

| Notebook | Try on Colab | Description                                             |
----------|--------------|---------------------------------------------------------|
| [load_data.ipynb](tutorials/load_data.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/load_data.ipynb)   | Load and explore RelBench data
| [train_model.ipynb](tutorials/train_model.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/train_model.ipynb)| Train your first GNN-based model on RelBench
| [custom_dataset.ipynb](tutorials/custom_dataset.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/custom_dataset.ipynb)   | Use your own data in RelBench
| [custom_task.ipynb](tutorials/custom_task.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/custom_task.ipynb)| Define your own ML tasks in RelBench


# Contributing

Please check out [CONTRIBUTING.md](CONTRIBUTING.md) if you are interested in contributing datasets, tasks, bug fixes, etc. to RelBench.


# Cite RelBench

If you use RelBench in your work, please cite our position and benchmark papers:

```bibtex
@inproceedings{rdl,
  title={Position: Relational Deep Learning - Graph Representation Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and Huang, Kexin and Lenssen, Jan Eric and Ranjan, Rishabh and Robinson, Joshua and Ying, Rex and You, Jiaxuan and Leskovec, Jure},
  booktitle={Forty-first International Conference on Machine Learning}
}
```

```bibtex
@misc{relbench,
      title={RelBench: A Benchmark for Deep Learning on Relational Databases},
      author={Joshua Robinson and Rishabh Ranjan and Weihua Hu and Kexin Huang and Jiaqi Han and Alejandro Dobles and Matthias Fey and Jan E. Lenssen and Yiwen Yuan and Zecheng Zhang and Xinwei He and Jure Leskovec},
      year={2024},
      eprint={2407.20060},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.20060},
}
```
