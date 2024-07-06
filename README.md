<p align="center"><img src="https://relbench.stanford.edu/img/logo.png" alt="logo" width="600px" /></p>

----

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://relbench.stanford.edu)
[![PyPI version](https://badge.fury.io/py/relbench.svg)](https://badge.fury.io/py/relbench)
[![Testing Status](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml/badge.svg)](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40RelBench)](https://twitter.com/RelBench)

**Get Started:** loading data &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1PAOktBqh_3QzgAKi53F4JbQxoOuBsUBY?usp=sharing), training model &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1_z0aKcs5XndEacX1eob6csDuR4DYhGQU?usp=sharing).


 [<img align="center" src="https://relbench.stanford.edu/img/favicon.png" width="20px" />  **Website**](https://relbench.stanford.edu) | [**Vision Paper**](https://relbench.stanford.edu/paper.pdf) |  [**Benchmark Paper**](https://relbench.stanford.edu/paper.pdf) | [**Mailing List**](https://groups.google.com/forum/#!forum/relbench/join)

# Overview

<!-- The Relational Deep Learning Benchmark (RelBench) is a collection of realistic, large-scale, and diverse benchmark datasets for machine learning on relational databases. RelBench supports deep learning framework agnostic data loading, task specification, standardized data splitting, and transforming data into graph format. RelBench also provides standardized evaluation metric computations and a leaderboard for tracking progress. -->

<!-- <p align="center"><img src="https://relbench.stanford.edu/img/relbench-fig.png" alt="pipeline" /></p> -->

Relational Deep Learning is a new approach for end-to-end representation learning on data spread across multiple tables, such as in a _relational database_ (see our [vision paper](https://relbench.stanford.edu/paper.pdf)). Relational databases are the world's most widely used database management system, and are used for industrial and scientific purposes accross many domains. RelBench is a benchmark designed to facilitate efficient, robust and reproducible research in end-to-end deep learning on relational databases. RelBench contains 7 realistic, large-scale, and diverse relational databases spanning domains including medical, social networks, e-commerce and sport. Each database has multiple predictive tasks (29 in total) defined, each carefully scoped to be both challenging and of domain-specific importance. It provides full support for data downloading, task specification and standardized evaluation in an ML-framework-agnostic manner.

Additionally, RelBench provides a first open-source implementation of a Graph Neural Network based approach to relational deep learning. This implementation uses [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) to load the data as a graph and train GNN models, and [PyTorch Frame](https://github.com/pyg-team/pytorch-frame) to encode the various types of table columns. Finally, there is an open [leaderboard](https://huggingface.co/relbench) for tracking progress.

<!---**News July 3rd 2024: RelBench v1 is now released!**-->

# Key Papers

 [**RelBench Paper**](https://relbench.stanford.edu/paper.pdf) [RelBench: A Benchmark for Deep Learning
on Relational Databases.]

This paper details our approach to designing the RelBench benchmark. It also includes a key user study showing that relational deep learning can produce performant models with a fraction of the manual human effort required by typical data science pipelines. This paper is useful for a detailed understanding of RelBench and our initial benchmarking results. If you just want to quickly familiarize with the data and tasks, the [**website**](https://relbench.stanford.edu) is a better place to start.
<!---Joshua Robinson*, Rishabh Ranjan*, Weihua Hu*, Kexin Huang*, Jiaqi Han, Alejandro Dobles, Matthias Fey, Jan Eric Lenssen, Yiwen Yuan, Zecheng Zhang, Xinwei He, Jure Leskovec-->

 [**Vision Paper**](https://relbench.stanford.edu/paper.pdf) [Relational Deep Learning: Graph Representation
Learning on Relational Databases.]

This paper outlines our proposal for how to do end-to-end deep learning on relational databases by combining graph neural networsk with deep tabular models. We reccomend reading this paper if you want to think about new methods for end-to-end deep learning on relational databases. The paper includes a section on possible directions for future research to give a snapshot of some of the research possilibities there are in this area.

<!--- Matthias Fey*, Weihua Hu*, Kexin Huang*, Jan Eric Lenssen*, Rishabh Ranjan, Joshua Robinson*, Rex Ying, Jiaxuan You, Jure Leskovec.-->

# Design of RelBench

<p align="center"><img src="https://relbench.stanford.edu/img/relbench-fig.png" alt="logo" width="900px" /></p>

RelBench has the following main components:
1. 7 databases, each automatically downloadable for ease of use (with the exception of H&M, for which RelBench gives other instructions)
2. Easy 1-line loading of data, including loading the raw tables, and also code for constructing a graph from pkey-fkey links
3. Your own model, which can use any deep learning stack since RelBench is framework-agnostic. We provide a first model implementation using PyTorch Geometric and PyTorch Frame.
4. Standardized evaluators - all you need to do is produce a list of predictions for test samples, and RelBench computes metrics to ensure standardized evaluation
5. A leaderboard you can upload your results to, to track SOTA progress.


# Installation

You can install RelBench using `pip`:

```
pip install relbench
```

This will allow usage of the RelBench data and task loading functionality. To additionally use the example GNN scripts in the ```examples``` directory, and the graph-related helper functions found in ```relbench/modeling``` it is also necessary to install [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) and [PyTorch Frame](https://github.com/pyg-team/pytorch-frame). PyTorch Frame can simply be installed with


```
pip install pytorch_frame
```

and the PyTorch Geometric installation instructions can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Note that as well as ```torch_geometric```, you will also need to install the optional dependencies ```pyg_lib```, ```torch_scatter```, ```torch_sparse```.

# Package Usage

Here we describe key functions of RelBench. RelBench provides a collection of APIs for easy access to machine-learning-ready relational databases.

To see all available datasets:
```python
from relbench.datasets import dataset_names
print(dataset_names)
```

For a concrete example, to obtain the `rel-stack` relational database, a database of questions and answers from Stack Exchange, do:

```python
from relbench.datasets import get_dataset
dataset = get_dataset(name="rel-stack")
```

To see the tasks available for this dataset:
```python
print(dataset.task_names)
```

Next, to retrieve the `posts-votes` predictive task, which is to predict the upvotes of a post it will receive in the next 2 years, simply do:

```python
task = dataset.get_task("post-votes")
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

# Tutorials
To get started with RelBench, we provide some helpful Colab notebook tutorials. For now these tutorials cover (i) how to load data using RelBench, focusing on providing users with the understanding of RelBench data logic needed to use RelBench data freely with any desired ML models, and (ii) training a GNN predictive model to solve any tasks in RelBench.

| Name  | Description                                             |
|-------|---------------------------------------------------------|
| Loading Data &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1PAOktBqh_3QzgAKi53F4JbQxoOuBsUBY?usp=sharing)   | How to load and explore RelBench data
| Training models &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1_z0aKcs5XndEacX1eob6csDuR4DYhGQU?usp=sharing)| Train your first GNN-based model on RelBench.                   |



# Cite RelBench

If you use RelBench in your work, please cite our position paper and benchmark paper:
```
@article{relationaldeeplearning,
  title={Relational Deep Learning: Graph Representation Learning on Relational Tables},
  author={Matthias Fey, Weihua Hu, Kexin Huang, Jan Eric Lenssen, Rishabh Ranjan, Joshua Robinson, Rex Ying, Jiaxuan You, Jure Leskovec},
  journal={ICML Position Paper}
  year={2024}
}
```

```
@article{relbench,
  title={RelBench: A Benchmark for Deep Learning on Relational Databases},
  author={Joshua Robinson, Rishabh Ranjan, Weihua Hu, Kexin Huang, Jiaqi Han, Alejandro Dobles, Matthias Fey, Jan Eric Lenssen, Yiwen Yuan, Zecheng Zhang, Xinwei He, Jure Leskovec},
  year={2024}
}
```
