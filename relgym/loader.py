from typing import Dict
import os
import copy
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
from torch_frame.config.text_embedder import TextEmbedderConfig
from examples.inferred_stypes import dataset2inferred_stypes
from examples.text_embedder import GloveTextEmbedding  # May not be the best practice
from relgym.config import cfg
from relbench.data import RelBenchDataset
from relbench.datasets import get_dataset
from relbench.external.graph import (
    get_node_train_table_input,
    make_pkey_fkey_graph,
)


def create_dataset_and_task():
    r"""

    Load dataset objects.

    Returns: PyG dataset object

    """
    name = cfg.dataset.name
    cache_dir = cfg.dataset.cache_dir
    # Load Relbench dataset
    dataset: RelBenchDataset = get_dataset(name=name, process=True, cache_dir=cache_dir)
    task = dataset.get_task(cfg.dataset.task, process=True)
    return dataset, task


def transform_dataset_to_graph(dataset: RelBenchDataset):
    device = cfg.device
    col_to_stype_dict = copy.deepcopy(dataset2inferred_stypes[cfg.dataset.name])

    data: HeteroData = make_pkey_fkey_graph(
        dataset.db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256
        ),
        cache_dir=os.path.join(cfg.dataset.root_dir, f"{cfg.dataset.name}_materialized_cache"),
    )

    return data


def get_loader_and_entity(data, task):
    loader_dict: Dict[str, NeighborLoader] = {}
    for split, table in [
        ("train", task.train_table),
        ("val", task.val_table),
        ("test", task.test_table),
    ]:
        table_input = get_node_train_table_input(table=table, task=task)
        entity_table = table_input.nodes[0]
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[cfg.loader.num_neighbors for _ in range(cfg.model.num_layers)],
            # num_neighbors=[int(cfg.loader.num_neighbors / 2**i) for i in range(cfg.model.num_layers)],
            time_attr="time",
            temporal_strategy=cfg.loader.temporal_strategy,
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=cfg.loader.batch_size,
            shuffle=split == "train",
            num_workers=cfg.loader.num_workers,
            persistent_workers=cfg.loader.num_workers > 0,
        )

    return loader_dict, entity_table


def create_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    dataset, task = create_dataset_and_task()
    data = transform_dataset_to_graph(dataset)
    loader_dict, entity_table = get_loader_and_entity(data, task)

    return loader_dict, entity_table, task, data
