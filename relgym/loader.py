from typing import Dict
import os
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
import torch_frame
from torch_frame.config.text_embedder import TextEmbedderConfig
from examples.text_embedder import GloveTextEmbedding  # May not be the best practice
from relgym.config import cfg
from relbench.data import RelBenchDataset
from relbench.datasets import get_dataset
from relbench.external.graph import (
    get_stype_proposal,
    get_train_table_input,
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
    # Stores the informative text columns to retain for each table:
    dataset_to_informative_text_cols = {}
    dataset_to_informative_text_cols["rel-stackex"] = {
        "postHistory": ["Text"],
        "users": ["AboutMe"],
        "posts": ["Body", "Title", "Tags"],
        "comments": ["Text"],
    }
    col_to_stype_dict = get_stype_proposal(dataset.db)
    informative_text_cols: Dict = dataset_to_informative_text_cols[cfg.dataset.name]
    for table_name, stype_dict in col_to_stype_dict.items():
        for col_name, stype in list(stype_dict.items()):
            # Remove text columns except for the informative ones:
            if stype == torch_frame.text_embedded:
                if col_name not in informative_text_cols.get(table_name, []):
                    del stype_dict[col_name]

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
        table_input = get_train_table_input(table=table, task=task)
        entity_table = table_input.nodes[0]
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[cfg.loader.num_neighbors, cfg.loader.num_neighbors],
            time_attr="time",
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
