import copy
import os
import torch
from typing import Dict

from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader

from examples.inferred_stypes import dataset2inferred_stypes
from examples.text_embedder import get_text_embedder  # May not be the best practice
from relbench.data import RelBenchDataset
from relbench.datasets import get_dataset
from relbench.external.graph import get_node_train_table_input, make_pkey_fkey_graph, get_link_train_table_input
from relbench.external.loader import LinkNeighborLoader
from relbench.data.task_base import TaskType
from relgym.config import cfg


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
    if cfg.torch_frame_model.text_embedder == 'glove':  # Inherit
        cache_dir = f"{cfg.dataset.name}_materialized_cache"
    else:
        cache_dir = f"{cfg.dataset.name}_{cfg.torch_frame_model.text_embedder}_materialized_cache"

    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=get_text_embedder(
                cfg.torch_frame_model.text_embedder, device=device
            ),
            batch_size=256,
        ),
        cache_dir=os.path.join(
            cfg.dataset.root_dir,
            cache_dir,
        ),
    )

    return data, col_stats_dict


def create_loader(data, task):

    if task.task_type in [TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION]:  # for node-level tasks

        loader_dict: Dict[str, NeighborLoader] = {}
        for split, table in [
            ("train", task.train_table),
            ("val", task.val_table),
            ("test", task.test_table),
        ]:
            table_input = get_node_train_table_input(table=table, task=task)
            loader_dict[split] = NeighborLoader(
                data,
                num_neighbors=[
                    cfg.loader.num_neighbors for _ in range(cfg.model.num_layers)
                ],
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

    elif task.task_type == TaskType.LINK_PREDICTION:  # for link prediction task

        loader_dict: Dict = {}
        train_table_input = get_link_train_table_input(task.train_table, task)
        num_neighbors = [cfg.loader.num_neighbors for _ in range(cfg.model.num_layers)]
        train_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=num_neighbors,
            time_attr="time",
            src_nodes=train_table_input.src_nodes,
            dst_nodes=train_table_input.dst_nodes,
            num_dst_nodes=train_table_input.num_dst_nodes,
            src_time=train_table_input.src_time,
            share_same_time=cfg.loader.share_same_time,
            batch_size=cfg.loader.batch_size,
            temporal_strategy=cfg.loader.temporal_strategy,
            # if share_same_time is True, we use sampler, so shuffle must be set False
            shuffle=not cfg.loader.share_same_time,
            num_workers=cfg.loader.num_workers,
        )
        loader_dict["train"] = train_loader

        for split in ["val", "test"]:
            seed_time = task.val_seed_time if split == "val" else task.test_seed_time
            target_table = task.val_table if split == "val" else task.test_table
            src_node_indices = torch.from_numpy(target_table.df[task.src_entity_col].values)
            src_loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                time_attr="time",
                input_nodes=(task.src_entity_table, src_node_indices),
                input_time=torch.full(
                    size=(len(src_node_indices),), fill_value=seed_time, dtype=torch.long
                ),
                batch_size=cfg.loader.batch_size,
                shuffle=False,
                num_workers=cfg.loader.num_workers,
            )
            dst_loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                time_attr="time",
                input_nodes=task.dst_entity_table,
                input_time=torch.full(
                    size=(task.num_dst_nodes,), fill_value=seed_time, dtype=torch.long
                ),
                batch_size=cfg.loader.batch_size,
                shuffle=False,
                num_workers=cfg.loader.num_workers,
            )
            loader_dict[split] = (src_loader, dst_loader)

    else:
        raise NotImplementedError(task.task_type)

    return loader_dict

