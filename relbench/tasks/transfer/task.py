from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import NodeLoader

from relbench.base import EntityTask
from relbench.modeling.loader import CustomNodeLoader  # Re-use existing loader logic


class CrossDomainEntityTask:
    r"""A wrapper task that aggregates multiple EntityTasks (source domains)
    and one target EntityTask (target domain) for cross-domain transfer learning.

    Args:
        source_tasks (List[EntityTask]): List of tasks to train on.
        target_task (EntityTask): Task to evaluate zero-shot transfer on.
    """

    def __init__(
        self,
        source_tasks: List[EntityTask],
        target_task: EntityTask,
    ):
        self.source_tasks = source_tasks
        self.target_task = target_task
        
        # Verify that all tasks are of compatible types (e.g., all EntityTasks)
        # In v1, we assume they all share the same task_type (e.g. Binary Classification)
        # to simplify the loss function.
        self.task_type = target_task.task_type
        for t in source_tasks:
            if t.task_type != self.task_type:
                raise ValueError(
                    f"Task type mismatch: {t.task_type} vs {self.task_type}"
                )

    def get_loader(
        self,
        dataset_name_to_data: dict[str, HeteroData],
        split: str = "train",
        batch_size: int = 512,
        **kwargs,
    ):
        r"""Returns a loader that samples from the appropriate tasks.
        
        For 'train', it creates a CombinedLoader that samples from all source tasks.
        For 'val'/'test', it samples from the target task.
        """
        if split == "train":
            # For training, we want to sample from all source tasks.
            # We can simply return a list of loaders, or better, 
            # a generator that yields from them in a round-robin or mixed fashion.
            # For simplicity in this initial implementation, we will concatenate
            # the training data from all source tasks if they were one giant graph,
            # BUT since they are separate graphs (data objects), we need a 
            # custom CombinedLoader.
            
            loaders = []
            for task in self.source_tasks:
                # We need to map the task back to its dataset data object
                # This basic implementation assumes user passes a dict mapping 
                # dataset_name -> data object.
                dataset_name = task.dataset.dataset_name
                if dataset_name not in dataset_name_to_data:
                    raise ValueError(f"Data for {dataset_name} not provided.")
                
                data = dataset_name_to_data[dataset_name]
                table = task.get_table("train")
                
                # Get the node indices for training
                from relbench.modeling.graph import get_node_train_table_input
                table_input = get_node_train_table_input(table, task)
                
                loader = CustomNodeLoader(
                    data,
                    node_sampler=kwargs.get("node_sampler"), # User must provide sampler setup
                    input_nodes=table_input.nodes,
                    input_time=table_input.time,
                    transform=table_input.transform,
                    batch_size=batch_size,
                    shuffle=True,
                )
                loaders.append(loader)
            
            return CombinedLoader(loaders)

        else:
            # For val/test (target domain)
            dataset_name = self.target_task.dataset.dataset_name
            data = dataset_name_to_data[dataset_name]
            table = self.target_task.get_table(split)
            
            from relbench.modeling.graph import get_node_train_table_input
            table_input = get_node_train_table_input(table, self.target_task)
             
            return CustomNodeLoader(
                data,
                node_sampler=kwargs.get("node_sampler"),
                input_nodes=table_input.nodes,
                input_time=table_input.time,
                transform=table_input.transform,
                batch_size=batch_size,
                shuffle=False,
            )


class CombinedLoader:
    r"""A simple iterator that yields batches from multiple loaders."""
    def __init__(self, loaders):
        self.loaders = loaders
        # We define length as sum of lengths
        self._len = sum(len(l) for l in loaders)

    def __iter__(self):
        # We iterate through loaders sequentially for now.
        # Ideally we would interleave batches.
        for loader in self.loaders:
            for batch in loader:
                yield batch

    def __len__(self):
        return self._len
