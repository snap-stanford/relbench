import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.utils import filter_hetero_data, infer_filter_per_worker
from torch_geometric.sampler import (
    BaseSampler,
    HeteroSamplerOutput,
    NeighborSampler,
    NodeSamplerInput,
)
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, NodeType, OptTensor


class TimestampSampler(Sampler[int]):
    r"""A TimestampSampler that samples rows from the same timestamp."""

    def __init__(
        self,
        timestamp: Tensor,
        batch_size: int,
    ):
        self.batch_size = batch_size
        self.time_dict = {
            int(time): (timestamp == time).nonzero().view(-1)
            for time in timestamp.unique()
        }
        self.num_batches = sum(
            [indices.numel() // batch_size for indices in self.time_dict.values()]
        )

    def __iter__(self) -> Iterator[list[int]]:
        all_batches = []
        for indices in self.time_dict.values():
            # Random shuffle values:
            indices = indices[torch.randperm(indices.numel())]
            batches = torch.split(indices, self.batch_size)
            for batch in batches:
                if len(batch) < self.batch_size:
                    continue
                else:
                    all_batches.append(batch.tolist())

        random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        return self.num_batches


class CustomLinkDataset(Dataset):
    r"""A custom link prediction dataset. Sample source nodes, time, and one
    positive destination node"""

    def __init__(
        self,
        src_node_indices: Tensor,
        dst_node_indices_list: List[List[int]],
        num_dst_nodes: int,
        src_time: Tensor,
    ):
        self.src_node_indices = src_node_indices
        self.dst_node_indices_list = dst_node_indices_list
        self.num_dst_nodes = num_dst_nodes
        self.dst_node_sets_list = [
            set(indices) for indices in self.dst_node_indices_list
        ]
        self.src_time = src_time

    def __getitem__(self, index) -> Tensor:
        r"""Returns 1-dim tensor of size 4
        - source node index
        - positive destination node index
        - negative destination node index
        - source node time
        """
        excluding_set = self.dst_node_sets_list[index]
        random_idx = random.randint(0, self.num_dst_nodes - 1)
        max_trial = 5
        trials = 0
        while random_idx in excluding_set or trials > max_trial:
            random_idx = random.randint(0, self.num_dst_nodes - 1)
            trials += 1

        return torch.tensor(
            [
                self.src_node_indices[index],
                random.choice(self.dst_node_indices_list[index]),
                random_idx,
                self.src_time[index],
            ]
        )

    def __len__(self):
        return len(self.src_node_indices)


class CustomLinkLoader(DataLoader):
    r"""A custom data loader for link prediction. Based on
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/node_loader.html#NodeLoader

    Args:
        data (Any): A :class:`~torch_geometric.data.Data`,
            :class:`~torch_geometric.data.HeteroData`, or
            (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        node_sampler (torch_geometric.sampler.BaseSampler): The sampler
            implementation to be used with this loader.
            Needs to implement
            :meth:`~torch_geometric.sampler.BaseSampler.sample_from_nodes`.
            The sampler implementation must be compatible with the input
            :obj:`data` object.
        src_nodes (Tuple[NodetType, Tensor]): A tensor of source node indices.
        dst_nodes_list (Tuple[NodeType, List[List[int]]): A list of
            destination node indices for src_nodes[i].
        num_dst_nodes (int): Total number of destination nodes. Used to
            determine the range of negative samples.
        src_time (torch.Tensor, optional): Optional values to override the
            timestamp for the input nodes given in :obj:`input_nodes`. If not
            set, will use the timestamps in :obj:`time_attr` as default (if
            present). The :obj:`time_attr` needs to be set for this to work.
            (default: :obj:`None`)
        share_same_time (bool): Whether to share the seed time within mini-batch
            or not (default: :obj:`False`)
    """

    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        node_sampler: BaseSampler,
        src_nodes: Tuple[NodeType, Tensor],
        dst_nodes_list: Tuple[NodeType, List[List[int]]],
        num_dst_nodes: int,
        src_time: OptTensor = None,
        share_same_time: bool = False,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional[HeteroData] = None,
        input_id: OptTensor = None,
        **kwargs,
    ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        # always assume time is given
        assert src_time is not None

        self.data = data
        self.node_sampler = node_sampler
        self.src_nodes = src_nodes
        self.dst_nodes_list = dst_nodes_list
        self.num_dst_nodes = num_dst_nodes
        self.src_time = src_time
        self.share_same_time = share_same_time
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls
        self.input_id = input_id

        kwargs.pop("dataset", None)
        kwargs.pop("collate_fn", None)
        if share_same_time:
            kwargs.pop("sampler", None)
            kwargs["batch_sampler"] = TimestampSampler(
                src_time,
                kwargs["batch_size"],
            )
            kwargs.pop("batch_size", None)

        dataset = CustomLinkDataset(
            self.src_nodes[1],
            dst_nodes_list[1],
            num_dst_nodes,
            src_time,
        )
        self.src_node_type = self.src_nodes[0]
        self.dst_node_type = self.dst_nodes_list[0]

        super().__init__(dataset, collate_fn=self.collate_fn, **kwargs)

    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input nodes."""
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            src_out, pos_dst_out, neg_dst_out = out
            src_out = self.filter_fn(src_out)
            pos_dst_out = self.filter_fn(pos_dst_out)
            neg_dst_out = self.filter_fn(neg_dst_out)
            out = src_out, pos_dst_out, neg_dst_out
        return out

    def collate_fn(self, index: Tensor) -> Any:
        r"""Samples a subgraph from a batch of input nodes."""
        index = torch.stack(index)
        assert index.shape[1] == 4
        src_indices = index[:, 0].contiguous()
        pos_dst_indices = index[:, 1].contiguous()
        neg_dst_indices = index[:, 2].contiguous()
        time = index[:, 3].contiguous()

        src_out = self.node_sampler.sample_from_nodes(
            NodeSamplerInput(
                input_id=src_indices,
                node=src_indices,
                time=time,
                input_type=self.src_node_type,
            )
        )
        if self.filter_per_worker:  # Execute `filter_fn` in the worker process
            src_out = self.filter_fn(src_out, self.src_node_type)

        pos_dst_out = self.node_sampler.sample_from_nodes(
            NodeSamplerInput(
                input_id=pos_dst_indices,
                node=pos_dst_indices,
                time=time,
                input_type=self.dst_node_type,
            )
        )
        if self.filter_per_worker:  # Execute `filter_fn` in the worker process
            pos_dst_out = self.filter_fn(pos_dst_out, self.dst_node_type)

        neg_dst_out = self.node_sampler.sample_from_nodes(
            NodeSamplerInput(
                input_id=neg_dst_indices,
                node=neg_dst_indices,
                time=time,
                input_type=self.dst_node_type,
            )
        )
        if self.filter_per_worker:  # Execute `filter_fn` in the worker process
            neg_dst_out = self.filter_fn(neg_dst_out, self.dst_node_type)

        return src_out, pos_dst_out, neg_dst_out

    def filter_fn(
        self,
        out: HeteroSamplerOutput,
        node_type: NodeType,
    ) -> HeteroData:
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object to be used downstream.
        """
        if self.transform_sampler_output:
            out = self.transform_sampler_output(out)

        data = filter_hetero_data(  #
            self.data,
            out.node,
            out.row,
            out.col,
            out.edge,
            self.node_sampler.edge_permutation,
        )

        for key, node in out.node.items():
            if "n_id" not in data[key]:
                data[key].n_id = node

        for key, edge in (out.edge or {}).items():
            if edge is not None and "e_id" not in data[key]:
                edge = edge.to(torch.long)
                perm = self.node_sampler.edge_permutation
                if perm is not None and perm.get(key, None) is not None:
                    edge = perm[key][edge]
                data[key].e_id = edge

        data.set_value_dict("batch", out.batch)
        data.set_value_dict("num_sampled_nodes", out.num_sampled_nodes)
        data.set_value_dict("num_sampled_edges", out.num_sampled_edges)

        # TODO: Add this back in PyG 2.5.0
        # if out.orig_row is not None and out.orig_col is not None:
        #     for key in out.orig_row.keys():
        #         data[key]._orig_edge_index = torch.stack([
        #             out.orig_row[key],
        #             out.orig_col[key],
        #         ],
        #                                                  dim=0)

        data[node_type].input_id = out.metadata[0]
        data[node_type].seed_time = out.metadata[1]
        data[node_type].batch_size = out.metadata[0].size(0)

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()

        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LinkNeighborLoader(CustomLinkLoader):
    r"""A custom neighbor loader for link prediction.
    Based on https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_loader.html

    Args:
        src_nodes (Tuple[NodetType, Tensor]): A tensor of source node indices.
        dst_nodes_list (Tuple[NodeType, List[List[int]]): A list of
            destination node indices for src_nodes[i].
        num_dst_nodes (int): Total number of destination nodes. Used to
            determine the range of negative samples.
        src_time (torch.Tensor, optional): Optional values to override the
            timestamp for the input nodes given in :obj:`input_nodes`. If not
            set, will use the timestamps in :obj:`time_attr` as default (if
            present). The :obj:`time_attr` needs to be set for this to work.
            (default: :obj:`None`)
        share_same_time (bool): Whether to share the seed time within mini-batch
            or not (default: :obj:`False`)
    """

    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        src_nodes: Tuple[NodeType, Tensor],
        dst_nodes_list: Tuple[NodeType, List[List[int]]],
        num_dst_nodes: int,
        src_time: OptTensor = None,
        share_same_time: bool = False,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: Optional[bool] = None,
        neighbor_sampler: Optional[NeighborSampler] = None,
        **kwargs,
    ):
        if neighbor_sampler is None:
            neighbor_sampler = NeighborSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                weight_attr=weight_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get("num_workers", 0) > 0,
            )

        super().__init__(
            data=data,
            node_sampler=neighbor_sampler,
            src_nodes=src_nodes,
            dst_nodes_list=dst_nodes_list,
            num_dst_nodes=num_dst_nodes,
            src_time=src_time,
            share_same_time=share_same_time,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )
