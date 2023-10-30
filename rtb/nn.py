import torch_frame as pyf
import torch_geometric as pyg

import rtb


def to_pyf_dataset(table: rtb.data.Table) -> pyf.data.Dataset:
    r"""Converts a Table to a PyF Dataset.

    Primary key and foreign keys are removed in this process."""

    raise NotImplementedError


def make_graph(db: rtb.data.Database) -> pyg.data.HeteroData:
    """
    Models the database as a heterogeneous graph.

    Instead of node embeddings in data.x, we store the tensor frames in data.tf.
    """

    data = pyg.data.HeteroData()

    for name, table in db.tables.items():
        # materialize the tables
        pyf_dataset = to_pyf_dataset(table)
        pyf_dataset.materialize()
        data[name].tf = pyf_dataset.tensor_frame
        data[name].col_stats = pyf_dataset.col_stats

        # add edges
        for col_name, pkey_name in table.fkeys.items():
            fkey_idx = torch.tensor(table.df[table.primary_key])
            pkey_idx = torch.tensor(table.df[col_name])

            # fkey -> pkey edges
            data[name, "f2p::" + col_name, pkey_name].edge_index = torch.stack(
                [fkey_idx, pkey_idx]
            )
            # pkey -> fkey edges
            data[pkey_name, "p2f::" + col_name, name].edge_index = torch.stack(
                [pkey_idx, fkey_idx]
            )

    return data


class AddTargetLabelTransform:
    r"""Adds the target label to the batch. The batch consists of disjoint
    subgraphs loaded via temporal sampling. The same input node can occur twice
    with different timestamps, and thus different subgraphs and labels. Hence
    labels cannot be stored in the Data object directly, and must be attached
    to the batch after the batch is created."""

    def __init__(self, labels: list[int | float]):
        self.labels = torch.tensor(labels)

    def __call__(self, batch: pyg.data.Batch) -> pyg.data.Batch:
        batch.y = self.labels[batch.input_id]
        return batch
