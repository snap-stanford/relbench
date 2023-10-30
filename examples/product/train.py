import rtb

import torch
import torch_frame as pyf
import torch_geometric as pyg


# XXX: maybe we can abstract out a class for tabular encoder + GNN models
# and put it under rtb.nn.models


class Net(torch.nn.Module):
    def __init__(self, data: pyg.data.HeteroData):
        r"""The intended use is for data obtained from rtb.nn.make_graph.
        Hence, it is assumed that data[name].tf and data[name].col_stats
        attributes exist."""

        super().__init__()

        # make node encoders (tabular encoders)
        self.encs = torch.nn.ModuleDict()
        for name in data.node_types:
            self.encs[name] = pyf.nn.models.ResNet(
                channels=64,
                out_channels=64,
                num_layers=4,
                col_stats=data[name].col_stats,
                col_names_dict=data[name].tf.col_names_dict,
            )

        # make hetero GNN
        self.gnn = pyg.nn.to_hetero(
            pyg.nn.models.GCN(
                in_channels=64,
                hidden_channels=64,
                num_layers=2,
                out_channels=2,
            ),
            data.metadata(),
        )

    def forward(
        self,
        tf_dict: dict[str, pyf.data.TensorFrame],
        edge_index_dict: dict[str, torch.Tensor],
    ):
        # encode node features from tensor frames
        x_dict = {}
        for name, tf in tf_dict.items():
            x_dict[name] = self.encs[name](tf)

        # run GNN
        return self.gnn(x_dict, edge_index_dict)


def main():
    dataset = rtb.get_dataset(name="mtb-product", task_names=["churn"], root="data/")

    # at their own risk, users can merge the splits and make a graph directly
    # not really an issue since temporal sampling should not violate the splits anyway
    db_train = dataset.db_splits["train"]
    db_val = dataset.db_splits["val"]
    db = db_train + db_val

    task_train = dataset.task_splits["churn"]["train"]
    task_val = dataset.task_splits["churn"]["val"]
    input_node_type = task_train.entities.keys()[0]

    data = rtb.nn.make_graph(db)
    net = Net(data)

    opt = torch.optim.Adam(net.parameters())

    # captures the temporal handling
    train_loader = pyg.nn.NeighborLoader(
        data,
        num_neighbors=[10] * 2,
        shuffle=True,
        input_nodes=(
            input_node_type,
            torch.tensor(task_train.entities[input_node_type]),
        ),
        input_time=torch.tensor(task_train.time_stamps),
        time_attr=db.ctime_col,
        transform=rtb.nn.AddTargetLabelTransform(task_train.labels),
    )

    val_loader = pyg.nn.NeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        shuffle=False,
        input_nodes=(
            input_node_type,
            torch.tensor(task_val.entities[input_node_type]),
        ),
        input_time=torch.tensor(task_val.time_stamps),
        time_attr=db.ctime_col,
        transform=rtb.nn.AddTargetLabelTransform(task_val.labels),
    )

    for epoch in range(100):
        # train
        net.train()
        for batch in train_loader:
            batch_size = batch[input_node_type].batch_size

            out = net(batch.tf_dict, batch.edge_index_dict)
            yhat = out[input_node_type][:batch_size]
            loss = F.cross_entropy(yhat, batch.y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # validate
        with torch.no_grad():
            net.eval()
            for batch in val_loader:
                batch_size = batch[input_node_type].batch_size

                out = net(batch.tf_dict, batch.edge_index_dict)
                yhat = out[input_node_type][:batch_size]

                total_examples += batch_size
                total_correct += int((yhat.argmax(-1) == batch.y).sum())

        print(f"Epoch {epoch}: accuracy={total_correct / total_examples}")


if __name__ == "__main__":
    main()
