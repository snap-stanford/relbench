import rtb

import torch
import torch_frame as pyf
import torch_geometric as pyg


# XXX: maybe we can abstract out a class for tabular encoder + GNN models
# and put it under rtb.models


class Net(torch.nn.Module):
    def __init__(self, node_col_stats, node_col_names_dict, hetero_metadata):
        super().__init__()

        # make node encoders (tabular encoders)
        self.encs = torch.nn.ModuleDict()
        for name in node_col_stats.keys():
            self.encs[name] = pyf.nn.models.ResNet(
                channels=64,
                out_channels=64,
                num_layers=4,
                col_stats=node_col_stats[name],
                col_names_dict=node_col_names_dict[name],
            )

        # make hetero GNN
        self.gnn = pyg.nn.to_hetero(
            pyg.nn.models.GCN(
                in_channels=64,
                hidden_channels=64,
                num_layers=2,
                out_channels=2,
            ),
            hetero_metadata(),
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


WEEK = 7 * 24 * 60 * 60


def main():
    # instantiate dataset. this downloads and processes it, if required.
    dset = rtb.get_dataset(name="mtb-product", root="data/")

    # get the task. this does not create any task tables yet.
    task = dset.tasks["ltv"]

    # will see later if we want to have one, multiple or no window_sizes
    # directly tied to the task
    # for now, window_size is supplied externally everywhere
    window_size = WEEK

    # get snapshot of database visible at train_cutoff_time
    db_train = dset.db_snapshot(dset.train_cutoff_time)

    # also we don't use the terminology of "split" because that suggests
    # partitioning, whereas here val is a superset of train,
    # and test is a superset of val. hence, our terminology is "cutoff"

    # important: node col stats should be computed only over the train set
    node_col_stats = {}
    node_col_names_dict = {}
    for name, table in db_train.tables:
        pyf_dataset = rtb.utils.to_pyf_dataset(table)
        node_col_stats[name] = pyf_dataset.col_stats
        # XXX: col_names_dict is not a pyf_dataset attribute, but maybe should be?
        node_col_names_dict[name] = pyf_dataset.col_names_dict

    # we let the user sample the train and val time windows as they please

    # here we do a rolling window, but stride=window_size means no overlap
    # can do stride=DAY for more data, for example
    time_window_df = rtb.utils.rolling_window_sampler(
        dset.min_time, dset.train_cutoff_time, window_size, stride=window_size
    )

    # create the task table
    train_table = task.create(db_train, time_window_df)

    # need the db snapshot at val_cutoff_time to create the val table
    db_val = dset.db_snapshot(dset.val_cutoff_time)
    val_table = task.create(
        db_val,
        # just one time window into the future
        # could also use rtb.utils.one_window_sampler here
        pd.DataFrame(
            {
                "offset": [dset.train_cutoff_time],
                "cutoff": [dset.train_cutoff_time + WEEK],
            }
        ),
    )

    input_node_type = train_table.fkeys.values()[0]

    # make graph only for the train snapshot for safety
    data = rtb.utils.make_graph(db_train)
    net = Net(
        node_col_stats=node_col_stats,
        node_col_names_dict=node_col_names_dict,
        hetero_metadata=data.metadata(),
    )

    opt = torch.optim.Adam(net.parameters())

    # captures the temporal sampling/masking of nodes
    train_loader = pyg.nn.NeighborLoader(
        data,
        num_neighbors=[10] * 2,
        shuffle=True,
        input_nodes=(
            input_node_type,
            torch.tensor(train_table.df[train_table.fkeys.keys()[0]]),
        ),
        input_time=torch.tensor(train_table.df[train_table.time_col]),
        time_attr="time_stamp",
        transform=rtb.utils.AddTargetLabelTransform(
            train_table.df[train_table.target_col]
        ),
    )

    val_loader = pyg.nn.NeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        shuffle=False,
        input_nodes=(
            input_node_type,
            torch.tensor(val_table.df[val_table.fkeys.keys()[0]]),
        ),
        input_time=torch.tensor(val_table.df[val_table.time_col]),
        time_attr="time_stamp",
        transform=rtb.utils.AddTargetLabelTransform(val_table.df[val_table.target_col]),
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

    # the user cannot query the final snapshot of the database directly
    # to prevert leakage of test information

    # instead, we provide a method to create a test table through the dataset
    # here the sampler is not for the user to choose
    # the time window is fixed to be [val_cutoff_time, val_cutoff_time + time_window]
    test_table = dset.get_test_table("ltv", WEEK)

    # the input graph for the test is the snapshot of the database at val_cutoff_time
    data = rtb.utils.make_graph(db_val)

    test_loader = pyg.nn.NeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        shuffle=False,
        input_nodes=(
            input_node_type,
            torch.tensor(test_table.df[test_table.fkeys.keys()[0]]),
        ),
        input_time=torch.tensor(test_table.df[test_table.time_col]),
        time_attr="time_stamp",
        # note that AddTargetLabelTransform is not used here
    )

    with torch.no_grad():
        net.eval()

        yhats = []
        for batch in test_loader:
            batch_size = batch[input_node_type].batch_size

            out = net(batch.tf_dict, batch.edge_index_dict)
            yhat = out[input_node_type][:batch_size]

            yhats.append(yhat)

        yhat = torch.cat(yhats, dim=0)

    # the test ground-truth labels are also not exposed to the user
    print(dset.evaluate("ltv", yhat))


if __name__ == "__main__":
    main()
