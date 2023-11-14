import argparse

import rtb
from rtb.datasets import GrantDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="institution_one_year",
    choices=[
        "institution_one_year",
        "investigator_three_years",
        "program_three_years",
    ],
)
args = parser.parse_args()

dataset = GrantDataset(root="./data/")

# Pre-defined window size for this task:
window_size = dataset.tasks_window_size[args.task]
# TODO `window_size` should be 365 days but is 365 days + 6 hours.

train_table = dataset.make_train_table(args.task, window_size)
val_table = dataset.make_val_table(args.task, window_size)
test_table = dataset.make_test_table(args.task, window_size)

data = rtb.utils.make_pkey_fkey_graph(dataset._db)
