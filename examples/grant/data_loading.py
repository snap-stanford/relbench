from rtb.datasets.grant import GrantDataset
dset = GrantDataset(root="data/")
task_name = "institution_one_year" # investigator_three_years / program_three_years

window_size = dset.tasks_window_size[task_name] # pre-defined window size for this task

train_table = dset.make_train_table(task_name, window_size)
val_table = dset.make_val_table(task_name, window_size)
test_table = dset.make_test_table(task_name, window_size)