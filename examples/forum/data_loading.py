from rtb.datasets.forum import ForumDataset

dset = ForumDataset(root="/dfs/user/kexinh/rtb/data/")
task_name = "UserSumCommentScoresTask"  # PostUpvotesTask / UserNumPostsTask

window_size = dset.tasks_window_size[task_name]  # pre-defined window size for this task

train_table = dset.make_train_table(task_name, window_size)
val_table = dset.make_val_table(task_name, window_size)
test_table = dset.make_test_table(task_name, window_size)
