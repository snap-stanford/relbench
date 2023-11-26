from relbench.datasets import dataset_names, get_dataset
dataset = get_dataset(name="stack_exchange")
print(dataset.task_names)
# ['user_contribution', 'question_popularity']
task = dataset.get_task("user_contribution")
# task.train_table, task.val_table, task.test_table

import numpy as np
pred = np.array([0] * len(task.test_table.df))
task.evaluate(pred)
#{'accuracy': 0.9258449514152937, 'f1': 0.0, 'roc_auc': 0.5}