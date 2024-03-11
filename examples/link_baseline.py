import argparse
import random
from collections import Counter

import duckdb
import numpy as np
import pandas as pd

from relbench.datasets import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stackex")
parser.add_argument("--task", type=str, default="rel-stackex-comment-on-post")
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--method", type=str, default="most_frequent")
args = parser.parse_args()

dataset = get_dataset(name=args.dataset, process=True)
task = dataset.get_task(args.task, process=True)

train_table = task.train_table
test_table = task.test_table
val_table = task.val_table
PAD = -1


# Count the frequency of each int in the list
def count_and_sort(int_list):
    # Count the frequency of each int
    freq_counter = Counter(int_list)

    # Sort the ints by frequency in descending order
    sorted_ints = sorted(freq_counter, key=lambda x: freq_counter[x], reverse=True)

    # Return the sorted list of unique ints by frequency
    return sorted_ints


def count_sort_and_pad_preds(max_len, preds):
    padded_preds = []

    max_len = task.eval_k
    for p in preds:
        p = count_and_sort(p)
        if len(p) < max_len:
            p = p + (max_len - len(p)) * [PAD]
        else:
            p = p[:max_len]
        padded_preds.append(p)

    return padded_preds


# =============================================================================
# predict the most frequent dst entity for each src entity
# =============================================================================
def most_frequent_dst_entity(train_table, test_table, val_table, task):
    src_ids = list(task.train_table.df[task.src_entity_col])

    src_to_dst = {}

    for index, row in train_table.df.iterrows():
        src = row[task.src_entity_col]
        dst = row[task.dst_entity_col]
        if src not in src_to_dst:
            src_to_dst[src] = []
        src_to_dst[src] += dst

    for index, row in val_table.df.iterrows():
        src = row[task.src_entity_col]
        dst = row[task.dst_entity_col]
        if src not in src_to_dst:
            src_to_dst[src] = []
        src_to_dst[src] += dst

    preds = []
    for id in list(task.test_table.df[task.src_entity_col]):
        if id in src_to_dst:
            pred = src_to_dst[id]
        else:
            pred = (
                []
            )  # If the src entity is not in the train or val set, predict an empty list

        preds.append(pred)

    padded_preds = count_sort_and_pad_preds(task.eval_k, preds)
    preds = np.array(padded_preds)
    return preds


# =============================================================================
# predict random dst entities for each src entity
# =============================================================================
def random_dst_entity(train_table, test_table, val_table, task):
    src_ids = list(task.test_table.df[task.src_entity_col])
    max_len = task.eval_k

    dst_list = [
        item
        for sublist in list(train_table.df[task.dst_entity_col])
        for item in sublist
    ]
    dst_list += [
        item for sublist in list(val_table.df[task.dst_entity_col]) for item in sublist
    ]
    dst_list = list(set(dst_list))  # get unique ids

    preds = []
    for id in src_ids:
        pred = random.sample(dst_list, max_len)

        preds.append(pred)

    padded_preds = count_sort_and_pad_preds(task.eval_k, preds)
    preds = np.array(padded_preds)

    return preds


for run in range(args.repeats):
    if args.method == "most_frequent":
        preds = most_frequent_dst_entity(train_table, test_table, val_table, task)
    elif args.method == "random":
        preds = random_dst_entity(train_table, test_table, val_table, task)

    print(f"Run: {run}")
    print(f"Method: {args.method}")
    print(task.evaluate(preds))
