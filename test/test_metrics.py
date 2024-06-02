import numpy as np

from relbench.metrics import (
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
)


def test_link_prediction_metrics():
    num_src_nodes = 100
    eval_k = 10
    pred_isin = np.random.randint(0, 2, size=(num_src_nodes, eval_k), dtype=bool)
    dst_count = pred_isin.sum(axis=1) + np.random.randint(0, 5, size=(num_src_nodes,))
    recall = link_prediction_recall(pred_isin, dst_count)
    precision = link_prediction_precision(pred_isin, dst_count)
    map = link_prediction_map(pred_isin, dst_count)
    assert 0 <= recall <= 1
    assert 0 <= precision <= 1
    assert 0 <= map <= 1
