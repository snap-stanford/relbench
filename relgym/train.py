import logging
import math

import torch
from tqdm import tqdm

from relbench.data.task_base import TaskType
from relgym.config import cfg
from relgym.utils.checkpoint import load_ckpt, save_ckpt
from relgym.utils.epoch import is_ckpt_epoch, is_eval_epoch


def train_epoch(
    loader_dict, model, optimizer, scheduler, entity_table, loss_fn, loss_utils
):
    model.train()

    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        batch = batch.to(cfg.device)

        if cfg.model.use_self_join:  # Use the re-sampling method for retrieval bank
            bank_batch = next(loader_dict["bank"]).to(cfg.device)
        else:
            bank_batch = None

        optimizer.zero_grad()
        pred = model(
            batch.tf_dict,
            batch.edge_index_dict,
            batch[entity_table].seed_time,
            batch.time_dict,
            batch.batch_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
            batch[entity_table].y,  # used in SelfJoin with memory bank
            bank_batch,  # used in SelfJoin with re-sampling
        )

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        loss = loss_fn(pred, batch[entity_table].y)

        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    scheduler.step()
    return loss_accum / count_accum


@torch.no_grad()
def eval_epoch(
    loader_dict, model, task, entity_table, loss_fn, loss_utils, split="val"
):
    model.eval()

    pred_list = []
    for batch in tqdm(loader_dict[split]):
        batch = batch.to(cfg.device)

        if cfg.model.use_self_join:  # Use the re-sampling method for retrieval bank
            bank_batch = next(loader_dict["bank"]).to(cfg.device)
        else:
            bank_batch = None

        pred = model(
            batch.tf_dict,
            batch.edge_index_dict,
            batch[entity_table].seed_time,
            batch.time_dict,
            batch.batch_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
            None,  # y is not used in eval for SelfJoin with memory bank
            bank_batch,  # used in SelfJoin with re-sampling
        )

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = torch.sigmoid(pred)
        elif task.task_type == TaskType.REGRESSION:
            pred = torch.clamp(pred, loss_utils["clamp_min"], loss_utils["clamp_max"])

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    all_pred = torch.cat(pred_list, dim=0).numpy()
    if split == "val":
        metrics = task.evaluate(all_pred, task.val_table)
    elif split == "test":
        metrics = task.evaluate(all_pred)
    else:
        raise RuntimeError(f"split should be val or test, got {split}")

    return metrics


def train(
    loader_dict, model, optimizer, scheduler, task, entity_table, loss_fn, loss_utils
):
    r"""
    The core training pipeline

    Args:
        loader_dict: Dict of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    early_stop_counter = 0
    # if cfg.train.auto_resume:
    #     start_epoch = load_ckpt(model, optimizer, scheduler)
    #     logging.info('Start from epoch {}'.format(start_epoch))

    best_val_metric = 0 if cfg.higher_is_better else math.inf
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_loss = train_epoch(
            loader_dict, model, optimizer, scheduler, entity_table, loss_fn, loss_utils
        )
        logging.info(f"Epoch: {cur_epoch:02d}, Train loss: {train_loss}")
        if is_eval_epoch(cur_epoch):
            metrics = eval_epoch(
                loader_dict, model, task, entity_table, loss_fn, loss_utils, split="val"
            )
            logging.info(f"Val metrics: {metrics}")
            cur_val_metric = metrics[cfg.tune_metric]
            # Save the best model
            if (cfg.higher_is_better and cur_val_metric > best_val_metric) or (
                not cfg.higher_is_better and cur_val_metric < best_val_metric
            ):
                save_ckpt(model, optimizer, scheduler, best=True)
                best_val_metric = cur_val_metric
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            if (
                cfg.optim.early_stop is not None
                and early_stop_counter >= cfg.optim.early_stop
            ):
                logging.info(
                    f"Early stopped with counter {early_stop_counter} at epoch {cur_epoch}"
                )
                break
            else:
                logging.info(f"Early stop counter {early_stop_counter}")

        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

    # Test, load the best model
    load_ckpt(model, best=True)
    logging.info("Model loaded for evaluation")
    metrics = eval_epoch(
        loader_dict, model, task, entity_table, loss_fn, loss_utils, split="val"
    )
    logging.info(f"Val metrics: {metrics}")
    metrics = eval_epoch(
        loader_dict, model, task, entity_table, loss_fn, loss_utils, split="test"
    )
    logging.info(f"Test metrics: {metrics}")

    logging.info("Task done, results saved in {}".format(cfg.out_dir))
