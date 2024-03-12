import logging
import math
from typing import List

import torch
from torch import Tensor
from tqdm import tqdm
import torch.nn.functional as F

from relbench.data.task_base import TaskType
from relgym.config import cfg
from relgym.utils.checkpoint import load_ckpt, save_ckpt
from relgym.utils.epoch import is_ckpt_epoch, is_eval_epoch


def train_epoch_node(
    loader_dict, model, task, optimizer, scheduler, loss_fn, loss_utils
):
    model.train()

    entity_table = task.entity_table
    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        batch = batch.to(cfg.device)

        optimizer.zero_grad()
        pred = model(
            batch,
            entity_table,
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
def eval_epoch_node(
    loader_dict, model, task, loss_fn, loss_utils, split="val"
):
    model.eval()

    entity_table = task.entity_table

    pred_list = []
    for batch in tqdm(loader_dict[split]):
        batch = batch.to(cfg.device)

        pred = model(
            batch,
            entity_table,
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


def train_epoch_link(
    loader_dict, model, task, optimizer, scheduler, loss_fn, loss_utils
):
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    train_loader = loader_dict["train"]
    total_steps = min(len(train_loader), cfg.train.max_steps_per_epoch)
    for batch in tqdm(train_loader, total=total_steps):
        src_batch, batch_pos_dst, batch_neg_dst = batch
        src_batch, batch_pos_dst, batch_neg_dst = (
            src_batch.to(cfg.device),
            batch_pos_dst.to(cfg.device),
            batch_neg_dst.to(cfg.device),
        )
        x_src = model(src_batch, task.src_entity_table)
        x_pos_dst = model(batch_pos_dst, task.dst_entity_table)
        x_neg_dst = model(batch_neg_dst, task.dst_entity_table)

        # [batch_size, ]
        pos_score = torch.sum(x_src * x_pos_dst, dim=1)
        if cfg.loader.share_same_time:
            # [batch_size, batch_size]
            neg_score = x_src @ x_neg_dst.t()
            # [batch_size, 1]
            pos_score = pos_score.view(-1, 1)
        else:
            # [batch_size, ]
            neg_score = torch.sum(x_src * x_neg_dst, dim=1)
        optimizer.zero_grad()
        # BPR loss
        diff_score = pos_score - neg_score
        loss = F.softplus(-diff_score).mean()
        loss.backward()
        optimizer.step()

        loss_accum += float(loss) * x_src.size(0)
        count_accum += x_src.size(0)

        steps += 1
        if steps > cfg.train.max_steps_per_epoch:
            break

    return loss_accum / count_accum


@torch.no_grad()
def eval_epoch_link(
    loader_dict, model, task, loss_fn, loss_utils, split="val"
):
    model.eval()
    src_loader, dst_loader = loader_dict[split]

    dst_embs: List[Tensor] = []
    for batch in tqdm(dst_loader):
        batch = batch.to(cfg.device)
        emb = model(batch, task.dst_entity_table).detach()
        dst_embs.append(emb)
    dst_emb = torch.cat(dst_embs, dim=0)
    del dst_embs

    pred_index_mat_list: List[Tensor] = []
    for batch in tqdm(src_loader):
        batch = batch.to(cfg.device)
        emb = model(batch, task.src_entity_table)
        _, pred_index_mat = torch.topk(emb @ dst_emb.t(), k=task.eval_k, dim=1)
        pred_index_mat_list.append(pred_index_mat.cpu())
    pred = torch.cat(pred_index_mat_list, dim=0).numpy()

    if split == "val":
        metrics = task.evaluate(pred, task.val_table)
    elif split == "test":
        metrics = task.evaluate(pred)
    else:
        raise RuntimeError(f"split should be val or test, got {split}")

    return metrics


def train(
    loader_dict, model, optimizer, scheduler, task, loss_fn, loss_utils
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

    if task.task_type in [TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION]:
        train_epoch_fn, eval_epoch_fn = train_epoch_node, eval_epoch_node
    elif task.task_type == TaskType.LINK_PREDICTION:
        train_epoch_fn, eval_epoch_fn = train_epoch_link, eval_epoch_link
    else:
        raise NotImplementedError(task.task_type)

    best_val_metric = 0 if cfg.higher_is_better else math.inf
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_loss = train_epoch_fn(
            loader_dict, model, task, optimizer, scheduler, loss_fn, loss_utils
        )
        logging.info(f"Epoch: {cur_epoch:02d}, Train loss: {train_loss}")
        if is_eval_epoch(cur_epoch):
            metrics = eval_epoch_fn(
                loader_dict, model, task, loss_fn, loss_utils, split="val"
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
    metrics = eval_epoch_fn(
        loader_dict, model, task, loss_fn, loss_utils, split="val"
    )
    logging.info(f"Val metrics: {metrics}")
    metrics = eval_epoch_fn(
        loader_dict, model, task, loss_fn, loss_utils, split="test"
    )
    logging.info(f"Test metrics: {metrics}")

    logging.info("Task done, results saved in {}".format(cfg.out_dir))
