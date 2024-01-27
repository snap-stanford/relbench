import logging
import math
from tqdm import tqdm
import torch
from relgym.utils.checkpoint import load_ckpt, save_ckpt
from relgym.config import cfg
from relgym.utils.epoch import is_ckpt_epoch, is_eval_epoch
from relbench.data.task import TaskType


def train_epoch(loader_dict, model, optimizer, scheduler, entity_table, loss_fn):
    model.train()

    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        batch = batch.to(cfg.device)

        optimizer.zero_grad()
        pred = model(
            batch.tf_dict,
            batch.edge_index_dict,
            batch[entity_table].seed_time,
            batch.time_dict,
            batch.batch_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
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
def eval_epoch(loader_dict, model, task, entity_table, loss_fn, split='val'):
    model.eval()

    pred_list = []
    for batch in tqdm(loader_dict[split]):
        batch = batch.to(cfg.device)
        pred = model(
            batch.tf_dict,
            batch.edge_index_dict,
            batch[entity_table].seed_time,
            batch.time_dict,
            batch.batch_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = torch.sigmoid(pred)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    all_pred = torch.cat(pred_list, dim=0).numpy()
    if split == 'val':
        metrics = task.evaluate(all_pred, task.val_table)
    elif split == 'test':
        metrics = task.evaluate(all_pred)
    else:
        raise RuntimeError(f'split should be val or test, got {split}')

    return metrics


def train(loader_dict, model, optimizer, scheduler, task, entity_table, loss_fn):
    r"""
    The core training pipeline

    Args:
        loader_dict: Dict of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    # if cfg.train.auto_resume:
    #     start_epoch = load_ckpt(model, optimizer, scheduler)
    #     logging.info('Start from epoch {}'.format(start_epoch))

    best_val_metric = 0 if cfg.higher_is_better else math.inf
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_loss = train_epoch(loader_dict, model, optimizer, scheduler, entity_table, loss_fn)
        logging.info(f"Epoch: {cur_epoch:02d}, Train loss: {train_loss}")
        if is_eval_epoch(cur_epoch):
            metrics = eval_epoch(loader_dict, model, task, entity_table, loss_fn, split='val')
            logging.info(f"Val metrics: {metrics}")
            cur_val_metric = metrics[cfg.tune_metric]
            # Save the best model
            if (cfg.higher_is_better and cur_val_metric > best_val_metric) or (
                    not cfg.higher_is_better and cur_val_metric < best_val_metric):
                save_ckpt(model, optimizer, scheduler, best=True)
                best_val_metric = cur_val_metric
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

    # Test, load the best model
    load_ckpt(model, best=True)
    logging.info("Model loaded for evaluation")
    metrics = eval_epoch(loader_dict, model, task, entity_table, loss_fn, split='val')
    logging.info(f"Val metrics: {metrics}")
    metrics = eval_epoch(loader_dict, model, task, entity_table, loss_fn, split='test')
    logging.info(f"Test metrics: {metrics}")

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))
