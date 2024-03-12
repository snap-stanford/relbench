import argparse
import logging
import sys

from torch_geometric import seed_everything

sys.path.append("./")

from relbench.data.task_base import TaskType
from relgym.config import cfg, dump_cfg, load_cfg, set_out_dir, set_run_dir
from relgym.loader import create_loader, create_dataset_and_task, transform_dataset_to_graph
from relgym.logger import setup_printing
from relgym.loss import create_loss_fn
from relgym.models.model_builder import create_model
from relgym.optimizer import create_optimizer, create_scheduler
from relgym.train import train

from relgym.utils.comp_budget import params_count
from relgym.utils.device import auto_select_device


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description="RelGym")

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        type=str,
        required=True,
        help="The configuration file path.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="The number of repeated jobs."
    )
    parser.add_argument(
        "--mark_done",
        action="store_true",
        help="Mark yaml as done after a job has finished.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="See graphgym/config.py for remaining options.",
    )
    parser.add_argument(
        "--auto_select_device",
        action="store_true",
        help="Automatically select gpu for training.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    # torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    if args.auto_select_device:
        auto_select_device()
    else:
        cfg.device = "cuda"
    # Load dataset
    dataset, task = create_dataset_and_task()
    data, col_stats_dict = transform_dataset_to_graph(dataset)
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        setup_printing()
        # Set configurations for each run
        seed_everything(cfg.seed)
        # Set machine learning pipeline
        loader_dict = create_loader(data, task)
        model = create_model(
            data=data,
            col_stats_dict=col_stats_dict,
            task=task,
            to_device=cfg.device,
            shallow_list=[
                task.dst_entity_table
            ] if cfg.model.use_shallow and task.task_type == TaskType.LINK_PREDICTION else [],
        )
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)
        loss_fn, loss_utils = create_loss_fn(task)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info("Num parameters: %s", cfg.params)
        # Start training
        train(
            loader_dict,
            model,
            optimizer,
            scheduler,
            task,
            loss_fn,
            loss_utils,
        )
        logging.info(f"Complete trial {i}")
        cfg.seed = cfg.seed + 1

