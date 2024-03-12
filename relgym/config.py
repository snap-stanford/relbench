import os
import shutil

from yacs.config import CfgNode as CN

# Global config object
cfg = CN()


def set_cfg(cfg):
    r"""
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name

    :return: configuration use by the experiment.
    """
    if cfg is None:
        return cfg

    # ----------------------------------------------------------------------- #
    # Basic options
    # ----------------------------------------------------------------------- #

    # Set print destination: stdout / file / both
    cfg.print = "both"

    # Select device: 'cpu', 'cuda:0', 'auto'
    cfg.device = "auto"

    # Output directory
    cfg.out_dir = "results"

    # Config name (in out_dir)
    cfg.cfg_dest = "config.yaml"

    # Random seed
    cfg.seed = 42

    # Max threads used by PyTorch
    cfg.num_threads = 6

    # ----------------------------------------------------------------------- #
    # Loader options
    # ----------------------------------------------------------------------- #
    cfg.loader = CN()

    # Number of neighbors in NeighborSampler
    cfg.loader.num_neighbors = None

    # Number of workers for loader
    cfg.loader.num_workers = None

    # Batch size
    cfg.loader.batch_size = None

    # Temporal strategy in 'uniform', 'last'
    cfg.loader.temporal_strategy = "uniform"

    # Share same time for mini batch to enable shared negative samples in link prediction
    cfg.loader.share_same_time = False

    # ----------------------------------------------------------------------- #
    # Dataset options
    # ----------------------------------------------------------------------- #
    cfg.dataset = CN()

    # Name of the dataset
    cfg.dataset.name = None

    # Name of the task
    cfg.dataset.task = None

    # The cache directory
    cfg.dataset.cache_dir = None

    # The root directory for torch frame embeddings
    cfg.dataset.root_dir = None

    # ----------------------------------------------------------------------- #
    # Training options
    # ----------------------------------------------------------------------- #
    cfg.train = CN()

    # Whether resume from existing experiment
    cfg.train.auto_resume = False

    # The period for evaluation
    cfg.train.eval_period = 1

    # The period for checkpoint saving
    cfg.train.ckpt_period = 40

    # The maximum step per epoch in link prediction
    cfg.train.max_steps_per_epoch = 2000

    # ----------------------------------------------------------------------- #
    # Validation options
    # ----------------------------------------------------------------------- #
    cfg.val = CN()

    # ----------------------------------------------------------------------- #
    # Torch Frame Model options
    # ----------------------------------------------------------------------- #
    cfg.torch_frame_model = CN()

    # The hidden channels for the model
    cfg.torch_frame_model.channels = 128

    # The number of torch frame model layers
    cfg.torch_frame_model.num_layers = 2

    # The model class
    cfg.torch_frame_model.torch_frame_model_cls = "ResNet"

    # test embedder
    cfg.torch_frame_model.text_embedder = "glove"

    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    cfg.model = CN()

    # The hidden channels for the model
    cfg.model.channels = 128

    # The output channels for the model
    cfg.model.out_channels = 1

    # The aggregation method of GNN message passing
    cfg.model.aggr = "sum"

    # The aggregation method of the heterogeneous information
    cfg.model.hetero_aggr = "sum"

    # The graph convolution operation, in ['sage', 'gat', 'gc']
    cfg.model.conv = None

    # The number of conv layers
    cfg.model.num_layers = 2

    # Perturb input edges, in [None, 'drop_all', 'rand_perm']
    cfg.model.perturb_edges = None

    # Feature dropout
    cfg.model.feature_dropout = 0.0

    # Mask input feature
    cfg.model.mask_features = False

    # Norm type for the prediction head
    cfg.model.norm = "batch_norm"

    # Use shallow embedding for destination nodes in link prediction
    cfg.model.use_shallow = False

    # ----------------------------------------------------------------------- #
    # Optimizer options
    # ----------------------------------------------------------------------- #
    cfg.optim = CN()

    # optimizer: sgd, adam
    cfg.optim.optimizer = "adam"

    # Base learning rate
    cfg.optim.base_lr = 0.01

    # L2 regularization
    cfg.optim.weight_decay = 0.0

    # SGD momentum
    cfg.optim.momentum = None

    # scheduler: none, steps, cos
    cfg.optim.scheduler = "none"

    # Steps for 'steps' policy (in epochs)
    cfg.optim.steps = [30, 60, 90]

    # Learning rate multiplier for 'steps' policy
    cfg.optim.lr_decay = 0.1

    # Maximal number of epochs
    cfg.optim.max_epoch = 20

    # Number of earlystopping counter
    cfg.optim.early_stop = None


def dump_cfg(cfg):
    r"""
    Dumps the config to the output directory specified in
    :obj:`cfg.out_dir`

    Args:
        cfg (CfgNode): Configuration node

    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg_file = os.path.join(cfg.out_dir, cfg.cfg_dest)
    with open(cfg_file, "w") as f:
        cfg.dump(stream=f)


def load_cfg(cfg, args):
    r"""
    Load configurations from file system and command line

    Args:
        cfg (CfgNode): Configuration node
        args (ArgumentParser): Command argument parser

    """
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    # assert_cfg(cfg)


def makedirs_rm_exist(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def get_fname(fname):
    r"""
    Extract filename from file name path

    Args:
        fname (string): Filename for the yaml format configuration file
    """
    fname = fname.split("/")[-1]
    if fname.endswith(".yaml"):
        fname = fname[:-5]
    elif fname.endswith(".yml"):
        fname = fname[:-4]
    return fname


def set_out_dir(out_dir, fname):
    r"""
    Create the directory for full experiment run

    Args:
        out_dir (string): Directory for output, specified in :obj:`cfg.out_dir`
        fname (string): Filename for the yaml format configuration file

    """
    fname = get_fname(fname)
    cfg.out_dir = os.path.join(out_dir, fname)
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.out_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.out_dir)


def set_run_dir(out_dir):
    r"""
    Create the directory for each random seed experiment run

    Args:
        out_dir (string): Directory for output, specified in :obj:`cfg.out_dir`
        fname (string): Filename for the yaml format configuration file

    """
    cfg.run_dir = os.path.join(out_dir, str(cfg.seed))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


# 1. set default cfg values first, some modules in contrib might rely on this
set_cfg(cfg)
