import logging
import os
import sys

from relgym.config import cfg


def setup_printing():
    """
    Set up printing options

    """
    logging.root.handlers = []
    logging_cfg = {"level": logging.INFO, "format": "%(message)s"}

    def handle_exception(exc_type, exc_value, exc_traceback):
        import sys

        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical("Exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    os.makedirs(cfg.run_dir, exist_ok=True)
    h_file = logging.FileHandler("{}/logging.log".format(cfg.run_dir))
    h_stdout = logging.StreamHandler(sys.stdout)
    if cfg.print == "file":
        logging_cfg["handlers"] = [h_file]
    elif cfg.print == "stdout":
        logging_cfg["handlers"] = [h_stdout]
    elif cfg.print == "both":
        logging_cfg["handlers"] = [h_file, h_stdout]
    else:
        raise ValueError("Print option not supported")
    logging.basicConfig(**logging_cfg)
