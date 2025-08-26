import json
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import List

import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils.constants import TransformCategoryConstants

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """
    Create a directory and all necessary parent directories if they do not exist.

    :param path: Path object representing the directory to ensure.
    """
    path.mkdir(parents=True, exist_ok=True)


def read_yaml(fpath=None) -> DictConfig:
    """
    Read a YAML configuration file into an OmegaConf DictConfig object.

    :param fpath: Path to the YAML file.
    :return: DictConfig containing the parsed YAML contents.
    """
    cfg = OmegaConf.load(str(fpath))
    return cfg


def load_loggers(cfg) -> List[TensorBoardLogger]:
    """
    Set up TensorBoard logging directories and instantiate tensorboard loggers.

    The log path is formed as: cfg.General.log_path / <config_parent> / <config_stem>.
    A subfolder 'tensorboard_logs' is used for the TensorBoardLogger.

    :param cfg: Configuration object with attributes:
                - General.log_path: Base directory for logs.
                - config: Path string to the config file.
    :return: List containing a single TensorBoardLogger instance.
    """
    base_log = Path(cfg.General.log_path)
    ensure_dir(base_log)

    config_path = Path(cfg.config)
    version = config_path.stem  # automatically strips suffix
    run_dir = base_log / config_path.parent.name / version

    # Add the timestamp suffix (current date, minutes, and seconds)
    if cfg.General.server == "train":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = run_dir / f"{version}_{timestamp}"

    ensure_dir(run_dir)

    cfg.log_path = str(run_dir)

    logger.info(f"Logs folder: {run_dir}")

    params_file_to_save = run_dir / "config_used.json"

    cfg_dict_to_save = OmegaConf.to_container(cfg, resolve=True)
    # Save the dictionary to the JSON file
    with open(params_file_to_save, "w") as f:
        json.dump(cfg_dict_to_save, f, indent=2)

    tb = TensorBoardLogger(
        save_dir=str(run_dir / "tensorboard_logs"),
        log_graph=False,
        default_hp_metric=False,
    )

    return [tb]


# ---->load Callback
def load_callbacks(cfg):
    """
    Create PyTorch Lightning callbacks for model checkpointing and learning rate monitoring.

    :param cfg: Configuration object with attributes:
                - log_path: Directory where checkpoints will be saved.
                - General.server: String indicating runtime mode ('train' for training).
                - Scheduler.name: Name of the LR scheduler (optional).
    :return: List of instantiated callback objects.
    """
    Mycallbacks = []

    # Make output path
    output_path = cfg.log_path
    Path(output_path).mkdir(exist_ok=True, parents=True)

    if cfg.General.server == "train":
        Mycallbacks.append(
            ModelCheckpoint(
                monitor=None,
                dirpath=str(cfg.log_path),
                verbose=True,
                save_last=True,
                save_top_k=0,
            )
        )
        if cfg.Scheduler.name is not None:
            Mycallbacks.append(LearningRateMonitor())

    return Mycallbacks


def check_parameters_validity(transformation_params: dict | DictConfig) -> None:
    """
    Validate augmentation parameter ranges for discrete and continuous transforms.

    Discrete transforms must be float in [0,1] or int in {0,1}.
    Continuous transforms must be list/tuple of two numbers in [-0.5, 0.5] with min ≤ max.

    :param transformation_params: Dict mapping transform names to parameter values.
    :raises KeyError: If a transform name is unrecognized.
    :raises ValueError: If a parameter value is outside its valid range.
    """
    discrete_keys = TransformCategoryConstants.DISCRETE_TRANSFORMATIONS.value
    continuous_keys = TransformCategoryConstants.CONTINOUS_TRANSFORMATIONS.value

    for k, v in transformation_params.items():
        if k in discrete_keys:
            if not (
                (isinstance(v, float) and 0.0 <= v <= 1.0)
                or (isinstance(v, int) and v in (0, 1))
            ):
                raise ValueError(
                    f"Discrete transformation '{k}' must be between 0 and 1 (float or int 0/1), got {v!r}"
                )

        elif k in continuous_keys:
            # must be a list or tuple of two items
            if not isinstance(v, (list, tuple, ListConfig)) or len(v) != 2:
                raise ValueError(
                    f"Continuous transformation '{k}' must be a list or tuple of two numbers."
                )
            # each element must be float in [-0.5, 0.5]
            if not all(
                (
                    (isinstance(i, float) or (isinstance(i, int) and i == 0))
                    and -0.5 <= i <= 0.5
                )
                for i in v
            ):
                raise ValueError(
                    f"Continuous transformation '{k}' values must be floats between -0.5 and 0.5, got {v}"
                )
            # ensure min ≤ max
            if v[0] > v[1]:
                raise ValueError(
                    f"Continuous transformation '{k}' must have min <= max, got {v}"
                )

        else:
            raise KeyError(f"Transformation '{k}' is not recognized.")


def print_run_summary(cfg, color_mode: str = "auto") -> None:
    """
    Colorized summary of cfg.
    """
    force = color_mode == "always"
    never = color_mode == "never"
    use_colors = (sys.stdout.isatty() or force) and not never

    RESET = "\033[0m" if use_colors else ""
    BOLD = "\033[1m" if use_colors else ""
    WHITE = "\033[97m" if use_colors else ""
    CYAN = "\033[96m" if use_colors else ""
    GREEN = "\033[92m" if use_colors else ""
    TOPLEVEL_COLOR = "\033[94m" if use_colors else ""

    SKIP_KEYS = {"config", "log_path", "load_loggers", "callbacks"}

    def section(title: str, indent: int = 0):
        pad = "  " * indent
        color = TOPLEVEL_COLOR if indent == 0 else CYAN
        print(f"{pad}{BOLD}{color}{title}:{RESET}")

    def kv(key: str, value, indent: int = 0, val_color: str = GREEN):
        pad = "  " * indent
        # Keep sequences as a single list representation; don't expand items.
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            rendered = pformat(list(value), width=120, compact=True)
        else:
            rendered = pformat(value, width=120, compact=True)
        print(f"{pad}{BOLD}{WHITE}{key}{RESET}: {val_color}{rendered}{RESET}")

    def as_mapping(obj):
        """Treat dicts or objects-with-__dict__ as mappings; preserve insertion order."""
        if isinstance(obj, Mapping):
            return True, obj
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            clean = {
                k: v for k, v in d.items() if not k.startswith("_") and not callable(v)
            }
            return True, clean
        return False, None

    def recurse(key, value, indent: int):
        if key in SKIP_KEYS:
            return
        is_map, mapobj = as_mapping(value)
        if is_map:
            section(key, indent)
            for k, v in mapobj.items():
                if k in SKIP_KEYS:
                    continue
                recurse(k, v, indent + 1)
        else:
            kv(key, value, indent)

    # Start at root
    is_root_map, root = as_mapping(cfg)
    if is_root_map:
        for top_key, top_val in root.items():
            if top_key in SKIP_KEYS:
                continue
            recurse(top_key, top_val, indent=0)
    else:
        kv("cfg", cfg, indent=0)
