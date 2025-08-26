import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def create_optimizer(optimizer_config: dict, model: nn.Module) -> optim.Optimizer:
    """
    Create an optimizer for model parameters based on configuration.

    :param optimizer_config: Dictionary containing:
        - name (str): Name of the torch.optim optimizer class (e.g., 'Adam', 'SGD').
        - parameters (dict): Keyword arguments for optimizer initialization (e.g., lr, weight_decay).
    :param model: The model whose parameters will be optimized.
    :return: An optimizer instance from torch.optim.

    :raises ValueError: If the specified optimizer class does not exist in torch.optim.
    """
    opt_name = optimizer_config.get("name")
    params = optimizer_config.get("parameters", {})
    if not hasattr(optim, opt_name):
        raise ValueError(f"Optimizer {opt_name} does not exist in torch.optim.")
    opt_cls = getattr(optim, opt_name)
    return opt_cls(model.parameters(), **params)


def create_scheduler(
    scheduler_config: dict, optimizer: optim.Optimizer
) -> lr_scheduler._LRScheduler | None:
    """
    Create a learning rate scheduler based on configuration.

    :param scheduler_config: Dictionary containing:
        - name (str | None): Name of the torch.optim.lr_scheduler class (e.g., 'StepLR', 'CosineAnnealingLR'), or None.
        - parameters (dict): Keyword arguments for scheduler initialization (e.g., step_size, gamma).
    :param optimizer: Optimizer instance to attach scheduler to.
    :return: Scheduler instance or None if no scheduler specified.

    :raises ValueError: If the specified scheduler class does not exist in torch.optim.lr_scheduler.
    """
    sched_name = scheduler_config.get("name")
    if sched_name is None:
        return None
    params = scheduler_config.get("parameters", {})
    if not hasattr(lr_scheduler, sched_name):
        raise ValueError(
            f"Scheduler {sched_name} does not exist in torch.optim.lr_scheduler."
        )
    sched_cls = getattr(lr_scheduler, sched_name)
    return sched_cls(optimizer, **params)
