from typing import List, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import flwr as fl


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """Function to convert model to list of numpy arrays of the weights."""
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    """Function to load parameter numpy list to model."""
    params = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params})
    net.load_state_dict(state_dict=state_dict, strict=True)


def on_fit_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return a config for a client"""
    config: Dict[str, fl.common.Scalar] = {
        "server_round": server_round,
        "epochs": 2 if server_round < 2 else 1,
        "batch_size": 2
    }
    return config

def on_evaluate_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return a config for the evaluation process"""
    config: Dict[str, fl.common.Scalar] = {
        "server_round": server_round,
        "batch_size": 4
    }
    return config


# if we return other multiple other metrics which are returned, then define these functions.
fit_metrics_aggregation_fn = None
evaluate_metrics_aggregation_fn = None
