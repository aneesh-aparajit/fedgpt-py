from typing import List
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """Function to convert model to list of numpy arrays of the weights."""
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    """Function to load parameter numpy list to model."""
    params = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params})
    net.load_state_dict(state_dict=state_dict, strict=True)
