from typing import List
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    return [val.cpu().detach().numpy() for _, val in net.state_dict().items()]

def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    params = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k:torch.tensor(v) for k,v in params})
    print(net.load_state_dict(state_dict=state_dict, strict=True))
