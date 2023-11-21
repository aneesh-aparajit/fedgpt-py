from typing import List
import logging
import flwr as fl
import torch
import torch.nn as nn
import numpy as np

from torch.cuda import amp
from torch.utils.data import DataLoader
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Parameters
)

from utils import get_parameters, set_parameters
from models.gpt import NanoGpt
from models.engine import train, test


DEVICE = torch.device("cuda:0" if torch.backends.cuda.is_built() else "cpu")
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flwr {fl.__version__}")


# ---------------------------------- Client ---------------------------------- #
class GptClient(fl.client.Client):
    def __init__(
        self, cid: str, net: nn.Module, 
        trainloader: DataLoader, validloader: DataLoader,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
    ) -> None:
        super().__init__()
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.validloader = validloader
        self.optimizer = optimizer
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        logging.info(f"[Client {self.cid}] get_parameters, config: {ins.config}")
        # get the paramaters of the networl
        ndarrays: List[np.ndarray] = get_parameters(self.net)
        # serialize the ndarrys to parameter object
        parameters: Parameters = ndarrays_to_parameters(ndarrays=ndarrays)
        # set the status
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(status=status, parameters=parameters)
    
    def fit(self, ins: FitIns) -> FitRes:
        logging.info(f"[Client {self.cid}] fit, config: {ins.config}")
        # Deseriualize the parameters
        parameters = ins.parameters
        ndarrays = parameters_to_ndarrays(parameters=parameters)

        # update the local model
        set_parameters(net=self.net, parameters=ndarrays)
        optimizer = self.optimizer(self.net.parameters(), lr=3e-4)
        train(
            model=self.net, 
            optim=optimizer, 
            dataloader=self.trainloader, 
            scalar=amp.grad_scaler.GradScaler(), 
            device=DEVICE, 
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
        )

        # get the updated parameters to send back to the server.
        ndarrays = get_parameters(self.net)
        parameters = ndarrays_to_parameters(ndarrays=ndarrays)
        
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status, 
            parameters=parameters, 
            num_examples=len(self.trainloader), 
            metrics={}
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        logging.info(f"[Client {self.cid}] evaluate, config: {ins.config}")

        # deserialize
        ndarrays = parameters_to_ndarrays(parameters=ins.parameters)

        set_parameters(net=self.net, parameters=ndarrays)
        loss = test(model=self.net, dataloader=self.validloader, device=DEVICE)

        status = Status(code=Code.OK, message="Success")

        return EvaluateRes(
            status=status,
            loss=loss,
            num_examples=len(self.validloader),
            metrics={}
        )
