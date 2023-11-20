import os
from typing import Callable, Optional, List, Tuple, Dict, Union

import logging
import torch
import torch.nn.functional as F
import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from models.gpt import NanoGpt
from utils import get_parameters, set_parameters

DEVICE = torch.device("cuda:0" if torch.backends.cuda.is_built() else "cpu")


class FedAvgWithWeightSaving(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        dataloader: torch.utils.data.DataLoader = None,
        do_global_eval: bool = False,
        loss_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy,
        checkpoint_path: str = "./models.pth",
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        accept_failures: bool = False,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.dataloader = dataloader
        self.do_global_eval = do_global_eval
        self.loss_fct = loss_fct
        self.checkpoint_path = checkpoint_path
        self.global_loss = np.infty
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.accept_failures = accept_failures

    def __repr__(self) -> str:
        return "FedAvgWithWeightSaving"
    
    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Initialize the global model parameters"""
        net = NanoGpt()

        if not os.path.exists(self.checkpoint_path):
            # check if checkpoints are found, else randomly initialize the model and populate global checkpoint.
            torch.save(net.state_dict(), f=self.checkpoint_path)
            logging.warn(msg="Existing best checkpoints not found. Randomly initializing global model paramaters")
        else:
            state_dict = torch.load(f=self.checkpoint_path)
            logging.info(net.load_state_dict(state_dict=state_dict, strict=True))
            logging.info(msg="Existing best checkpoints found. Loading those parameters automatically.")
        
        ndarrays = get_parameters(net=net)
        return ndarrays_to_parameters(ndarrays=ndarrays)
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Returns the sample size and the required number of available clients for fit."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients
    
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Returns the sample size and the required number of available clients for eval"""
        num_clients = int(num_available_clients*self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model paramaters using the evaluation function"""
        if not self.do_global_eval:
            return None
        
        # load the model
        parameters_ndarrays = parameters_to_ndarrays(parameters=parameters)
        net = NanoGpt()
        set_parameters(net=net, parameters=parameters_ndarrays)
        net = net.to(DEVICE)
        logging.info(msg=f"[Server Round: {server_round}], Loading global model parameters on the model.")

        if self.dataloader is None:
            logging.warn(msg=f"Centralized DataLoader not found. Restart server with the dataset, if centralized validation is to be performed.")
            return None

        # run the evaluation process on the centralized server.
        running_loss, dataset_size = 0, 0
        with torch.no_grad():
            for batch in self.dataloader:
                batch = {k:v.to(DEVICE) for k,v in batch.items()}
                _, loss = net.forward(**batch)
                bs = batch['labels'].size(0)
                running_loss += loss.item()*bs
                dataset_size += bs
        
        loss = running_loss / dataset_size

        if loss < self.global_loss:
            logging.info(msg=f"Best Global Loss achieved.")
            logging.info(msg=f"Global loss reduced from: {self.global_loss:5f} to {loss:.5f}")
            self.global_loss = loss
            logging.info(msg=f"Overwriting existing checkpoint at {self.checkpoint_path} with current checkpoint")
            logging.info(torch.save(net.state_dict(), f=self.checkpoint_path))

        return loss, {}
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        logging.info(msg=f"[Server Round #{server_round}], Sampling Clients For Fit.")
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom Fit config fn called
            config = self.on_fit_config_fn(server_round)
        fitins = FitIns(parameters=parameters, config=config)

        # Sample Clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client, fitins) for client in clients]
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        logging.info(msg=f"[Server Round #{server_round}], Sampling Clients For Evaluation.")

        # we don't need to configure clients if the fraction_eval is 0.0
        if self.fraction_evaluate == 0.0:
            return []
        
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        eval_ins = EvaluateIns(parameters=parameters, config=config)

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client, eval_ins) for client in clients]
    
    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) ->  Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted avg"""
        logging.info(msg=f"[Server Round {server_round}], Fit Aggregation")
        if not results:
            return None, {}
        
        # Do not aggregate if there are any failures.
        if failures and not self.accept_failures:
            return None, {}
        
        weights_results = [
            (parameters_to_ndarrays(parameters=res.parameters), res.num_examples)
            for _, res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided.
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            logging.warn(msg="No fit_metrics_aggregation_fn provided")
        
        return parameters_aggregated, metrics_aggregated
    
    def aggregate_evalute(
        self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate Evaluation losses"""
        logging.info(msg=f"[Server Round {server_round}], Evaluate Aggregation")
        
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        
        # Aggregate Loss
        loss_aggregated = weighted_loss_avg(results=[
            (res.num_examples, res.loss) for _, res in results
        ])

        # Aggregate custom metrics
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            logging.warn("No evaluate_metrics_aggregation_fn provided")
        
        return loss_aggregated, metrics_aggregated
