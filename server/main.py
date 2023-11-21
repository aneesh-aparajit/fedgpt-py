from __future__ import annotations
from argparse import ArgumentParser
import flwr as fl
import torch.nn.functional as F

from strategy import FedAvgWithWeightSaving
from utils import (
    on_fit_config_fn, 
    on_evaluate_config_fn, 
    fit_metrics_aggregation_fn, 
    evaluate_metrics_aggregation_fn
)

from models.dataset import load_datasets

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--server_address", type=str, default="[::]:8080", help="gRPC server path for communicating model parameters")
    parser.add_argument("--num_rounds", type=int, default=2, help=f"Number of Federated rounds.")
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fraction of clients to be used for fitting of the model")
    parser.add_argument("--fraction_evaluate", type=float, default=1.0, help="Fraction of clients to be used for the evaluation process.")
    parser.add_argument("--min_fit_clients", type=int, default=2, help=f"The minimum number of clients to call the fit method (defaults: 2)")
    parser.add_argument("--min_available_clients", type=int, default=2, help=f"The minimum number of clients to start sampling of clients (defaults: 2)")
    parser.add_argument("--min_evaluate_clients", type=int, default=2, help=f"The minimum number of clients for the evaluation process. (defaults: 2)")
    parser.add_argument("--do_global_eval", type=bool, default=False, help=f"Do global evaluation? (defaults: False)")
    parser.add_argument("--checkpoint_path", type=str, default="../models/model.pth", help=f"Path to store model ckpt (defaults: '../models/model.pth')")
    parser.add_argument("--accept_failures", type=bool, default=True, help=f"Accept Client Failures? (defaults: False)")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    _, _, testloader = load_datasets(num_clients=args.min_available_clients)

    strategy = FedAvgWithWeightSaving(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        dataloader=testloader,
        do_global_eval=args.do_global_eval,
        loss_fct=F.cross_entropy,
        checkpoint_path=args.checkpoint_path,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        accept_failures=args.accept_failures
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy, 
    )
