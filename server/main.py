from argparse import ArgumentParser
import flwr as fl
import torch.nn.functional as F

from server.strategy import FedAvgWithWeightSaving
from server.utils import (
    on_fit_config_fn, 
    on_evaluate_config_fn, 
    fit_metrics_aggregation_fn, 
    evaluate_metrics_aggregation_fn
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--fraction_fit", type=float, default=1.0,
        help="Fraction of clients to be used for fitting of the model"
    )

    parser.add_argument(
        "--fraction_evaluate", type=float, default=1.0,
        help="Fraction of clients to be used for the evaluation process."
    )

    return parser.parse_args()

if __name__ == '__main__':
    testloader = None

    args = parse_args()

    strategy = FedAvgWithWeightSaving(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evalute,
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
        server_address=None,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy, 
    )
