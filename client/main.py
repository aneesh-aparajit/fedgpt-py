from argparse import ArgumentParser

import numpy as np
import torch
import flwr as fl


# ------------------------------- Add CLI args ------------------------------- #
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--server_addr",
        type=str,
        default="[::]:8080",
        help=f"gRPC server address (defaults: '[::]:8080')"
    )
    parser.add_argument(
        "--cid",
        type=str,
        metavar="N",
        help="ID of current client (defaults: 0)"
    )

    return parser.parse_args()

