from argparse import ArgumentParser

import torch
import flwr as fl

from client.client import GptClient
from models.dataset import load_datasets

# ------------------------------- Add CLI args ------------------------------- #
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--server_addr", type=str, default="[::]:8080", help=f"gRPC server address (defaults: '[::]:8080')")
    parser.add_argument("--cid", type=int, required=True, help="ID of current client (defaults: 0)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the folder where the data is stored.")
    parser.add_argument("--nb_clients", type=int, default=10, help="Total Number of clients.")
    parser.add_argument("--log_host", type=str, help="Log Server (default: None)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    trainloaders, validloaders, _ = load_datasets(num_clients=args.nb_clients)

    client = GptClient(
        cid=str(args.cid),
        trainloader=trainloaders[int(args.cid)], 
        validloader=validloaders[int(args.cid)], 
        optimizer=torch.optim.Adam
    )

    fl.client.start_client(
        server_address=args.server_address, 
        client=client
    )
