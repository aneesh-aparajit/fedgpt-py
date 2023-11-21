from argparse import ArgumentParser

import torch
import flwr as fl

from client import GptClient
from models.dataset import load_datasets
from models.gpt import NanoGpt, GptConfig

# ------------------------------- Add CLI args ------------------------------- #
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--server_address", type=str, default="[::]:8080", help=f"gRPC server address (defaults: '[::]:8080')")
    parser.add_argument("--cid", type=int, required=True, help="ID of current client (defaults: 0)")
    parser.add_argument("--nb_clients", type=int, default=10, help="Total Number of clients.")
    parser.add_argument("--log_host", type=str, help="Log Server (default: None)")
    parser.add_argument("--batch_size", type=int, help="Batch Size (defualts: 2)", default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    trainloaders, validloaders, _ = load_datasets(num_clients=args.nb_clients, batch_size=args.batch_size)
    model = NanoGpt(
        vocab_size=GptConfig.vocab_size,
        n_embed=GptConfig.n_embed,
        n_heads=GptConfig.n_head,
        buffer_size=GptConfig.buffer_size,
        n_blocks=GptConfig.n_layers,
    )


    client = GptClient(
        cid=str(args.cid),
        trainloader=trainloaders[int(args.cid)], 
        validloader=validloaders[int(args.cid)], 
        optimizer=torch.optim.Adam,
        net=model
    )

    fl.client.start_client(
        server_address=args.server_address, 
        client=client
    )
