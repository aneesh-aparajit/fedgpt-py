#!/bin/bash

# Extracted from https://github.com/adap/flower/blob/main/src/py/flwr_example/quickstart_pytorch/run-clients.sh
set -e

SERVER_ADDR="[::]:8080"
NUM_CLIENTS=2

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTSl; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python -m client.main \
        --cid=$i \
        --server_addr=$SERVER_ADDR \
        --nb_clients=$NUM_CLIENTS &
done

echo "Started $NUM_CLIENTS clients."
