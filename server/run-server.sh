#!/bin/bash

# Start Flower server
python -m server.main \
    --server_address="[::]:8080" \
    --num_rounds=3 \
    --fraction_fit=1.0 \
    --fraction_evaluate=1.0 \
    --min_fit_clients=2 \
    --min_available_clients=2 \
    --min_evaluate_clients=2 \
    --do_global_eval=True \
    --checkpoint_path="model.pth" \
    --accept_failures=True
