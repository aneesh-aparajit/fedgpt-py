# Federated GPT.
> Implementing a federated learning on [nanoGPT](https://github.com/karpathy/nanoGPT).

- The experiments and other rough implementations of the code can be found here. [fedGPT](https://github.com/aneesh-aparajit/fedGPT).
- This repository consists of the packaged code of the above experiments.


## Run the Server
```zsh
cd server/
docker build -t federated-server .
docker run -it -p 8080:8080 federated-server:latest
```

## Run the Client
```zsh
cd client/
docker build -t federated-client .
docker run -it -p 8080:8080 federated-client:latest
```

## File Structure
```
.
├── README.md
├── client
│   ├── Dockerfile
│   ├── __init__.py
│   ├── client.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── engine.py
│   │   └── gpt.py
│   ├── run-client.sh
│   └── utils.py
├── poetry.lock
├── pyproject.toml
└── server
    ├── Dockerfile
    ├── __init__.py
    ├── main.py
    ├── model.pth
    ├── models
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── engine.py
    │   └── gpt.py
    ├── requirements.txt
    ├── run-server.sh
    ├── strategy.py
    └── utils.py

5 directories, 25 files
```