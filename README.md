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
