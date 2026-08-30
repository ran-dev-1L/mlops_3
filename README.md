# mlops_3
repository for mlops 3rd assignment
# mlops-pytorch-pipeline

An end-to-end MLOps pipeline that trains a PyTorch image classifier on Fashion-MNIST,
containerizes training and serving with Docker, and deploys both to Kubernetes.

## Architecture

```
                       ┌─────────────────────┐
                       │   Fashion-MNIST      │
                       │   (auto-downloaded)  │
                       └──────────┬───────────┘
                                  │
                                  ▼
 ┌───────────────────────────────────────────────────────────┐
 │                     Training (Docker / K8s Job)             │
 │  src/dataset.py → src/model.py (SimpleCNN) → src/train.py   │
 │  reads: configs/training_config.yaml                        │
 │  writes: checkpoints/classifier_v1.pt                       │
 └───────────────────────────┬───────────────────────────────┘
                              │ checkpoint (PVC / volume)
                              ▼
 ┌───────────────────────────────────────────────────────────┐
 │              Serving (Docker / K8s Deployment)               │
 │  src/serve.py — FastAPI                                      │
 │  GET  /health   → 200 if model loaded                        │
 │  POST /predict  → class probabilities for an uploaded image  │
 └───────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                     Client (curl / app)
```

- **Model**: a small CNN (`SimpleCNN`, 2 conv blocks) trained from scratch on
  Fashion-MNIST (28×28 grayscale, 10 classes). Chosen over a fine-tuned ResNet-18
  to keep training fast on CPU-only local hardware while still hitting strong
  accuracy (~89.5% validation accuracy after 10 epochs).
- **Training** reads all hyperparameters from `configs/training_config.yaml`,
  logs structured JSON per epoch, checkpoints the best model by validation loss,
  and supports early stopping.
- **Serving** loads the trained checkpoint once at startup and exposes a
  FastAPI app with `/health` and `/predict`.

## Repository Structure

```
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .github/workflows/ci.yml
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── serve.py
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   ├── configmap.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
└── tests/
    └── test_model.py
```

## Prerequisites

- Python 3.10+
- Docker Desktop (or a Docker-enabled VM)
- kubectl CLI
- A Kubernetes cluster (Minikube, kind, or a cloud-managed cluster)

## Setup — Local (no Docker)

```bash
git clone https://github.com/<your-username>/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline

pip install -r requirements/train.txt
python src/train.py          # trains SimpleCNN on Fashion-MNIST, saves a checkpoint

cd src
pip install -r ../requirements/serve.txt
uvicorn serve:app --host 0.0.0.0 --port 8080
```

Test the serving endpoints (from another terminal):

```bash
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
curl http://localhost:8080/health
```

> On Windows PowerShell, use `curl.exe` (not the `curl` alias) so the
> `-X`/`-F` flags are interpreted correctly.

## Setup — Docker

```bash
# Build training image
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Run training with mounted volumes
docker run --rm \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/checkpoints:/app/checkpoints \
  mlops-train:v1

# Build serving image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Run serving (checkpoint mounted read-write from training output)
docker run --rm -p 8080:8080 \
  -v ${PWD}/checkpoints:/app/checkpoints \
  mlops-serve:v1

# Test prediction endpoint
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

(Replace `${PWD}` with `$(pwd)` on macOS/Linux shells.)

## Setup — Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

# once training completes:
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

kubectl get pods -n ml-training
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Git Workflow

- `main` — stable, always-deployable.
- `develop` — integration branch for completed features.
- `feature/*` — one branch per unit of work, merged into `develop` via PR.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `chore:`, etc.).
