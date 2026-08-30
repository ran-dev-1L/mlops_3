# mlops-pytorch-pipeline

An end-to-end MLOps pipeline that trains a PyTorch image classifier on Fashion-MNIST,
containerizes training and serving with Docker, and deploys both to Kubernetes
(tested locally with Minikube), with autoscaling and CI in place.

## Architecture

```
                       ┌─────────────────────┐
                       │   Fashion-MNIST      │
                       │   (auto-downloaded)  │
                       └──────────┬───────────┘
                                  │
                                  ▼
 ┌───────────────────────────────────────────────────────────┐
 │                Training (Docker / K8s Job)                  │
 │  src/dataset.py → src/model.py (SimpleCNN) → src/train.py   │
 │  reads:  configs/training_config.yaml (via ConfigMap)       │
 │  writes: checkpoints/classifier_v1.pt (via PVC)             │
 └───────────────────────────┬───────────────────────────────┘
                              │ checkpoint (PVC / volume mount)
                              ▼
 ┌───────────────────────────────────────────────────────────┐
 │           Serving (Docker / K8s Deployment + HPA)            │
 │  src/serve.py — FastAPI, 2-5 replicas, autoscaled on CPU      │
 │  GET  /health   → 200 if model loaded                        │
 │  POST /predict  → class probabilities for an uploaded image  │
 │  exposed via a ClusterIP Service                             │
 └───────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                     Client (curl / app)
```

- **Model**: `SimpleCNN`, a small 2-conv-block CNN trained from scratch on
  Fashion-MNIST (28x28 grayscale, 10 classes). Chosen over a fine-tuned
  ResNet-18 to keep training fast on CPU-only hardware while still reaching
  strong accuracy (about 89-90% validation accuracy after 10 epochs, verified
  locally, in Docker, and inside a Kubernetes Job).
- **Training** reads all hyperparameters from `configs/training_config.yaml`
  (mounted from a ConfigMap in Kubernetes), logs structured JSON per epoch,
  checkpoints the best model by validation loss, and supports early stopping.
- **Serving** loads the trained checkpoint once at startup and exposes a
  FastAPI app with `/health` and `/predict`, running as a non-root user in
  its container.
- **Autoscaling**: a HorizontalPodAutoscaler keeps the serving Deployment
  between 2 and 5 replicas based on CPU utilization (target 70%).
- **CI**: a GitHub Actions workflow runs the unit tests on every push and
  pull request to `main` and `develop`.

## Repository Structure

```
mlops-pytorch-pipeline/
├── README.md
├── pytest.ini
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
- Minikube (used for the local Kubernetes cluster in this project)

## Setup — Local (no Docker)

```bash
git clone https://github.com/ran-dev-1L/mlops_3.git
cd mlops_3

pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/train.txt
python src/train.py          # trains SimpleCNN on Fashion-MNIST, saves a checkpoint

cd src
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r ../requirements/serve.txt
uvicorn serve:app --host 0.0.0.0 --port 8080
```

Test the serving endpoints (from another terminal, inside `src/` so
`test_image.png` resolves):

```bash
curl.exe -X POST http://localhost:8080/predict -F "image=@test_image.png"
curl.exe http://localhost:8080/health
```

> On Windows PowerShell, use `curl.exe` (not the `curl` alias) so the
> `-X`/`-F` flags are interpreted correctly.

## Running Tests

```bash
pip install pytest
pytest tests/test_model.py -v
```

`pytest.ini` at the repo root adds `src/` to the Python path so the tests can
import `model.py` regardless of which directory pytest is run from.

## Setup — Docker

```bash
# Build training image (CPU-only PyTorch wheels keep the image small and the build fast)
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Run training with mounted volumes
docker run --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/checkpoints:/app/checkpoints `
  mlops-train:v1

# Build serving image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Run serving (checkpoint mounted read-write from training output)
docker run --rm -p 8080:8080 `
  -v ${PWD}/checkpoints:/app/checkpoints `
  mlops-serve:v1

# Test prediction endpoint
curl.exe -X POST http://localhost:8080/predict -F "image=@src/test_image.png"
```

(On macOS/Linux shells, replace the backtick line continuations with `\` and
`${PWD}` with `$(pwd)`.)

The serving image runs as a non-root user and includes a `HEALTHCHECK`
instruction that polls `GET /health`.

## Setup — Kubernetes (Minikube)

```bash
# Start a local cluster
minikube start --driver=docker

# Make the images built above available to the cluster
minikube image load mlops-train:v1
minikube image load mlops-serve:v1

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

# Watch training run to completion
kubectl get pods -n ml-training
kubectl logs -f job/model-training -n ml-training

# Once training completes, deploy serving + autoscaling
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml

# Optional: enable metrics-server so the HPA has real CPU data
minikube addons enable metrics-server
kubectl apply -f k8s/hpa.yaml
kubectl get hpa -n ml-training

# Verify pods are healthy
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# Test the prediction endpoint
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl.exe http://localhost:8080/health
curl.exe -X POST http://localhost:8080/predict -F "image=@src/test_image.png"
```

### Verified results

- Training Job completed 10 epochs inside the cluster, best validation
  accuracy about 89.97%, with the checkpoint persisted to a PersistentVolumeClaim.
- Both serving replicas reached `1/1 Running` and passed their readiness
  probes.
- `GET /health` returned `{"status":"ok"}` and `POST /predict` returned a
  correct, high-confidence prediction through the Service, via
  `kubectl port-forward`.
- With `metrics-server` enabled, `kubectl get hpa` showed live CPU targets
  (for example `0%/70%`) instead of `<unknown>`, confirming the
  HorizontalPodAutoscaler is correctly wired to the Deployment.

## CI

`.github/workflows/ci.yml` runs on every push and pull request to `main` and
`develop`. It installs CPU-only training dependencies and runs
`pytest tests/test_model.py -v`.

## Git Workflow

- `main` — stable, always-deployable.
- `develop` — integration branch for completed features.
- `feature/*` — one branch per unit of work, merged into `develop` via PR.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `chore:`, etc.).