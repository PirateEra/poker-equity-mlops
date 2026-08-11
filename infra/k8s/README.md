# Kubernetes (API only)

This is a separate way to run the FastAPI app. It is not a replacement for Docker Compose.

- Docker Compose = the full local stack (app, MLflow, Airflow, MySQL, Prometheus, AlertManager, Grafana). Use that for normal development and the full demo.
- Kubernetes here = only the poker API (Deployment + Service). I added it mainly to show I can package the serving layer and deploy it with kubectl on a local cluster (kind / minikube).

Airflow, MySQL, MLflow, Prometheus, etc. are not migrated into k8s in this project. If the API needs a model, MLflow still has to be reachable somehow (usually Compose mlflow on the host).

## What you need

- Docker
- kubectl
- kind or minikube
- A trained model in MLflow (easiest: run Compose, trigger the Airflow training DAG first)

## How to run

### 1. Build the image

From the repo root:

```bash
make build
```

### 2. Load the image into the cluster

`imagePullPolicy: Never`, so the cluster will not pull from a registry. You have to load the local image yourself.

kind:

```bash
kind load docker-image poker:latest
```

minikube:

```bash
minikube image load poker:latest
```

(or `eval $(minikube docker-env)` and then `make build` so the image is built inside minikube’s docker)

### 3. MLflow URI

The Deployment gets `MLFLOW_TRACKING_URI` from `configmap.yaml`. Default is `http://host.docker.internal:8080`, which is meant for “Compose mlflow is running on my machine on port 8080”.

If that does not work on your setup, change the ConfigMap:
- kind / Docker Desktop / WSL2: `host.docker.internal` often works
- plain Linux: you might need your host IP instead, e.g. `http://172.17.0.1:8080`
- minikube: try `http://host.minikube.internal:8080`

Then:

```bash
kubectl apply -f infra/k8s/configmap.yaml
kubectl rollout restart deployment/poker-api
```

### 4. Apply the manifests

```bash
kubectl apply -f infra/k8s/configmap.yaml
kubectl apply -f infra/k8s/deployment.yaml
kubectl apply -f infra/k8s/service.yaml
```

Check:

```bash
kubectl get pods,svc -l app=poker-api
kubectl logs -l app=poker-api --tail=50
```

### 5. Hit the API

```bash
kubectl port-forward svc/poker-api 5001:5001
```

Then open http://localhost:5001/docs (or `/health`, `/metrics`).

Example /predict input:

```json
{
  "S1": 1,
  "C1": 1,
  "S2": 1,
  "C2": 13,
  "S3": 1,
  "C3": 12,
  "S4": 1,
  "C4": 11,
  "S5": 1,
  "C5": 10
}
```