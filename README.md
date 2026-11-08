# Industrial Machine Failure Prediction System

## About

This project predicts whether an industrial machine is likely to fail based on
its operating conditions. It uses the AI4I 2020 predictive-maintenance dataset
and a trained classification model. The deployed model takes six inputs:

- Product quality variant (`L`, `M`, or `H`)
- Air temperature and process temperature (K)
- Rotational speed (rpm)
- Torque (Nm)
- Tool wear (min)

The app validates these values and returns a failure prediction and probability.
You can use the web form or the JSON API. This is a demonstration and decision
support tool, not a replacement for machine inspections or safety procedures.

The repository separates model training assets from model serving: the dataset,
notebook, and training workflow stay outside the production container. The
container includes only the web app, inference dependencies, and the trained
model. Docker and Kubernetes deployment instructions follow.

## Run with Docker

Build the production image from the repository root:

```powershell
docker build -t machinefailure_ml .
```

Run the app on port 8080. The container runs as a non-root user; the read-only
filesystem and temporary filesystem further restrict runtime writes:

```powershell
docker run --rm -p 8080:8080 --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m machinefailure_ml
```

Open `http://localhost:8080/` for the form, `http://localhost:8080/health` for
the container health check, or send JSON predictions to `POST /api/predict`.
The image installs only the serving dependencies in `requirements.txt` and
copies the application plus compressed inference model. Training data,
notebooks, and training-only libraries such as imbalanced-learn and pandas are
not included. The packaged model keeps only its scaler and classifier; the
training sampler is not needed to make predictions.

## Run on Kubernetes

The manifest creates a two-replica Deployment and an internal ClusterIP
Service:

```powershell
kubectl apply -f deployment.yaml
kubectl rollout status deployment/machinefailure
kubectl port-forward service/machinefailure 8080:8080
```

Then open `http://localhost:8080/`. The manifest uses the local image name
`machinefailure_ml:latest`. Make that image available to your cluster before
applying the manifest (for Minikube, run
`minikube image load machinefailure_ml:latest`), or change the image to a
registry URL accessible by your cluster. The container runs non-root with a
read-only root filesystem; Kubernetes probes `/health` and mounts a temporary
writable `/tmp`.
