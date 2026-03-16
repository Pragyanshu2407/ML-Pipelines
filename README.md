# ML Regression Testing Pipeline

An automated CI/CD pipeline that re-trains an ML model on every
git push, compares it against the current champion model using
statistical tests, and only deploys if the new model is better.

## Stack
- DVC — data and pipeline versioning
- MLflow + DagsHub — experiment tracking and model registry
- GitHub Actions — CI/CD automation
- CML — metric reports in pull requests
- Great Expectations — data validation
- Evidently AI — drift detection
- Locust + Prometheus + Grafana — load testing and monitoring

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run pipeline locally
```bash
dvc repro
```