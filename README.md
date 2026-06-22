# Adversarial-Robust ML Pipeline

An end-to-end MLOps pipeline that trains an image classifier, automatically attacks it with adversarial methods (FGSM, PGD, data poisoning), measures robustness, applies defenses, and **gates deployment in CI/CD** if the model falls below a robustness threshold.

This project applies DevSecOps principles -- automated security testing as a release gate -- to machine learning models.

## Status
Work in progress. See docs/architecture.md for the planned design.

## Why this exists
Most ML portfolio projects stop at "trained a model, got X% accuracy." This one asks: what happens when someone tries to break it, and does the pipeline catch that automatically?

## Architecture (high level)
Data -> Training (PyTorch) -> Adversarial Attack Suite -> Robustness Gate (GitHub Actions) -> FastAPI serving

## Tech stack
- PyTorch (model + attacks)
- torchattacks / Adversarial Robustness Toolbox (attack implementations)
- MLflow (experiment tracking & model versioning)
- GitHub Actions (CI/CD robustness gate)
- FastAPI (serving + robustness report endpoint)
- Docker (containerized deployment)

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Project structure
## Roadmap
- [ ] Baseline CNN classifier + MLflow tracking
- [ ] FGSM attack implementation + evaluation
- [ ] PGD attack implementation + evaluation
- [ ] Data poisoning experiment
- [ ] Adversarial training defense
- [ ] Input sanitization defense
- [ ] GitHub Actions robustness gate
- [ ] FastAPI serving + Docker
