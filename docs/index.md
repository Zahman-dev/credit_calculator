# Credit Risk Calculator Documentation 📚

Welcome to the official documentation portal for **Credit Risk Calculator**—an end-to-end MLOps project for real-time credit risk assessment.

---

## Key Features

- **FastAPI REST service** for low-latency predictions.
- **Scikit-learn & XGBoost** models with a rich preprocessing pipeline.
- **MLflow** experiment tracking and model registry.
- **Docker Compose** one-command deployment.
- **Prometheus & Grafana** observability stack.

---

## Quickstart (TL;DR)

```bash
git clone https://github.com/your-username/credit_risk_calculator.git
cd credit_risk_calculator
cp env.example .env
# Start API, MLflow, Jupyter
docker-compose up --build
# Or include monitoring
docker-compose --profile monitoring up --build
```

Once the stack is running:

| Service | URL |
|---------|-----|
| API (Swagger) | http://localhost:8001/docs |
| MLflow | http://localhost:5001 |
| JupyterLab | http://localhost:8888 |
| Prometheus* | http://localhost:9090 |
| Grafana* | http://localhost:3000 |

\* Only if the `monitoring` profile is enabled.

---

## Documentation Map

Use the sidebar to navigate between sections:

1. **Architecture** – System overview and diagrams.
2. **API Reference** – Auto-generated Python & endpoint docs.
3. **ADRs** – Architecture Decision Records.

Need more help? Open an issue on [GitHub](https://github.com/your-username/credit_risk_calculator). 