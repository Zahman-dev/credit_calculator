# System Architecture

```mermaid
flowchart TD
    subgraph API Layer
        A[FastAPI Server]
    end

    subgraph Model Registry
        C[MLflow Registry]
    end

    subgraph Monitoring
        D[Prometheus]
        E[Grafana]
    end

    subgraph Data & Storage
        F[(Models Dir)]
        G[(Dataset CSV)]
    end

    A -- requests --> C
    A -- metrics --> D
    D -- dashboard --> E
    C -- artifacts --> F
    A -- uses --> G
```

The platform adheres to a micro-ML-service style. The API layer serves predictions and orchestrates data flows; MLflow manages experiment tracking and model versioning; Prometheus & Grafana provide observability. 