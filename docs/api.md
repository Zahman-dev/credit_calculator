# API Reference 🌐

The REST API is powered by **FastAPI** and automatically documents itself via Swagger/OpenAPI. When the server is running you can explore and test endpoints at **`/docs`**.

---

## Base URL

Local Docker deployment:

```txt
http://localhost:8001
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Root endpoint with API status & metadata |
| `GET`  | `/health` | Liveness probe—returns `healthy` if the model is loaded |
| `POST` | `/predict` | Credit risk prediction for a single applicant (optionally secured by API key) |
| `GET`  | `/metrics` | Prometheus metrics endpoint (not visible in Swagger) |

---

## Schema – `POST /predict`

The request body must match the Pydantic model `CreditDataInput`. Below is a minimal example including **all** mandatory fields.

```json
{
  "Duration": 12,
  "Credit_amount": 5000,
  "Age": 35,
  "Installment_rate": 3,
  "Present_residence": 2,
  "Existing_credits": 1,
  "Dependents": 1,
  "Checking_account": "A11",
  "Credit_history": "A34",
  "Purpose": "A43",
  "Savings_account": "A61",
  "Employment": "A73",
  "Personal_status_sex": "A93",
  "Other_debtors": "A101",
  "Property": "A121",
  "Other_installment_plans": "A143",
  "Housing": "A152",
  "Job": "A173",
  "Telephone": "A192",
  "Foreign_worker": "A201"
}
```

### Response

```json
{
  "risk_prediction": "Good",
  "risk_probability": 0.14,
  "confidence": "High"
}
```

---

## Authentication (Optional)

If the environment variable `API_KEYS` is set (comma-separated list) the `/predict` endpoint requires an API key supplied via the `X-API-Key` header:

```bash
curl -H "X-API-Key: my-secret-key" \   \
     -H "Content-Type: application/json" \   \
     -d '{ /* payload */ }' \   \
     http://localhost:8001/predict
```

Without `API_KEYS`, the endpoint is open—ideal for local experiments.

---

## Python Client Example

```python
import requests

payload = {  # Same as JSON above }
resp = requests.post("http://localhost:8001/predict", json=payload)
print(resp.json())
```

---

For the complete Python API surface—configuration classes, training utilities, etc.—see the auto-generated module reference below.

::: src 