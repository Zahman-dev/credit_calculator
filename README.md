# Credit Risk Scoring API 🏦

A comprehensive machine learning project that predicts credit risk using the German Credit Dataset. This project demonstrates the complete MLOps lifecycle from data analysis to production deployment.

## 🚀 Features

- **Machine Learning Pipeline**: Automated data preprocessing and model training
- **REST API**: FastAPI-based API for real-time credit risk predictions
- **MLflow Integration**: Experiment tracking and model versioning
- **Docker Support**: Containerized deployment ready for production
- **Monitoring**: Health checks and performance metrics
- **Production Ready**: Scalable architecture with proper error handling

## 📊 Dataset

This project uses the [UCI German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), which contains information about credit applicants and their risk classification.

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd credit_risk_calculator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   - Recommended (PEP-517):
     ```bash
     # Requires pip>=23.1
     pip install -e .[dev]
     ```

4. **Configure environment variables**
   ```bash
   cp env.example .env  # edit values as needed
   ```

4. **Dataset**
   Hiçbir manuel indirme gerekmez. `src.data_ingestion` modülü çalışırken veriyi otomatik olarak
   indirir ve `data/german_credit_data.csv` olarak ön-işler. Dolayısıyla ilk eğitim ya da test
   sırasında veri kümesi kendiliğinden hazır hâle gelir.

## 🔧 Usage

### 1. Data Analysis & Model Training

Start MLflow tracking server:
```bash
mlflow ui
```

Train the model:
```bash
python -m src.train --model xgboost
```

Compare models:
```bash
python -m src.train --model compare
```

### 2. Run the API

Start the FastAPI server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 3. Make Predictions

Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Duration": 12,
       "Credit_amount": 5000.0,
       "Age": 35,
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
     }'
```

## 🐳 Docker Deployment

### Build and run locally:

```bash
# Build the image
docker build -t credit-risk-api .

# Run the container
docker run -p 8080:80 credit-risk-api
```

### Deploy to cloud platforms:

The Docker image is ready for deployment on:
- **Render**: Connect your GitHub repo and deploy automatically
- **Heroku**: Use container registry
- **AWS ECS/EKS**: Deploy with container orchestration
- **Google Cloud Run**: Serverless container deployment

## 🧪 Testing

This project uses `pytest` for unit and integration testing.

1.  **Install development dependencies:**
    Proje kök dizininde aşağıdaki komutla hem çalışma zamandaki hem de geliştirme
    bağımlılıklarını kurun (PEP-517 / `pyproject.toml` tabanlı).
    ```bash
    pip install -e .[dev]
    ```

2.  **Run tests:**
    From the root directory, run `pytest`. It's recommended to run it via `python -m pytest` to ensure it uses the correct virtual environment.
    ```bash
    python -m pytest
    ```

## 📁 Project Structure

```
credit_risk_calculator/
├── .github/workflows/         # CI/CD pipelines
├── data/                      # Dataset files
├── notebooks/                 # Jupyter notebooks for EDA
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── pipeline.py           # Data preprocessing pipeline
│   └── train.py              # Model training script
├── tests/                    # Unit tests
├── models/                   # Saved model artifacts
├── .gitignore
├── Dockerfile
├── main.py                   # FastAPI application
└── README.md
```

## 🧪 API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `POST /predict` - Credit risk prediction endpoint
- `GET /docs` - Interactive API documentation
- `GET /metrics` - Prometheus metrics endpoint (scrape target)

## 📈 Model Performance

The project includes multiple models:
- **Logistic Regression**: Baseline interpretable model
- **XGBoost**: High-performance gradient boosting model

Performance metrics tracked:
- Accuracy
- ROC AUC Score
- Precision, Recall, F1-Score
- Cross-validation scores

## 🔄 MLOps Pipeline

1. **Data Ingestion**: Load and validate German Credit Dataset
2. **Data Preprocessing**: Handle missing values, encode categories, scale features
3. **Model Training**: Train multiple models with hyperparameter tuning
4. **Model Evaluation**: Comprehensive metrics and cross-validation
5. **Model Registry**: Version control with MLflow
6. **API Development**: FastAPI with automatic documentation
7. **Containerization**: Docker for consistent deployment
8. **Monitoring**: Health checks and performance tracking

## 🚀 Production Deployment

### Environment Variables

Set these environment variables for production:

```bash
MLFLOW_TRACKING_URI=<your-mlflow-server>
MLFLOW_MODEL_NAME=credit_risk_model
# veya `.env` dosyası oluşturup aşağıdaki gibi doldurabilirsiniz
# (bkz. pydantic-settings kullanımı):
#
# MLFLOW_TRACKING_URI=http://localhost:5000
# MLFLOW_MODEL_NAME=credit_risk_model
```

### Scaling Considerations

- Use multiple worker processes in production
- Implement caching for frequently accessed models
- Set up proper logging and monitoring
- Use a reverse proxy (nginx) for load balancing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the German Credit Dataset
- FastAPI team for the excellent web framework
- MLflow team for experiment tracking capabilities
- Scikit-learn and XGBoost communities

## 📧 Contact

- GitHub: [Your GitHub Profile]
- LinkedIn: [Your LinkedIn Profile]
- Email: [Your Email]

## 📚 Documentation

Full documentation (architecture, Python reference, ADRs) is available via MkDocs:

```bash
# Serve locally
pip install mkdocs-material mkdocstrings[python]
mkdocs serve
```

Deployed site: https://example.com/docs

---

**Made with ❤️ for the ML community** 