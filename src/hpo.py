"""hpo.py
Hyperparameter optimization for Credit Risk Calculator using Optuna.
Run as a standalone module:

    python -m src.hpo --trials 50 --timeout 600

This will search XGBoost parameters, log each trial to MLflow, and register the best model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import mlflow
import optuna # type: ignore    
from optuna.samplers import TPESampler # type: ignore
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

from .config import (
    GERMAN_CREDIT_DATA_PATH,
    MODEL_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
)
from .pipeline import create_full_pipeline, load_german_credit_data


def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective function that returns cross-validated ROC AUC."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    # Load data once per trial (could be optimized via global cache)
    X, y = load_german_credit_data(GERMAN_CREDIT_DATA_PATH)

    pipeline = create_full_pipeline(XGBClassifier(**params))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    mean_auc = auc_scores.mean()

    # Log params and metric to MLflow inside Optuna hook
    mlflow.log_params(params)
    mlflow.log_metric("cv_auc", mean_auc)

    return mean_auc  # Optuna will maximize by default if study.direction="maximize"


def optimize(trials: int, timeout: int | None = None) -> None:
    """Run optimization and log best model to MLflow."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    with mlflow.start_run(run_name="hpo-optuna"):
        study.optimize(objective, n_trials=trials, timeout=timeout)

        best_params = study.best_trial.params
        best_score = study.best_value

        # Save best params to JSON artifact
        artifact_path = Path("best_params.json")
        artifact_path.write_text(json.dumps(best_params, indent=2))
        mlflow.log_artifact(str(artifact_path))

        # Train final model with best params
        X, y = load_german_credit_data(GERMAN_CREDIT_DATA_PATH)
        final_pipeline = create_full_pipeline(XGBClassifier(**best_params))
        final_pipeline.fit(X, y)

        # Register model
        mlflow.sklearn.log_model(  # type: ignore
            sk_model=final_pipeline,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
            input_example=X.head(),  # type: ignore
        )

        print("Best ROC AUC:", best_score)
        print("Best params:", best_params)

        # Optionally save locally
        save_local_env = os.getenv("SAVE_LOCAL_MODEL", "false").lower()
        if save_local_env in {"1", "true", "yes"}:
            MODEL_DIR.mkdir(exist_ok=True)
            import joblib

            joblib.dump(final_pipeline, MODEL_DIR / "xgboost_optuna_best.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Time limit in seconds")

    args = parser.parse_args()
    optimize(trials=args.trials, timeout=args.timeout) 