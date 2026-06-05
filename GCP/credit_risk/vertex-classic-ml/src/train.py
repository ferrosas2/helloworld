"""
train.py — Vertex AI Training Container Script
------------------------------------------------
Runs INSIDE a Vertex AI Custom Training Job container.

Responsibilities:
  - Parse hyperparameters via argparse
  - Generate a synthetic Credit Risk dataset
  - Train an XGBoost Classifier
  - Log params & metrics to Vertex AI Experiments
  - Report the primary metric to Vertex Vizier via cloudml-hypertune
  - Save the serialised model to Cloud Storage (GCS)

Usage (local smoke-test):
    python src/train.py \
        --max_depth 4 \
        --learning_rate 0.1 \
        --n_estimators 100 \
        --subsample 0.8 \
        --model_dir gs://YOUR_BUCKET/models/credit-risk
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Tuple

import hypertune
import joblib
import pandas as pd
from google.cloud import aiplatform
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_NAME: str = "credit-risk-experiment"
N_SAMPLES: int = 10_000
N_FEATURES: int = 20
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
MODEL_FILENAME: str = "model.joblib"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse training hyperparameters and runtime configuration."""
    parser = argparse.ArgumentParser(description="XGBoost Credit-Risk Trainer")

    # --- Hyperparameters (tunable via Vizier) ---
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="Maximum tree depth for XGBoost.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Boosting learning rate (eta).",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of boosting rounds.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Subsample ratio of the training instances.",
    )
    parser.add_argument(
        "--colsample_bytree",
        type=float,
        default=0.8,
        help="Subsample ratio of columns per tree.",
    )

    # --- Runtime configuration ---
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("AIP_MODEL_DIR", "/tmp/model"),
        help=(
            "GCS URI (gs://bucket/path) or local path where model.joblib "
            "will be saved. Falls back to the AIP_MODEL_DIR env var."
        ),
    )
    parser.add_argument(
        "--project_id",
        type=str,
        default=os.environ.get("CLOUD_ML_PROJECT_ID", ""),
        help="GCP Project ID used to initialise Vertex AI SDK.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.environ.get("CLOUD_ML_REGION", "us-central1"),
        help="GCP region used to initialise Vertex AI SDK.",
    )
    parser.add_argument(
        "--experiment_run",
        type=str,
        default="",
        help="Optional: explicit experiment run name (auto-generated if blank).",
    )

    return parser.parse_args()


def build_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic Credit Risk tabular dataset."""
    logger.info(
        "Generating synthetic dataset: %d samples × %d features",
        N_SAMPLES,
        N_FEATURES,
    )
    X_raw, y_raw = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],   # realistic class imbalance for credit risk
        random_state=RANDOM_STATE,
    )
    feature_cols = [f"feature_{i:02d}" for i in range(N_FEATURES)]
    X = pd.DataFrame(X_raw, columns=feature_cols)
    y = pd.Series(y_raw, name="default")
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    args: argparse.Namespace,
) -> XGBClassifier:
    """Instantiate and fit an XGBoost classifier."""
    model = XGBClassifier(
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    logger.info("Training XGBClassifier with params: %s", model.get_params())
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, float]:
    """Return (accuracy, f1_score) on the test split."""
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="binary"))
    logger.info("Evaluation — accuracy: %.4f | f1_score: %.4f", acc, f1)
    return acc, f1


def save_model(model: XGBClassifier, model_dir: str) -> str:
    """
    Serialise the model to model_dir.

    Supports both local paths and GCS URIs (gs://...).
    For GCS, the file is written locally first then uploaded via
    google-cloud-storage (bundled with google-cloud-aiplatform).
    """
    local_path = f"/tmp/{MODEL_FILENAME}"
    joblib.dump(model, local_path)
    logger.info("Model serialised locally to %s", local_path)

    if model_dir.startswith("gs://"):
        # Strip the leading "gs://" and split into bucket / blob path
        from google.cloud import storage  # type: ignore[import-untyped]

        gcs_path = model_dir[5:]  # remove "gs://"
        bucket_name, _, prefix = gcs_path.partition("/")
        blob_name = f"{prefix}/{MODEL_FILENAME}".lstrip("/")

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        destination = f"gs://{bucket_name}/{blob_name}"
        logger.info("Model uploaded to GCS: %s", destination)
        return destination

    # Local / mounted volume path
    import shutil

    os.makedirs(model_dir, exist_ok=True)
    dest = os.path.join(model_dir, MODEL_FILENAME)
    shutil.copy(local_path, dest)
    logger.info("Model copied to %s", dest)
    return dest


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Initialise Vertex AI SDK & start an experiment run
    # ------------------------------------------------------------------
    init_kwargs: dict[str, str] = {"experiment": EXPERIMENT_NAME}
    if args.project_id:
        init_kwargs["project"] = args.project_id
    if args.region:
        init_kwargs["location"] = args.region

    aiplatform.init(**init_kwargs)  # type: ignore[arg-type]

    import uuid

    run_name: str = args.experiment_run or f"run-{uuid.uuid4().hex[:8]}"
    run_ctx = aiplatform.start_run(run=run_name)
    logger.info("Vertex AI Experiment run started: %s", run_ctx.name)

    # ------------------------------------------------------------------
    # 2. Build dataset & split
    # ------------------------------------------------------------------
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ------------------------------------------------------------------
    # 3. Log hyperparameters to Vertex AI Experiments
    # ------------------------------------------------------------------
    params: dict[str, float | int] = {
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }
    aiplatform.log_params(params)
    logger.info("Logged params to Vertex Experiments: %s", params)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    model = train_model(X_train, y_train, args)

    # ------------------------------------------------------------------
    # 5. Evaluate & log metrics
    # ------------------------------------------------------------------
    accuracy, f1 = evaluate_model(model, X_test, y_test)
    metrics: dict[str, float] = {"accuracy": accuracy, "f1_score": f1}
    aiplatform.log_metrics(metrics)
    logger.info("Logged metrics to Vertex Experiments: %s", metrics)

    # ------------------------------------------------------------------
    # 6. Report primary metric to Vertex Vizier (HPT)
    #    cloudml-hypertune reads CLOUD_ML_HP_METRIC_* env vars injected
    #    by the Vizier service when running inside an HPT job.
    # ------------------------------------------------------------------
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="f1_score",
        metric_value=f1,
        global_step=args.n_estimators,
    )
    logger.info("Reported f1_score=%.4f to Vertex Vizier via hypertune.", f1)

    # ------------------------------------------------------------------
    # 7. Persist model artifact
    # ------------------------------------------------------------------
    destination = save_model(model, args.model_dir)
    logger.info("Training complete. Artifact at: %s", destination)

    # End the experiment run cleanly
    aiplatform.end_run()


if __name__ == "__main__":
    main()
