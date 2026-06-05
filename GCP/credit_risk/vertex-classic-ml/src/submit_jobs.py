"""
submit_jobs.py — Local Orchestrator
-------------------------------------
Run this script from your laptop, a CI/CD pipeline, or an Airflow DAG to
trigger Vertex AI workloads.  It never executes training itself — compute
lives entirely on Google Cloud.

Required environment variables:
    PROJECT_ID   — GCP project identifier
    REGION       — GCP region, e.g. us-central1
    BUCKET_NAME  — GCS bucket name (without gs://) for model artefacts

Optional environment variables:
    TRAIN_IMAGE   — Override the pre-built training container image URI
    SERVICE_ACCOUNT — Service account e-mail for the training job
    NETWORK       — VPC network for private IP jobs

Usage:
    # Submit a one-off custom training job
    python src/submit_jobs.py --job-type training

    # Submit a hyperparameter tuning job via Vertex Vizier
    python src/submit_jobs.py --job-type hpt

    # Submit both (sequential)
    python src/submit_jobs.py --job-type all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — override via environment variables where appropriate
# ---------------------------------------------------------------------------

# Pre-built Vertex AI training container for Python 3.10 / CPU workloads.
# See full list: https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
DEFAULT_TRAIN_IMAGE: str = (
    "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest"
)

EXPERIMENT_NAME: str = "credit-risk-experiment"
DISPLAY_NAME_TRAINING: str = "credit-risk-xgboost-training"
DISPLAY_NAME_HPT: str = "credit-risk-xgboost-hpt"

# Machine type used for each trial worker
MACHINE_TYPE: str = "n1-standard-4"

# HPT configuration
MAX_TRIAL_COUNT: int = 6
PARALLEL_TRIAL_COUNT: int = 3


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    """Return the value of an environment variable or raise RuntimeError."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set or is empty."
        )
    return value


def _load_config() -> dict[str, str]:
    """Load and validate all required runtime configuration from env vars."""
    return {
        "project_id": _require_env("PROJECT_ID"),
        "region": _require_env("REGION"),
        "bucket_name": _require_env("BUCKET_NAME"),
        "train_image": os.environ.get("TRAIN_IMAGE", DEFAULT_TRAIN_IMAGE),
        "service_account": os.environ.get("SERVICE_ACCOUNT", ""),
        "network": os.environ.get("NETWORK", ""),
    }


# ---------------------------------------------------------------------------
# Job 1 — Custom Training Job
# ---------------------------------------------------------------------------

def submit_custom_training_job(
    config: dict[str, str],
    max_depth: int = 4,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    sync: bool = True,
) -> aiplatform.CustomTrainingJob:
    """
    Configure and submit a Vertex AI Custom Training Job.

    The job uses a pre-built Google-managed container and points to
    `src/train.py` (uploaded to GCS alongside the source distribution,
    or baked into a custom image in production).

    Parameters
    ----------
    config:
        Runtime config dict produced by _load_config().
    max_depth, learning_rate, n_estimators, subsample, colsample_bytree:
        Hyperparameters forwarded to the training container via args.
    sync:
        If True (default) block until the job completes.

    Returns
    -------
    The submitted CustomTrainingJob instance.
    """
    logger.info("Initialising Vertex AI SDK — project=%s region=%s",
                config["project_id"], config["region"])

    aiplatform.init(
        project=config["project_id"],
        location=config["region"],
        experiment=EXPERIMENT_NAME,
        staging_bucket=f"gs://{config['bucket_name']}",
    )

    model_dir: str = (
        f"gs://{config['bucket_name']}/models/credit-risk/training"
    )

    # Build the args list that will be passed to train.py inside the container
    container_args: list[str] = [
        "--max_depth", str(max_depth),
        "--learning_rate", str(learning_rate),
        "--n_estimators", str(n_estimators),
        "--subsample", str(subsample),
        "--colsample_bytree", str(colsample_bytree),
        "--model_dir", model_dir,
        "--project_id", config["project_id"],
        "--region", config["region"],
    ]

    job = aiplatform.CustomTrainingJob(
        display_name=DISPLAY_NAME_TRAINING,
        # Path to the training script relative to the package root.
        # In a production setup this would be a URI inside the staging bucket
        # (via script_path) or baked into a custom container image.
        script_path="src/train.py",
        container_uri=config["train_image"],
        requirements=[
            "google-cloud-aiplatform==1.58.0",
            "scikit-learn==1.4.2",
            "xgboost==2.0.3",
            "pandas==2.2.2",
            "cloudml-hypertune==0.1.0.dev6",
            "joblib==1.4.2",
        ],
        model_serving_container_image_uri=None,  # no online serving endpoint needed
    )

    logger.info("Submitting Custom Training Job: %s", DISPLAY_NAME_TRAINING)

    run_kwargs: dict = dict(
        args=container_args,
        replica_count=1,
        machine_type=MACHINE_TYPE,
        sync=sync,
    )
    if config["service_account"]:
        run_kwargs["service_account"] = config["service_account"]
    if config["network"]:
        run_kwargs["network"] = config["network"]

    job.run(**run_kwargs)  # type: ignore[arg-type]

    logger.info(
        "Custom Training Job completed. Resource name: %s", job.resource_name
    )
    return job


# ---------------------------------------------------------------------------
# Job 2 — Hyperparameter Tuning Job (Vertex Vizier)
# ---------------------------------------------------------------------------

def submit_hpt_job(
    config: dict[str, str],
    sync: bool = True,
) -> aiplatform.HyperparameterTuningJob:
    """
    Configure and submit a Vertex AI Hyperparameter Tuning Job.

    Vertex Vizier will orchestrate `max_trial_count` trials, each running
    `src/train.py` with a different combination of hyperparameters sampled
    from the defined search space.  The primary metric is `f1_score` (maximize).

    Parameters
    ----------
    config:
        Runtime config dict produced by _load_config().
    sync:
        If True (default) block until all trials finish.

    Returns
    -------
    The submitted HyperparameterTuningJob instance.
    """
    logger.info("Initialising Vertex AI SDK — project=%s region=%s",
                config["project_id"], config["region"])

    aiplatform.init(
        project=config["project_id"],
        location=config["region"],
        experiment=EXPERIMENT_NAME,
        staging_bucket=f"gs://{config['bucket_name']}",
    )

    model_dir: str = (
        f"gs://{config['bucket_name']}/models/credit-risk/hpt"
    )

    # Fixed args that are NOT part of the search space
    base_args: list[str] = [
        "--model_dir", model_dir,
        "--project_id", config["project_id"],
        "--region", config["region"],
    ]

    # ------------------------------------------------------------------
    # 1. Build the CustomJob using from_local_script (handles packaging)
    # ------------------------------------------------------------------
    custom_job = aiplatform.CustomJob.from_local_script(
        display_name=f"{DISPLAY_NAME_HPT}-trial",
        script_path="src/train.py",
        container_uri=config["train_image"],
        args=base_args,
        replica_count=1,
        machine_type=MACHINE_TYPE,
        requirements=[
            "google-cloud-aiplatform==1.58.0",
            "scikit-learn==1.4.2",
            "xgboost==2.0.3",
            "pandas==2.2.2",
            "cloudml-hypertune==0.1.0.dev6",
            "joblib==1.4.2",
        ],
        staging_bucket=f"gs://{config['bucket_name']}",
    )

    # ------------------------------------------------------------------
    # 2. Define the hyperparameter search space
    # ------------------------------------------------------------------
    parameter_spec: dict[str, hpt._ParameterSpec] = {
        "learning_rate": hpt.DoubleParameterSpec(
            min=0.01,
            max=0.3,
            scale="log",         # log-uniform sampling suits learning rates
        ),
        "max_depth": hpt.IntegerParameterSpec(
            min=2,
            max=8,
            scale="linear",
        ),
        "n_estimators": hpt.IntegerParameterSpec(
            min=50,
            max=300,
            scale="linear",
        ),
        "subsample": hpt.DoubleParameterSpec(
            min=0.5,
            max=1.0,
            scale="linear",
        ),
        "colsample_bytree": hpt.DoubleParameterSpec(
            min=0.5,
            max=1.0,
            scale="linear",
        ),
    }

    # ------------------------------------------------------------------
    # 3. Define the metric to optimise
    # ------------------------------------------------------------------
    metric_spec: dict[str, str] = {
        "f1_score": "maximize",
    }

    # ------------------------------------------------------------------
    # 4. Build and submit the HPT job
    # ------------------------------------------------------------------
    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=DISPLAY_NAME_HPT,
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=MAX_TRIAL_COUNT,
        parallel_trial_count=PARALLEL_TRIAL_COUNT,
        search_algorithm=None,   # None → Vertex Vizier default (Bayesian)
    )

    logger.info(
        "Submitting HPT Job: %s (%d trials, %d parallel)",
        DISPLAY_NAME_HPT,
        MAX_TRIAL_COUNT,
        PARALLEL_TRIAL_COUNT,
    )

    run_kwargs: dict = dict(sync=sync)
    if config["service_account"]:
        run_kwargs["service_account"] = config["service_account"]

    hpt_job.run(**run_kwargs)  # type: ignore[arg-type]

    logger.info(
        "HPT Job completed. Resource name: %s", hpt_job.resource_name
    )
    return hpt_job


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertex AI Classic ML Orchestrator — Credit Risk"
    )
    parser.add_argument(
        "--job-type",
        choices=["training", "hpt", "all"],
        default="training",
        help=(
            "Which job to submit: 'training' (Custom Training Job), "
            "'hpt' (Hyperparameter Tuning Job), or 'all' (both sequentially)."
        ),
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        default=False,
        help="Return immediately after job submission without waiting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sync: bool = not args.no_sync

    try:
        config = _load_config()
    except RuntimeError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    if args.job_type in ("training", "all"):
        submit_custom_training_job(config, sync=sync)

    if args.job_type in ("hpt", "all"):
        submit_hpt_job(config, sync=sync)


if __name__ == "__main__":
    main()
