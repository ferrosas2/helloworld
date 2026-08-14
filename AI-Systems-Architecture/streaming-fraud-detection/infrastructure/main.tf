# =============================================================================
# Terraform: Streaming Fraud Detection Infrastructure (GCP)
# =============================================================================
# Provisions all GCP resources needed for the end-to-end pipeline:
#   - Pub/Sub topics and subscriptions (ingestion + flagged routing)
#   - BigQuery dataset and tables (analytical warehouse)
#   - Cloud Storage bucket (embeddings, documents, pipeline artifacts)
#   - Artifact Registry (Docker images)
#   - Cloud Run service (FastAPI serving)
#   - Service Account with least-privilege IAM
#   - Dataflow job template (streaming pipeline)
# =============================================================================

terraform {
  required_version = ">= 1.3.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# =============================================================================
# Variables
# =============================================================================

variable "project_id" {
  type        = string
  description = "Google Cloud Project ID"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "GCP region for all resources"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Environment name (production, staging)"
}

variable "gcs_bucket_name" {
  type        = string
  description = "GCS bucket for embeddings, documents, and pipeline artifacts"
}

variable "vertex_prediction_endpoint_id" {
  type        = string
  description = "Vertex AI Online Prediction endpoint ID for fraud classifier"
}

variable "vertex_index_id" {
  type        = string
  description = "Vertex AI Vector Search Index ID"
}

variable "vertex_endpoint_id" {
  type        = string
  description = "Vertex AI Vector Search Index Endpoint ID"
}

variable "cloud_run_image_tag" {
  type        = string
  default     = "latest"
  description = "Docker image tag for Cloud Run deployment"
}

# =============================================================================
# 1. Pub/Sub: Ingestion & Routing Topics
# =============================================================================

resource "google_pubsub_topic" "pos_transactions" {
  name = "pos-transactions"

  message_retention_duration = "86400s" # 24 hours

  labels = {
    team        = "fraud-detection"
    environment = var.environment
    component   = "ingestion"
  }
}

resource "google_pubsub_subscription" "pos_transactions_dataflow" {
  name  = "pos-transactions-dataflow-sub"
  topic = google_pubsub_topic.pos_transactions.id

  # Exactly-once delivery for Dataflow
  enable_exactly_once_delivery = true

  # Acknowledge deadline (Dataflow manages this internally)
  ack_deadline_seconds = 60

  # Retain unacked messages for 24 hours (disaster recovery)
  message_retention_duration = "86400s"
  retain_acked_messages      = false

  # Exponential backoff for retries
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  # Dead letter policy: move undeliverable messages after 5 attempts
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dead_letter.id
    max_delivery_attempts = 5
  }

  labels = {
    team    = "fraud-detection"
    consumer = "dataflow"
  }
}

resource "google_pubsub_topic" "flagged_transactions" {
  name = "flagged-transactions"

  message_retention_duration = "172800s" # 48 hours (critical path)

  labels = {
    team        = "fraud-detection"
    environment = var.environment
    component   = "routing"
  }
}

resource "google_pubsub_subscription" "flagged_deep_analysis" {
  name  = "flagged-transactions-deep-analysis-sub"
  topic = google_pubsub_topic.flagged_transactions.id

  ack_deadline_seconds       = 120 # Deep analysis takes 1-2s, leave buffer
  message_retention_duration = "172800s"

  # Push subscription to Cloud Run (auto-triggers deep analysis)
  push_config {
    push_endpoint = "${google_cloud_run_v2_service.fraud_api.uri}/api/v1/deep-analysis-push"

    oidc_token {
      service_account_email = google_service_account.fraud_api_sa.email
    }
  }

  labels = {
    team    = "fraud-detection"
    consumer = "cloud-run-deep-analysis"
  }
}

resource "google_pubsub_topic" "dead_letter" {
  name = "fraud-detection-dead-letter"

  labels = {
    team      = "fraud-detection"
    component = "dead-letter"
  }
}

# =============================================================================
# 2. BigQuery: Analytical Warehouse
# =============================================================================

resource "google_bigquery_dataset" "fraud_detection" {
  dataset_id    = "fraud_detection"
  friendly_name = "Fraud Detection"
  description   = "Real-time POS fraud detection data warehouse"
  location      = var.region

  default_table_expiration_ms = 31536000000 # 365 days

  labels = {
    team        = "fraud-detection"
    environment = var.environment
  }
}

# =============================================================================
# 3. Cloud Storage: Artifacts & Embeddings
# =============================================================================

resource "google_storage_bucket" "fraud_artifacts" {
  name                        = var.gcs_bucket_name
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      num_newer_versions = 3
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age = 90 # Move old embeddings to cheaper storage after 90 days
    }
  }

  labels = {
    team        = "fraud-detection"
    environment = var.environment
  }
}

# =============================================================================
# 4. Artifact Registry: Docker Images
# =============================================================================

resource "google_artifact_registry_repository" "fraud_api" {
  location      = var.region
  repository_id = "streaming-fraud-detection"
  description   = "Docker images for streaming fraud detection API and pipeline"
  format        = "DOCKER"

  cleanup_policies {
    id     = "keep-recent-images"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }

  labels = {
    team        = "fraud-detection"
    environment = var.environment
  }
}

# =============================================================================
# 5. Service Account: Least-Privilege IAM
# =============================================================================

resource "google_service_account" "fraud_api_sa" {
  account_id   = "streaming-fraud-api-sa"
  display_name = "Streaming Fraud Detection API Service Account"
  description  = "Least-privilege SA for Cloud Run fraud detection service"
}

resource "google_service_account" "dataflow_sa" {
  account_id   = "streaming-fraud-dataflow-sa"
  display_name = "Streaming Fraud Detection Dataflow Service Account"
  description  = "SA for Dataflow streaming pipeline workers"
}

# --- Cloud Run SA IAM Bindings (Least Privilege) ---

resource "google_project_iam_member" "api_gcs_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.fraud_api_sa.email}"
}

resource "google_project_iam_member" "api_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.fraud_api_sa.email}"
}

resource "google_project_iam_member" "api_pubsub_subscriber" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.fraud_api_sa.email}"
}

# --- Dataflow SA IAM Bindings ---

resource "google_project_iam_member" "dataflow_worker" {
  project = var.project_id
  role    = "roles/dataflow.worker"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_project_iam_member" "dataflow_pubsub" {
  project = var.project_id
  role    = "roles/pubsub.editor"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_project_iam_member" "dataflow_bq_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_project_iam_member" "dataflow_bq_jobuser" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_project_iam_member" "dataflow_gcs" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_project_iam_member" "dataflow_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

# =============================================================================
# 6. Cloud Run: FastAPI Serving Layer
# =============================================================================

resource "google_cloud_run_v2_service" "fraud_api" {
  name                = "streaming-fraud-api"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.fraud_api_sa.email

    scaling {
      min_instance_count = var.environment == "production" ? 1 : 0
      max_instance_count = var.environment == "production" ? 50 : 10
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.fraud_api.repository_id}/streaming-fraud-api:${var.cloud_run_image_tag}"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle = true # Only allocate CPU during request processing
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "GCP_REGION"
        value = var.region
      }
      env {
        name  = "GCS_BUCKET_NAME"
        value = google_storage_bucket.fraud_artifacts.name
      }
      env {
        name  = "VERTEX_PREDICTION_ENDPOINT_ID"
        value = var.vertex_prediction_endpoint_id
      }
      env {
        name  = "VERTEX_INDEX_ID"
        value = var.vertex_index_id
      }
      env {
        name  = "VERTEX_ENDPOINT_ID"
        value = var.vertex_endpoint_id
      }
      env {
        name  = "GEMINI_MODEL"
        value = "gemini-2.5-flash"
      }
      env {
        name  = "HIGH_RISK_THRESHOLD"
        value = "0.7"
      }
      env {
        name  = "MEDIUM_RISK_THRESHOLD"
        value = "0.3"
      }
      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }

      # Startup probe to verify GCP connectivity before receiving traffic
      startup_probe {
        http_get {
          path = "/readiness"
        }
        initial_delay_seconds = 5
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      # Liveness probe
      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds    = 30
        timeout_seconds   = 5
        failure_threshold = 3
      }
    }

    # Maximum request timeout
    timeout = "300s"
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  labels = {
    team        = "fraud-detection"
    environment = var.environment
  }
}

# Public access (for demo/internal use — restrict in production)
resource "google_cloud_run_v2_service_iam_member" "allow_unauthenticated" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.fraud_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# =============================================================================
# Outputs
# =============================================================================

output "api_endpoint" {
  value       = google_cloud_run_v2_service.fraud_api.uri
  description = "Cloud Run API endpoint URL"
}

output "pubsub_ingestion_topic" {
  value       = google_pubsub_topic.pos_transactions.id
  description = "Pub/Sub topic for POS transaction ingestion"
}

output "pubsub_flagged_topic" {
  value       = google_pubsub_topic.flagged_transactions.id
  description = "Pub/Sub topic for flagged high-risk transactions"
}

output "gcs_bucket_url" {
  value       = google_storage_bucket.fraud_artifacts.url
  description = "GCS bucket URL for artifacts"
}

output "artifact_registry_url" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.fraud_api.repository_id}"
  description = "Artifact Registry URL for Docker images"
}

output "bigquery_dataset" {
  value       = google_bigquery_dataset.fraud_detection.dataset_id
  description = "BigQuery dataset for fraud detection"
}

output "dataflow_service_account" {
  value       = google_service_account.dataflow_sa.email
  description = "Dataflow pipeline service account email"
}
