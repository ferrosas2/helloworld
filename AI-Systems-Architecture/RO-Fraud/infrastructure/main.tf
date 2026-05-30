# Terraform configuration block
terraform {
  required_version = ">= 1.3.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.50.0"
    }
  }
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

# Input Variables for parameterized provisioning
variable "project_id" {
  type        = string
  description = "The Google Cloud Project ID to provision resources in."
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "The Google Cloud Region for resources."
}

variable "gcs_bucket_name" {
  type        = string
  description = "Globally unique name for the GCS Bucket storing vector embeddings."
}

variable "artifact_registry_repo_id" {
  type        = string
  default     = "ro-fraud-repo"
  description = "ID of the Google Artifact Registry repository."
}

variable "cloud_run_service_name" {
  type        = string
  default     = "ro-fraud-service"
  description = "The deployment name of the Cloud Run API scale-to-zero microservice."
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = "Docker image tag to deploy."
}

variable "vertex_index_id" {
  type        = string
  description = "Vertex AI Vector Search Index ID (to pass to Cloud Run as env variable)."
}

variable "vertex_endpoint_id" {
  type        = string
  description = "Vertex AI Vector Search Index Endpoint ID (to pass to Cloud Run as env variable)."
}

# 1. Google Cloud Storage Bucket for Golden Dataset Embeddings (RAID-level, secure, with lifecycle rule)
resource "google_storage_bucket" "embeddings_bucket" {
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
}

# 2. Google Artifact Registry repository to hold the FastAPI Docker image
resource "google_artifact_registry_repository" "api_repo" {
  location      = var.region
  repository_id = var.artifact_registry_repo_id
  description   = "Docker repository for ROI/RO-Fraud AI Systems Cloud Run container images"
  format        = "DOCKER"
}

# 3. Secure Service Account for running the Cloud Run microservice safely with minimal required permissions
resource "google_service_account" "cloud_run_sa" {
  account_id   = "ro-fraud-api-sa"
  display_name = "RO-Fraud FastAPI Cloud Run Service Account"
}

# Grant necessary IAM permissions to the Cloud Run Service Account (Least Privilege)
resource "google_project_iam_member" "gcs_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_project_iam_member" "vertex_reader" {
  project = var.project_id
  role    = "roles/aiplatform.viewer"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_project_iam_member" "vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# 4. Google Cloud Run Service deploying the FastAPI application securely
resource "google_cloud_run_v2_service" "fastapi_api" {
  name     = var.cloud_run_service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run_sa.email

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.api_repo.repository_id}/${var.cloud_run_service_name}:${var.image_tag}"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
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
        value = google_storage_bucket.embeddings_bucket.name
      }
      env {
        name  = "VERTEX_INDEX_ID"
        value = var.vertex_index_id
      }
      env {
        name  = "VERTEX_ENDPOINT_ID"
        value = var.vertex_endpoint_id
      }
    }
  }

  # Allow the system to automatically handle scaling/rolling parameters
  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

# 5. Public access IAM Policy to allow external invocation of the Cloud Run API if user needs public availability,
# or restricted. Here, we present a template for public exposure (representing operations UI consumption)
# with a clear note.
resource "google_cloud_run_v2_service_iam_member" "allow_unauthenticated" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.fastapi_api.name
  role     = "roles/run.invoker"
  member   = "allUsers" # Can be targeted to specific domains or users for high compliance environments
}

# Outputs for useful post-deployment operational integration
output "gcs_bucket_url" {
  value       = google_storage_bucket.embeddings_bucket.url
  description = "Reference GCS Bucket URL for vector search ingestion."
}

output "artifact_registry_repository_url" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.api_repo.repository_id}"
  description = "Docker Registry Endpoint URL for deployment push."
}

output "api_endpoint" {
  value       = google_cloud_run_v2_service.fastapi_api.uri
  description = "Secure Public HTTPS Endpoint URL for the newly deployed Cloud Run Service."
}
