# Deployment Guide

This guide covers the **simplest path** to deploy RO-Fraud on GCP for a working demo,
then tear it down. It uses Cloud Run's source-based deploy (no local Docker build, no
Terraform required). Terraform and `cloudbuild.yaml` are included in the repo as
reference for a full production setup, but are not needed for this flow.

> All values like `YOUR_PROJECT_ID` are placeholders. Never commit real project IDs,
> bucket names, or endpoint IDs to the repo. Put real values in a local `.env` (gitignored).

---

## Prerequisites

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login

# Enable required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com
```

---

## Step 1 — Create the Vector Search index

The retrieval layer needs a live Vertex AI Vector Search index.

```bash
# Create a bucket and run the ingestion pipeline (embeds sample claims, uploads JSONL)
gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=us-central1

# Create local .env (gitignored) with your real values
#   GCP_PROJECT_ID=...
#   GCP_REGION=us-central1
#   GCS_BUCKET_NAME=YOUR_BUCKET_NAME
#   VERTEX_INDEX_ID=placeholder
#   VERTEX_ENDPOINT_ID=placeholder

pip install -r requirements.txt
python pipeline/build_vector_index.py
```

Then in the **Cloud Console → Vertex AI → Vector Search**:
1. Create Index — dimensions `768`, distance `DOT_PRODUCT_DISTANCE`, source `gs://YOUR_BUCKET_NAME/vector_search/vertex_vectors.jsonl` (~45 min).
2. Deploy the index to a new endpoint (~15 min).
3. Note the **Index ID** and **Endpoint ID**:
   ```bash
   gcloud ai indexes list --region=us-central1
   gcloud ai index-endpoints list --region=us-central1
   ```
4. Update your local `.env` with the real IDs.

> Cost note: a deployed endpoint bills continuously. Tear it down right after the demo (Step 4).

---

## Step 2 — Deploy the API (simplest path)

One command. Cloud Run builds the container from source, pushes it, and deploys it.

```bash
gcloud run deploy ro-fraud-service \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_REGION=us-central1,GCS_BUCKET_NAME=YOUR_BUCKET_NAME,VERTEX_INDEX_ID=YOUR_INDEX_ID,VERTEX_ENDPOINT_ID=YOUR_ENDPOINT_ID"
```

Grab the URL:
```bash
SERVICE_URL=$(gcloud run services describe ro-fraud-service --region us-central1 --format='value(status.url)')
echo $SERVICE_URL
```

> `--allow-unauthenticated` makes the endpoint public, which is fine for a short demo.
> For anything beyond a rehearsal, drop that flag and require authenticated invocation.

---

## Step 3 — Test it

```bash
# Health
curl $SERVICE_URL/health

# Readiness (verifies Vertex connectivity)
curl $SERVICE_URL/readiness

# Analyze a claim
curl -X POST $SERVICE_URL/api/v1/analyze-claim \
  -H "Content-Type: application/json" \
  -d '{"claim_id":"C-DEMO-001","customer_id":"CUST-DEMO","claim_text":"Car stolen from lot, no police report filed.","claim_amount":45000.0}'
```

---

## Step 4 — Teardown (do this after the demo to stop billing)

```bash
# 1. Delete the Cloud Run service
gcloud run services delete ro-fraud-service --region us-central1 --quiet

# 2. Undeploy + delete the Vector Search endpoint (the main cost) — via Console or:
gcloud ai index-endpoints undeploy-index YOUR_ENDPOINT_ID \
  --deployed-index-id=YOUR_DEPLOYED_INDEX_ID --region=us-central1
gcloud ai index-endpoints delete YOUR_ENDPOINT_ID --region=us-central1 --quiet
gcloud ai indexes delete YOUR_INDEX_ID --region=us-central1 --quiet

# 3. (Optional) delete the bucket
gcloud storage rm --recursive gs://YOUR_BUCKET_NAME
```

---

## Troubleshooting (quick)

- **Cloud Run won't start**: check `gcloud run services logs read ro-fraud-service --region us-central1`. Usual causes: missing env var, container not listening on `8080`.
- **Readiness fails / no results**: confirm the index is *deployed* to the endpoint and the IDs in env vars match `gcloud ai index-endpoints list`.
- **Permission denied on Vertex/GCS**: the Cloud Run runtime service account needs `roles/aiplatform.user` and `roles/storage.objectViewer`.

---

## Full production setup (reference only)

For a real production deployment, the repo also includes:
- `infrastructure/main.tf` — Terraform for GCS, Artifact Registry, least-privilege SA, Cloud Run.
- `cloudbuild.yaml` — CI/CD pipeline (test → scan → build → deploy).
- `docs/COST_OPTIMIZATION.md` — cost levers and estimates.

These are not required for the simple demo flow above.
