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

## Step 1 — Create and populate the Vector Search index

The retrieval layer needs a live Vertex AI Vector Search index that contains your
embeddings, plus the original document texts stored in GCS so matched IDs can be
resolved back into text at query time.

```bash
# 1. Create the bucket
gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=us-central1

# 2. Create local .env (gitignored) from the template and fill in real values
copy .env.example .env
#   GCP_PROJECT_ID=...
#   GCP_REGION=us-central1
#   GCS_BUCKET_NAME=YOUR_BUCKET_NAME
#   VERTEX_INDEX_ID=placeholder   (until created)
#   VERTEX_ENDPOINT_ID=placeholder (until deployed)

# 3. Run the ingestion pipeline. This:
#    - embeds the sample claims with text-embedding-004
#    - uploads vectors to gs://YOUR_BUCKET_NAME/vector_search/vertex_vectors.json
#    - stores each chunk's text at gs://YOUR_BUCKET_NAME/documents/{id}
pip install -r requirements.txt
python pipeline/build_vector_index.py
```

> **File format matters**: Vertex AI Vector Search requires a `.json` extension
> (it parses JSON-Lines content). The `vector_search/` folder must contain ONLY
> vector-data files — the pipeline puts the metadata map under `metadata/` and the
> document texts under `documents/` to avoid ingestion errors.

### 1a. Create the index (Console or CLI)

In **Cloud Console → Vertex AI → Vector Search → Create Index**:

| Field                         | Value                        | Notes                                                              |
| ----------------------------- | ---------------------------- | ------------------------------------------------------------------ |
| Dimensions                    | `768`                        | Matches `text-embedding-004` output                                |
| Approximate neighbors count   | `150`                        | Candidate pool for the approximate stage before exact reranking    |
| Distance measure              | `Dot product distance`       | Recommended for these embeddings                                   |
| Algorithm type                | `Tree-AH`                    | Approximate nearest neighbor                                       |
| Update method                 | `Batch`                      | Index is built from the GCS data folder                            |
| Shard size                    | `Small`                      | Fine for this demo data volume                                     |
| Data source (Cloud Storage)   | `gs://YOUR_BUCKET_NAME/vector_search/` | Folder containing `vertex_vectors.json`                  |

> **Note on `Approximate neighbors count`**: this is the number of candidate
> neighbors the Tree-AH approximate search gathers before exact distance
> reranking returns the final top-k. It must be larger than the query `k`
> (the app queries with `k=3`). `150` is the conventional default and is sized
> for a real dataset; with the small sample set it simply returns all available
> vectors.

If you created the index empty (before uploading data), populate it afterwards via
CLI using a metadata file (`pipeline/index_update_metadata.json` points at the
`vector_search/` folder):

```bash
gcloud ai indexes update YOUR_INDEX_ID \
  --region us-central1 \
  --metadata-file=pipeline/index_update_metadata.json

# Verify the vectors landed
gcloud ai indexes describe YOUR_INDEX_ID --region us-central1 --format="yaml(indexStats)"
```

### 1b. Deploy the index to an endpoint

```bash
# Create the endpoint in the Console, or reuse an existing one, then:
gcloud ai index-endpoints deploy-index YOUR_ENDPOINT_ID \
  --region us-central1 \
  --index=YOUR_INDEX_ID \
  --deployed-index-id=ro_fraud_deployed \
  --display-name="RO Fraud Deployed Index"
```

This is long-running (~15-30 min) and blocks the terminal. Poll for completion in
another shell:

```bash
gcloud ai index-endpoints describe YOUR_ENDPOINT_ID \
  --region us-central1 --format="value(deployedIndexes[].id)"
# empty = still deploying; "ro_fraud_deployed" = live
```

### 1c. Note the IDs and update `.env`

```bash
gcloud ai indexes list --region=us-central1
gcloud ai index-endpoints list --region=us-central1
```

> Cost note: a deployed endpoint bills continuously. Tear it down right after the demo (Step 4).

---

## Step 1d — Grant the Cloud Run service account bucket access

The service account that runs Cloud Run needs to read the bucket. The online
`VectorSearchVectorStore` calls `get_bucket()`, which requires `storage.buckets.get`
(not covered by `objectViewer` alone):

```bash
SA="ro-fraud-api-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET_NAME \
  --member="serviceAccount:$SA" --role="roles/storage.objectViewer"
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET_NAME \
  --member="serviceAccount:$SA" --role="roles/storage.legacyBucketReader"
```

---

## Step 2 — Deploy the API (simplest path)

One command. Cloud Run builds the container from source, pushes it, and deploys it.

```bash
gcloud run deploy ro-fraud-service \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_REGION=us-central1,GCS_BUCKET_NAME=YOUR_BUCKET_NAME,VERTEX_INDEX_ID=YOUR_INDEX_ID,VERTEX_ENDPOINT_ID=YOUR_ENDPOINT_ID,GEMINI_MODEL=gemini-2.5-flash"
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
  --deployed-index-id=ro_fraud_deployed --region=us-central1
gcloud ai index-endpoints delete YOUR_ENDPOINT_ID --region=us-central1 --quiet
gcloud ai indexes delete YOUR_INDEX_ID --region=us-central1 --quiet

# 3. (Optional) delete the bucket
gcloud storage rm --recursive gs://YOUR_BUCKET_NAME
```

---

## Troubleshooting (quick)

These reflect issues actually hit while deploying this system:

- **Cloud Run won't start**: check `gcloud run services logs read ro-fraud-service --region us-central1`. Usual causes: missing env var, container not listening on `8080`.
- **`storage.buckets.get` denied**: the runtime SA needs bucket read. Grant `roles/storage.objectViewer` AND `roles/storage.legacyBucketReader` on the bucket (see Step 1d). `objectViewer` alone does not cover the `get_bucket()` call the vector store makes.
- **`Document with id ... not found in document storage`**: the index returned a match but its text isn't in GCS. Ensure the pipeline wrote `documents/{id}` blobs (Step 1, `store_documents_for_retrieval`) with IDs identical to the embedding IDs.
- **`404 Publisher Model gemini-1.5-pro was not found`**: the 1.5 models are retired in newer projects. Use a current model via `GEMINI_MODEL` (default `gemini-2.5-flash`).
- **`unknown format` on index update**: the data file must end in `.json` (not `.jsonl`), and the `vector_search/` folder must contain only vector files.
- **Readiness fails / no results**: confirm the index is *deployed* to the endpoint (`deployedIndexes[].id` is non-empty) and the IDs in env vars match `gcloud ai index-endpoints list`.

---

## Full production setup (reference only)

For a real production deployment, the repo also includes:
- `infrastructure/main.tf` — Terraform for GCS, Artifact Registry, least-privilege SA, Cloud Run.
- `cloudbuild.yaml` — CI/CD pipeline (test → scan → build → deploy).
- `docs/COST_OPTIMIZATION.md` — cost levers and estimates.

These are not required for the simple demo flow above.
