# Cost Optimization Guide: Streaming Fraud Detection

## Architecture Cost Overview

The streaming fraud detection system has a mix of **fixed costs** (always-on infrastructure) and **variable costs** (per-transaction).

### Cost Distribution by Component

| Component | Cost Type | Monthly Estimate (Medium Traffic) |
|-----------|-----------|----------------------------------|
| Vertex AI Vector Search Endpoint | Fixed | ~$360 |
| Cloud Dataflow (streaming) | Semi-fixed | ~$200-400 |
| Cloud Run (API serving) | Variable | ~$30-80 |
| BigQuery (storage + queries) | Variable | ~$20-50 |
| Pub/Sub (ingestion) | Variable | ~$5-15 |
| Vertex AI Online Prediction | Variable | ~$50-150 |
| Gemini 2.5 Flash (deep analysis) | Variable | ~$10-30 |
| GCS (storage) | Fixed | ~$5-10 |
| Artifact Registry | Fixed | ~$2-5 |
| **Total** | | **~$680-1,100/month** |

## Traffic-Based Cost Estimates

| Traffic Level | Daily Transactions | Monthly Cost | Cost per Transaction |
|--------------|-------------------|--------------|---------------------|
| Low | 10,000 | ~$420 | $0.0014 |
| Medium | 50,000 | ~$680 | $0.00045 |
| High | 500,000 | ~$1,200 | $0.00008 |
| Enterprise | 5,000,000 | ~$4,500 | $0.00003 |

> Economy of scale: per-transaction cost decreases 47x from low to enterprise.

## Cost Drivers (Ranked by Impact)

### 1. Vertex AI Vector Search Endpoint (~$360/month fixed)
**The dominant fixed cost** regardless of traffic volume.

**Optimization strategies:**
- Use batch endpoints for non-real-time retrieval (cheaper per query)
- Share the endpoint across multiple services
- Consider downgrading to a smaller machine type for low-traffic periods
- Evaluate if FAISS (self-hosted) is viable for your scale

### 2. Cloud Dataflow Workers (~$200-400/month)
Streaming pipelines keep workers running continuously.

**Optimization strategies:**
- Set `max_num_workers` carefully (start with 3, autoscale to 20)
- Use `n1-standard-2` or `e2-standard-2` worker types (cheaper)
- Enable Dataflow Prime for automatic right-sizing
- Monitor worker CPU utilization — over-provisioning wastes money

### 3. Vertex AI Online Prediction (~$50-150/month)
Per-prediction cost for the fraud classifier endpoint.

**Optimization strategies:**
- Enable prediction caching for repeat card tokens within short windows
- Use smaller machine types for the serving endpoint
- Consider batch prediction for re-scoring (10x cheaper than online)
- Enable traffic splitting for A/B testing without doubling endpoint cost

### 4. Gemini 2.5 Flash (~$10-30/month)
Only triggered for high-risk transactions (score >= 0.7, ~5% of traffic).

**Why Flash over Pro:**
- Input: $0.00015/1K tokens vs $0.00125/1K tokens (8x cheaper)
- Output: $0.0006/1K tokens vs $0.005/1K tokens (8x cheaper)
- Latency: comparable for short structured outputs
- Quality: sufficient for fraud risk summaries (not creative writing)

**Optimization strategies:**
- Temperature=0.0 (deterministic = cacheable)
- max_output_tokens=1024 (prevents runaway generation)
- Cache responses for identical transaction patterns
- Batch multiple flagged transactions into a single LLM call where possible

### 5. Cloud Run (~$30-80/month)
Serverless — only pay during request processing.

**Current configuration:**
- 2 vCPU, 2 GiB memory per instance
- Concurrency: 80 requests/instance
- Min instances: 1 (production), 0 (staging)

**Optimization strategies:**
- Scale to zero in staging (already done)
- Tune concurrency (higher = fewer instances needed)
- Use CPU-only allocation during requests (already done via `cpu_idle = true`)
- Right-size memory (monitor actual usage, reduce if possible)

## Immediate Actions (Week 1)

1. **Response caching (Redis/Memorystore)**: Cache fast-path scores for identical card_token + merchant within 5-min window
2. **Rate limiting**: Prevent abuse (Cloud Armor or application-level)
3. **Budget alerts**: Set at $500, $800, $1000 thresholds
4. **Billing export to BigQuery**: Enable for detailed cost analysis

## Short-Term Optimizations (Month 1)

1. **Dataflow worker right-sizing**: Monitor CPU/memory, adjust machine types
2. **Prediction endpoint caching**: Enable Vertex AI prediction caching
3. **Docker image optimization**: Multi-stage build, <500MB final image
4. **Dead letter monitoring**: Ensure failed messages aren't re-processed (cost leak)

## Long-Term Strategy (Quarter 1)

1. **Model distillation**: Train a smaller, faster model for obvious low-risk transactions
2. **Tiered scoring**: Skip Vertex AI entirely for transactions matching safe patterns
3. **Reserved capacity**: Committed use discounts for Dataflow and Vertex AI
4. **Regional optimization**: Deploy in cheapest region that meets latency SLOs

## Cost Monitoring Commands

```bash
# Set budget alert
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Streaming Fraud Detection Budget" \
  --budget-amount=1000 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=80 \
  --threshold-rule=percent=100

# Export billing to BigQuery
gcloud billing accounts set-billing-export \
  --billing-account=BILLING_ACCOUNT_ID \
  --dataset-id=billing_export

# View Dataflow job costs
gcloud dataflow jobs list --region=us-central1 --format="table(id,name,state)"
```
