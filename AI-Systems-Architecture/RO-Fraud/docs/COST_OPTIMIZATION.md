# Cost Optimization Guide for RO-Fraud System

## Overview
This document outlines cost optimization strategies for the RO-Fraud RAG system deployed on GCP.

## Current Cost Drivers

### 1. **Vertex AI Vector Search**
- **Index Storage**: ~$0.30/GB/month
- **Query Costs**: ~$0.10 per 1,000 queries
- **Deployment**: ~$0.50/hour for deployed endpoint

**Optimization Strategies:**
- Use **batch queries** when possible to reduce per-query overhead
- Implement **caching** for frequently queried claims
- Consider **index refresh schedules** instead of real-time updates
- Use **streaming endpoints** only when needed; otherwise use batch endpoints

### 2. **Gemini 1.5 Pro (LLM Generation)**
- **Input tokens**: ~$0.00125 per 1K tokens
- **Output tokens**: ~$0.00375 per 1K tokens

**Optimization Strategies:**
- **Prompt optimization**: Reduce unnecessary context in prompts
- **Temperature=0.0**: Already implemented for deterministic, cost-effective responses
- **Token limits**: Set max_output_tokens to prevent runaway generation
- **Caching**: Cache responses for identical claims (implement Redis/Memorystore)

### 3. **Cloud Run**
- **CPU allocation**: Billed per 100ms of CPU time
- **Memory**: Billed per GB-second
- **Requests**: First 2 million requests/month free

**Current Configuration:**
```
CPU: 1 vCPU
Memory: 1 GiB
```

**Optimization Strategies:**
- **Scale to zero**: Already enabled (no cost when idle)
- **Concurrency**: Increase max concurrent requests per instance (default: 80)
- **Min instances**: Keep at 0 for dev/staging, 1-2 for production (warm start)
- **Request timeout**: Set appropriate timeout (currently default 300s)
- **CPU allocation**: Use "CPU is only allocated during request processing" (default)

### 4. **Cloud Storage (GCS)**
- **Standard Storage**: ~$0.020/GB/month
- **Operations**: Minimal cost for read/write operations

**Optimization Strategies:**
- Use **lifecycle policies** (already implemented: keep 3 versions)
- Archive old embeddings to **Nearline** or **Coldline** storage
- Compress JSONL files before upload

### 5. **Artifact Registry**
- **Storage**: ~$0.10/GB/month
- **Network egress**: Varies by region

**Optimization Strategies:**
- Delete old/unused image tags regularly
- Use **multi-stage Docker builds** to reduce image size
- Enable **vulnerability scanning** to avoid security costs

## Cost Monitoring Setup

### 1. Set Budget Alerts
```bash
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="RO-Fraud Monthly Budget" \
  --budget-amount=500 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100
```

### 2. Enable Cost Breakdown by Service
```bash
# View current month costs
gcloud billing accounts list
gcloud billing projects describe PROJECT_ID

# Export billing data to BigQuery for analysis
gcloud billing accounts set-billing-export \
  --billing-account=BILLING_ACCOUNT_ID \
  --dataset-id=billing_export \
  --project=PROJECT_ID
```

### 3. Use Cloud Monitoring for Resource Utilization
- Monitor **Cloud Run CPU/Memory utilization**
- Track **Vertex AI query volume and latency**
- Set alerts for **unusual spending patterns**

## Estimated Monthly Costs (Production)

### Low Traffic (1,000 requests/month)
- Cloud Run: ~$0 (within free tier)
- Vertex AI Vector Search: ~$360/month (deployed endpoint)
- Gemini API: ~$5-10/month
- GCS + Artifact Registry: ~$5/month
- **Total: ~$370-380/month**

### Medium Traffic (50,000 requests/month)
- Cloud Run: ~$20-30/month
- Vertex AI Vector Search: ~$360/month + $5 queries
- Gemini API: ~$200-300/month
- GCS + Artifact Registry: ~$10/month
- **Total: ~$595-705/month**

### High Traffic (500,000 requests/month)
- Cloud Run: ~$150-200/month
- Vertex AI Vector Search: ~$360/month + $50 queries
- Gemini API: ~$2,000-3,000/month
- GCS + Artifact Registry: ~$20/month
- **Total: ~$2,580-3,630/month**

## Cost Reduction Recommendations

### Immediate Actions (0-1 week)
1. ✅ Implement response caching with Redis/Memorystore
2. ✅ Add request rate limiting to prevent abuse
3. ✅ Set max_output_tokens limit on Gemini calls
4. ✅ Enable Cloud Run concurrency optimization

### Short-term (1-4 weeks)
1. ✅ Implement batch processing for bulk claim analysis
2. ✅ Add query result caching in Vector Search
3. ✅ Optimize Docker image size (multi-stage builds)
4. ✅ Set up cost anomaly detection alerts

### Long-term (1-3 months)
1. ✅ Evaluate **Gemini 1.5 Flash** for lower-cost alternative
2. ✅ Consider **reserved capacity** for predictable workloads
3. ✅ Implement **tiered service levels** (fast/standard/batch)
4. ✅ Explore **model distillation** for simpler cases

## Cost Optimization Checklist

- [ ] Budget alerts configured
- [ ] Billing export to BigQuery enabled
- [ ] Response caching implemented
- [ ] Rate limiting configured
- [ ] Max token limits set
- [ ] Docker image optimized (<500MB)
- [ ] Old Artifact Registry images cleaned
- [ ] GCS lifecycle policies verified
- [ ] Cloud Run concurrency tuned
- [ ] Monitoring dashboards created
- [ ] Cost anomaly alerts configured

## Additional Resources

- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing)
- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Cost Optimization Best Practices](https://cloud.google.com/architecture/cost-optimization)
