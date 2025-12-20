# Quick Start Guide: Two-Stage Ranking System

## 🚀  Demo

### 1. Clone & Setup (30 seconds)
```bash
cd two-stage-ranking
pip install -r requirements.txt
```

### 2. Train Model Locally (2 minutes)
```bash
python src/train.py \
  --bucket ltr-models-frp \
  --key data/ltr_training_data.csv \
  --output-dir ./models
```

### 3. Test Inference (10 seconds)
```bash
python src/inference.py --model-path ./models/model.json
```

### 4. Run Full Demo (3 minutes)
```bash
python examples/demo_pipeline.py
```

---

## 🎯 Key Points 
### Problem Understanding
- **Challenge**: Ranking unique auction items without click history
- **Scale**: 100K+ items, <100ms latency requirement
- **Complexity**: Multi-objective optimization (relevance + margin + diversity)

### Technical Architecture
- **Stage 1**: OpenSearch vector search (100K → 100 items in ~20ms)
- **Stage 2**: XGBoost LambdaMART re-ranking (100 → 10 items in ~50ms)
- **Total Latency**: ~70ms end-to-end

### Production Readiness
✅ Modular code with argparse CLI
✅ Docker containerization
✅ SageMaker-compatible handlers
✅ Comprehensive logging
✅ Type hints and docstrings
✅ Infrastructure-as-Code templates

### MLOps Best Practices
- **Training**: Reproducible with versioned data and hyperparameters
- **Deployment**: Multi-option (SageMaker, Lambda, ECS)
- **Monitoring**: Data capture enabled for drift detection
- **Testing**: Unit tests scaffolded

### Business Impact
- **Relevance**: NDCG@10 optimization ensures top results
- **Profitability**: Features include cost/margin for revenue optimization
- **Scalability**: Designed for millions of queries/day

---

## 📊 Demo Flow

1. **Show Data**: Explain query groups and relevance labels
2. **Train Model**: Run training script, explain LambdaMART
3. **Inference**: Show sample ranking with business features
4. **Architecture**: Diagram two-stage pipeline
5. **Deployment**: Discuss Docker + SageMaker options

---

## 🔥 Further improvements

**Q: How would you handle cold-start for new items?**
A: Use content-based features (category, price, seller reputation) which don't require historical data. Stage 1 retrieval can use item embeddings from product descriptions.

**Q: How do you prevent overfitting in production?**
A: Cross-validation with different query groups, regularization (gamma, min_child_weight), monitor NDCG on holdout set daily.

**Q: What if OpenSearch returns < 10 candidates?**
A: Fallback logic: (1) Relax filters, (2) Use popularity baseline, (3) Return "no results" with alternative suggestions.

**Q: How would you A/B test this?**
A: Traffic splitting at API gateway level, log both rankings, measure click-through rate, revenue per session, user engagement.

**Q: What features would you add next?**
A: User context (location, device), temporal features (time of day), item velocity (views/hour), cross-category embeddings.

---

## 📁 File Structure Reference

```
two-stage-ranking/
├── src/
│   ├── train.py              # Training pipeline
│   └── inference.py          # Inference handler
├── infrastructure/
│   ├── sagemaker-training.json
│   └── sagemaker-endpoint.json
├── examples/
│   └── demo_pipeline.py      # End-to-end demo
├── notebooks/
│   └── ltr.ipynb             # Exploratory analysis
├── tests/
│   └── test_inference.py     # Unit tests
├── Dockerfile
├── requirements.txt
├── setup.py
└── README.md
```




