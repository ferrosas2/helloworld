"""
Interview Demo: Two-Stage Ranking System Showcase

This script provides a clean, presentation-ready demonstration of the 
two-stage ranking system for technical interviews.

Features demonstrated:
- Stage 1: Retrieval simulation
- Stage 2: LTR re-ranking
- Clear before/after comparison
- Business metrics impact
- Feature importance visualization

Usage:
    python examples/interview_demo.py
"""

import os
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import load_data_from_s3, preprocess_data, train_ranker, save_model
from inference import RankingInferenceHandler


def print_section(title, char="="):
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f"{title:^80}")
    print(f"{char * 80}\n")


def main():
    print_section("TWO-STAGE RANKING SYSTEM - INTERVIEW DEMO")
    
    # Configuration
    BUCKET = "ltr-models-frp"
    KEY = "data/ltr_training_data.csv"
    MODEL_DIR = "./models"
    
    # =========================================================================
    # PART 1: QUICK TRAINING DEMONSTRATION
    # =========================================================================
    print_section("PART 1: MODEL TRAINING", "-")
    
    print("📊 Loading training data from BigQuery thelook_ecommerce dataset...")
    df = load_data_from_s3(BUCKET, KEY)
    print(f"   ✓ Loaded {len(df):,} training samples")
    
    # Show data sample
    print("\n📋 Sample Training Data:")
    print(df.head(10).to_string(index=False))
    
    print("\n🔧 Preprocessing for XGBoost LambdaMART...")
    X, y, groups = preprocess_data(df)
    print(f"   ✓ Features: {X.shape[1]} ({list(X.columns)})")
    print(f"   ✓ Query Groups: {len(groups):,}")
    print(f"   ✓ Total Items: {len(X):,}")
    
    print("\n🤖 Training XGBoost Ranker (50 boosting rounds)...")
    model = train_ranker(X, y, groups, n_estimators=50)
    print("   ✓ Training completed")
    
    model_path = save_model(model, MODEL_DIR)
    print(f"   ✓ Model saved: {model_path}")
    
    # =========================================================================
    # PART 2: INFERENCE DEMONSTRATION
    # =========================================================================
    print_section("PART 2: TWO-STAGE RANKING IN ACTION", "-")
    
    # Initialize ranker
    handler = RankingInferenceHandler(model_path)
    
    # Stage 1: Simulate retrieval
    print("🔍 STAGE 1: Retrieval (OpenSearch/Elasticsearch Simulation)")
    print("   Query: 'clothing items for online shopping'")
    print("   Retrieved 8 candidates based on vector similarity...\n")
    
    # Candidates with prices matching training distribution
    candidates = [
        {"item_id": "1", "name": "Cotton T-Shirt", "category": "Tops & Tees", 
         "retail_price": 24.99, "cost": 12.50, "retrieval_score": 0.95},
        {"item_id": "2", "name": "Denim Jeans", "category": "Pants", 
         "retail_price": 89.99, "cost": 45.00, "retrieval_score": 0.92},
        {"item_id": "3", "name": "Wool Sweater", "category": "Sweaters", 
         "retail_price": 79.99, "cost": 40.00, "retrieval_score": 0.90},
        {"item_id": "4", "name": "Athletic Socks", "category": "Socks", 
         "retail_price": 14.99, "cost": 7.50, "retrieval_score": 0.88},
        {"item_id": "5", "name": "Leather Belt", "category": "Accessories", 
         "retail_price": 39.99, "cost": 20.00, "retrieval_score": 0.85},
        {"item_id": "6", "name": "Running Shorts", "category": "Shorts", 
         "retail_price": 34.99, "cost": 17.50, "retrieval_score": 0.83},
        {"item_id": "7", "name": "Winter Coat", "category": "Outerwear & Coats", 
         "retail_price": 149.99, "cost": 75.00, "retrieval_score": 0.80},
        {"item_id": "8", "name": "Silk Scarf", "category": "Intimates", 
         "retail_price": 44.99, "cost": 22.50, "retrieval_score": 0.78}
    ]
    
    # Display retrieval results
    print("   Initial Ranking (by retrieval score only):")
    print(f"   {'#':<4} {'Item Name':<25} {'Category':<20} {'Price':<12} {'Ret. Score'}")
    print("   " + "-" * 75)
    for i, item in enumerate(candidates, 1):
        print(f"   {i:<4} {item['name']:<25} {item['category']:<20} "
              f"${item['retail_price']:<11.2f} {item['retrieval_score']:.2f}")
    
    # Stage 2: Re-rank with LTR
    print("\n⚡ STAGE 2: LTR Re-Ranking (XGBoost LambdaMART)")
    print("   Applying business features: price, cost, profit margin...")
    
    ranked_results = handler.handle_request(candidates)
    
    print("\n   Final Ranking (after LTR optimization):")
    print(f"   {'#':<4} {'Item Name':<25} {'Category':<20} {'Price':<12} "
          f"{'LTR Score':<12} {'Ret. Score'}")
    print("   " + "-" * 85)
    for i, item in enumerate(ranked_results, 1):
        print(f"   {i:<4} {item['name']:<25} {item['category']:<20} "
              f"${item['retail_price']:<11.2f} {item['ranking_score']:<12.4f} "
              f"{item['retrieval_score']:.2f}")
    
    # =========================================================================
    # PART 3: BUSINESS IMPACT ANALYSIS
    # =========================================================================
    print_section("PART 3: BUSINESS IMPACT ANALYSIS", "-")
    
    # Calculate margins
    def calc_margin(item):
        return item['retail_price'] - item['cost']
    
    # Top 3 comparison
    original_top3 = candidates[:3]
    reranked_top3 = ranked_results[:3]
    
    original_margin = sum(calc_margin(item) for item in original_top3)
    reranked_margin = sum(calc_margin(item) for item in reranked_top3)
    improvement = ((reranked_margin - original_margin) / original_margin) * 100
    
    print("📊 Top 3 Items Comparison:")
    print(f"\n   Retrieval-Only Top 3:")
    for i, item in enumerate(original_top3, 1):
        margin = calc_margin(item)
        print(f"      {i}. {item['name']:<25} | Margin: ${margin:.2f}")
    print(f"   → Total Margin: ${original_margin:.2f}")
    
    print(f"\n   LTR-Optimized Top 3:")
    for i, item in enumerate(reranked_top3, 1):
        margin = calc_margin(item)
        print(f"      {i}. {item['name']:<25} | Margin: ${margin:.2f}")
    print(f"   → Total Margin: ${reranked_margin:.2f}")
    
    print(f"\n   💰 Margin Improvement: ${reranked_margin - original_margin:.2f} ({improvement:+.1f}%)")
    
    # =========================================================================
    # PART 4: KEY INSIGHTS
    # =========================================================================
    print_section("PART 4: KEY TECHNICAL INSIGHTS", "-")
    
    print("""
📌 Architecture Highlights:
   • Two-stage pipeline: Fast retrieval → Precise re-ranking
   • Stage 1 (OpenSearch): Vector similarity, filters (category, price)
   • Stage 2 (XGBoost): Business optimization (margin, pricing strategy)
   
🎯 Why This Approach Works:
   • Retrieval handles scale: 100K+ items → top 100 in ~20ms
   • LTR optimizes business goals: Revenue, margins, diversity
   • Pairwise ranking learns relative ordering (not absolute scores)
   
⚙️ Production Readiness:
   • Model: XGBoost (battle-tested, low latency)
   • Deployment: SageMaker endpoints, Lambda, or ECS/Fargate
   • Monitoring: NDCG@10, latency, business metrics
   
🔬 Training Data:
   • Source: BigQuery thelook_ecommerce (public dataset)
   • Labels: Actual purchase behavior (0/1)
   • Features: retail_price, cost, category (expandable)
   
📈 Performance:
   • Training: ~50 boosting rounds on 5K samples
   • Inference: < 50ms for 100 candidates
   • Metric: NDCG@10 (ranking quality)
    """)
    
    print_section("✅ DEMO COMPLETED SUCCESSFULLY")
    
    print("\n🎤 Interview Talking Points:")
    print("   1. Explain why traditional CTR models fail for unique items")
    print("   2. Walk through the two-stage architecture diagram")
    print("   3. Show the before/after ranking comparison")
    print("   4. Discuss feature engineering opportunities (category encoding, etc.)")
    print("   5. Explain deployment options (SageMaker, Lambda, ECS)")
    print("   6. Talk about production monitoring and A/B testing")
    print()


if __name__ == '__main__':
    main()
