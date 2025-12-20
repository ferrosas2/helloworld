"""
End-to-End Demo: Two-Stage Ranking Pipeline

This script demonstrates the complete workflow:
1. Train the XGBoost ranker
2. Save the model
3. Load the model for inference
4. Rank candidate items

Usage:
    python examples/demo_pipeline.py
"""

import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import load_data_from_s3, preprocess_data, train_ranker, save_model
from inference import RankingInferenceHandler


def main():
    print("=" * 80)
    print("TWO-STAGE RANKING SYSTEM - END-TO-END DEMO")
    print("=" * 80)
    
    # Configuration
    BUCKET = "ltr-models-frp"
    KEY = "data/ltr_training_data.csv"
    MODEL_DIR = "./models"
    
    print("\n[STEP 1] Training Phase")
    print("-" * 80)
    
    # Load data
    print("Loading training data from S3...")
    df = load_data_from_s3(BUCKET, KEY)
    print(f"✓ Loaded {len(df)} training samples")
    
    # Preprocess
    print("\nPreprocessing data...")
    X, y, groups = preprocess_data(df)
    print(f"✓ Prepared {len(X)} features across {len(groups)} query groups")
    
    # Train
    print("\nTraining XGBoost ranker...")
    model = train_ranker(X, y, groups, n_estimators=50)
    print("✓ Model training completed")
    
    # Save
    print("\nSaving model...")
    model_path = save_model(model, MODEL_DIR)
    print(f"✓ Model saved to {model_path}")
    
    print("\n" + "=" * 80)
    print("[STEP 2] Inference Phase")
    print("-" * 80)
    
    # Initialize inference handler
    print("Loading model for inference...")
    handler = RankingInferenceHandler(model_path)
    print("✓ Inference handler initialized")
    
    # Simulate Stage 1: Retrieval (OpenSearch)
    print("\n[Simulating Stage 1: OpenSearch Retrieval]")
    print("Query: 'luxury accessories for men'")
    print("Retrieved 5 candidate items from vector search...")
    
    candidate_items = [
        {
            "item_id": "ITEM_001",
            "name": "Vintage Rolex Watch",
            "category": "Watches",
            "retail_price": 4999.99,
            "cost": 2500.0,
            "retrieval_score": 0.92  # From OpenSearch
        },
        {
            "item_id": "ITEM_002",
            "name": "Designer Leather Wallet",
            "category": "Accessories",
            "retail_price": 299.99,
            "cost": 150.0,
            "retrieval_score": 0.89
        },
        {
            "item_id": "ITEM_003",
            "name": "Premium Sunglasses",
            "category": "Accessories",
            "retail_price": 499.99,
            "cost": 250.0,
            "retrieval_score": 0.87
        },
        {
            "item_id": "ITEM_004",
            "name": "Silk Tie Set",
            "category": "Accessories",
            "retail_price": 149.99,
            "cost": 75.0,
            "retrieval_score": 0.85
        },
        {
            "item_id": "ITEM_005",
            "name": "Luxury Briefcase",
            "category": "Bags",
            "retail_price": 899.99,
            "cost": 450.0,
            "retrieval_score": 0.83
        }
    ]
    
    print("\n[Stage 2: XGBoost Re-Ranking]")
    print("Applying Learning-to-Rank model...")
    
    # Run inference
    ranked_results = handler.handle_request(candidate_items)
    
    print("\n" + "=" * 80)
    print("FINAL RANKED RESULTS")
    print("=" * 80)
    
    print(f"\n{'Rank':<6} {'Item Name':<30} {'Category':<15} {'Price':<10} {'LTR Score':<12} {'Retrieval Score'}")
    print("-" * 90)
    
    for i, item in enumerate(ranked_results, 1):
        print(f"{i:<6} {item['name']:<30} {item['category']:<15} "
              f"${item['retail_price']:<9.2f} {item['ranking_score']:<12.4f} {item['retrieval_score']:.2f}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    print("""
✓ Stage 1 (OpenSearch) retrieved 5 candidates based on semantic similarity
✓ Stage 2 (XGBoost) re-ranked them using business features:
  - Profit margin: (retail_price - cost) / retail_price
  - Price point optimization
  - Category preferences
  
✓ Final ranking balances:
  - User relevance (from retrieval)
  - Business objectives (from LTR model)
  - Diversity (implicit in training data)
    """)
    
    # Calculate business metrics
    print("\n[BUSINESS IMPACT]")
    top_3_margin = sum([(item['retail_price'] - item['cost']) 
                        for item in ranked_results[:3]])
    print(f"✓ Top 3 items total margin: ${top_3_margin:.2f}")
    print(f"✓ Average item price: ${sum([item['retail_price'] for item in ranked_results]) / len(ranked_results):.2f}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
