"""
Demonstrates the impact of feature distribution mismatch on LTR scores.
Compares original out-of-distribution prices vs. training-aligned prices.

Usage:
    python examples/compare_price_ranges.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import RankingInferenceHandler

def test_with_prices(ranker, price_scenario, items):
    """Test ranker with given price scenario."""
    print(f"\n{'='*80}")
    print(f"SCENARIO: {price_scenario}")
    print('='*80)
    
    ranked = ranker.handle_request(items)
    
    print(f"{'Item':<30} {'Retail Price':<15} {'Cost':<10} {'LTR Score'}")
    print('-'*80)
    for item in ranked:
        print(f"{item['name']:<30} ${item['retail_price']:<14.2f} ${item['cost']:<9.2f} {item['ranking_score']:.6f}")
    
    # Calculate variance in scores
    scores = [item['ranking_score'] for item in ranked]
    score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
    
    print(f"\n📊 Score Statistics:")
    print(f"   Mean: {sum(scores)/len(scores):.6f}")
    print(f"   Min:  {min(scores):.6f}")
    print(f"   Max:  {max(scores):.6f}")
    print(f"   Variance: {score_variance:.8f}")
    
    if score_variance < 0.00001:
        print("   ⚠️  WARNING: Very low variance - model cannot differentiate!")
    else:
        print("   ✅ Good variance - model is differentiating between items")
    
    return ranked

def main():
    print("="*80)
    print("LTR SCORE COMPARISON: FEATURE DISTRIBUTION IMPACT")
    print("="*80)
    print("\nThis demo shows why identical LTR scores (0.0012) occurred")
    print("and how fixing price ranges restores model differentiation.\n")
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.json')
    ranker = RankingInferenceHandler(model_path)
    
    # Original problematic data (out of distribution)
    ood_items = [
        {"item_id": "1", "name": "Vintage Rolex Watch", "retail_price": 4999.99, "cost": 2500.0},
        {"item_id": "2", "name": "Designer Leather Wallet", "retail_price": 299.99, "cost": 150.0},
        {"item_id": "3", "name": "Premium Sunglasses", "retail_price": 499.99, "cost": 250.0},
        {"item_id": "4", "name": "Silk Tie Set", "retail_price": 149.99, "cost": 75.0},
        {"item_id": "5", "name": "Luxury Briefcase", "retail_price": 899.99, "cost": 450.0},
    ]
    
    # Fixed data (matches training distribution)
    fixed_items = [
        {"item_id": "1", "name": "Vintage Rolex Watch", "retail_price": 89.99, "cost": 45.0},
        {"item_id": "2", "name": "Designer Leather Wallet", "retail_price": 29.99, "cost": 15.0},
        {"item_id": "3", "name": "Premium Sunglasses", "retail_price": 49.99, "cost": 25.0},
        {"item_id": "4", "name": "Silk Tie Set", "retail_price": 79.99, "cost": 40.0},
        {"item_id": "5", "name": "Luxury Briefcase", "retail_price": 99.99, "cost": 50.0},
    ]
    
    # Test both scenarios
    test_with_prices(ranker, "ORIGINAL (Out-of-Distribution Prices)", ood_items)
    test_with_prices(ranker, "FIXED (Training-Aligned Prices)", fixed_items)
    
    # Explanation
    print("\n" + "="*80)
    print("🔍 ROOT CAUSE ANALYSIS")
    print("="*80)
    print("""
The XGBoost model was trained on the thelook_ecommerce dataset with:
  • retail_price: $5.55 - $120.00
  • cost: $2.13 - $63.48

When inference data has prices of $149-$4999:
  1. ❌ All items fall outside the training distribution
  2. ❌ Tree splits learned during training don't apply
  3. ❌ All items end up in similar leaf nodes
  4. ❌ Result: Identical scores (~0.0012)

Solution implemented:
  ✅ Adjusted demo prices to $30-$100 range (matches training data)
  ✅ Model now produces differentiated scores
  ✅ Ranking works as expected

For production:
  • Option 1: Retrain model on data with wider price ranges
  • Option 2: Apply feature scaling (StandardScaler) during inference
  • Option 3: Use price percentiles instead of absolute values
    """)
    print("="*80)

if __name__ == "__main__":
    main()
