"""
Inference Handler for XGBoost Learning-to-Rank Model

This module provides a SageMaker-compatible inference handler for the
XGBoost ranker model. It can be deployed as a real-time endpoint or
used for batch inference.

Author: MLOps Engineer
Date: December 2025
"""

import json
import logging
import os
from typing import List, Dict, Any

import xgboost as xgb
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RankingInferenceHandler:
    """
    Inference handler for XGBoost ranker model.
    
    Simulates a SageMaker endpoint that loads the model and performs
    real-time re-ranking of candidate items.
    """
    
    def __init__(self, model_path: str = '/opt/ml/model/model.json'):
        """
        Initialize the inference handler.
        
        Args:
            model_path: Path to the saved XGBoost model JSON file
        """
        self.model_path = model_path
        self.model = None
        self.feature_columns = ['retail_price', 'cost']  # Must match training
        
        logger.info(f"Initializing RankingInferenceHandler with model: {model_path}")
        self._load_model()
    
    def _load_model(self):
        """Load the XGBoost model from disk."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = xgb.XGBRanker()
            self.model.load_model(self.model_path)
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model type: {type(self.model)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def preprocess_input(self, payload: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Preprocess input payload into feature matrix.
        
        Args:
            payload: List of candidate items with features
                     Example: [
                         {"item_id": "123", "retail_price": 99.99, "cost": 50.0},
                         {"item_id": "456", "retail_price": 149.99, "cost": 80.0}
                     ]
        
        Returns:
            Feature DataFrame ready for prediction
        """
        logger.info(f"Preprocessing {len(payload)} candidate items...")
        
        # Convert to DataFrame
        df = pd.DataFrame(payload)
        
        # Validate required features
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract features in correct order
        X = df[self.feature_columns]
        
        logger.info(f"Extracted features: {self.feature_columns}")
        logger.info(f"Input shape: {X.shape}")
        
        return df, X
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ranking scores for candidate items.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of ranking scores (higher = more relevant)
        """
        logger.info("Generating ranking scores...")
        
        scores = self.model.predict(X)
        
        logger.info(f"Scores shape: {scores.shape}")
        logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        return scores
    
    def postprocess_output(
        self,
        df: pd.DataFrame,
        scores: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Sort items by ranking score and prepare output.
        
        Args:
            df: Original DataFrame with all item information
            scores: Predicted ranking scores
        
        Returns:
            List of items sorted by relevance (descending)
        """
        logger.info("Sorting items by ranking score...")
        
        # Add scores to DataFrame
        df['ranking_score'] = scores
        
        # Sort by score (descending)
        df_sorted = df.sort_values(by='ranking_score', ascending=False)
        
        # Convert to list of dicts
        results = df_sorted.to_dict('records')
        
        logger.info(f"Returned {len(results)} ranked items")
        
        return results
    
    def handle_request(self, payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main inference pipeline.
        
        Args:
            payload: List of candidate items to re-rank
        
        Returns:
            List of items sorted by predicted relevance
        """
        logger.info("=" * 80)
        logger.info("Processing inference request...")
        logger.info("=" * 80)
        
        # Preprocess
        df, X = self.preprocess_input(payload)
        
        # Predict
        scores = self.predict(X)
        
        # Postprocess
        results = self.postprocess_output(df, scores)
        
        logger.info("Inference completed successfully!")
        logger.info("=" * 80)
        
        return results


# SageMaker Inference Functions
# These functions follow the SageMaker inference container contract

def model_fn(model_dir: str):
    """
    Load model for SageMaker inference.
    
    This function is called once when the inference container starts.
    
    Args:
        model_dir: Path to the model artifacts directory
    
    Returns:
        Loaded model handler
    """
    model_path = os.path.join(model_dir, 'model.json')
    return RankingInferenceHandler(model_path)


def input_fn(request_body: str, content_type: str = 'application/json'):
    """
    Deserialize input data for inference.
    
    Args:
        request_body: Raw request body
        content_type: MIME type of the request
    
    Returns:
        Parsed input data
    """
    if content_type == 'application/json':
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: List[Dict[str, Any]], model: RankingInferenceHandler):
    """
    Perform prediction using the loaded model.
    
    Args:
        input_data: Preprocessed input data
        model: Loaded model handler
    
    Returns:
        Prediction results
    """
    return model.handle_request(input_data)


def output_fn(prediction: List[Dict[str, Any]], accept: str = 'application/json'):
    """
    Serialize prediction output.
    
    Args:
        prediction: Model prediction results
        accept: Expected response MIME type
    
    Returns:
        Serialized response
    """
    if accept == 'application/json':
        return json.dumps(prediction, indent=2)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# Standalone Testing
def main():
    """
    Test the inference handler locally.
    
    Simulates a real inference request with sample data.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test XGBoost ranker inference locally'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='model.json',
        help='Path to trained model file'
    )
    
    args = parser.parse_args()
    
    # Initialize handler
    handler = RankingInferenceHandler(args.model_path)
    
    # Sample inference payload (simulating candidate items from retrieval stage)
    sample_payload = [
        {
            "item_id": "ITEM_001",
            "name": "Vintage Watch",
            "category": "Watches",
            "retail_price": 299.99,
            "cost": 150.0
        },
        {
            "item_id": "ITEM_002",
            "name": "Designer Handbag",
            "category": "Bags",
            "retail_price": 599.99,
            "cost": 300.0
        },
        {
            "item_id": "ITEM_003",
            "name": "Leather Wallet",
            "category": "Accessories",
            "retail_price": 79.99,
            "cost": 40.0
        },
        {
            "item_id": "ITEM_004",
            "name": "Sunglasses",
            "category": "Accessories",
            "retail_price": 199.99,
            "cost": 100.0
        },
        {
            "item_id": "ITEM_005",
            "name": "Running Shoes",
            "category": "Footwear",
            "retail_price": 129.99,
            "cost": 65.0
        }
    ]
    
    print("\n" + "=" * 80)
    print("SAMPLE INFERENCE REQUEST")
    print("=" * 80)
    print(json.dumps(sample_payload, indent=2))
    
    # Run inference
    results = handler.handle_request(sample_payload)
    
    print("\n" + "=" * 80)
    print("RANKED RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    
    print("\n" + "=" * 80)
    print("RANKING SUMMARY")
    print("=" * 80)
    for i, item in enumerate(results, 1):
        print(f"{i}. {item['name']:25s} | Score: {item['ranking_score']:.4f} | "
              f"Price: ${item['retail_price']:.2f}")


if __name__ == '__main__':
    main()
