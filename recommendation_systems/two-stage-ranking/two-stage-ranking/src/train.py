"""
Training Module for XGBoost Learning-to-Rank (LambdaMART)

This script trains a ranking model for e-commerce product recommendations
using XGBoost's pairwise ranking objective. Designed for deployment on
AWS SageMaker or as a standalone training job.

Author: Fernando Rosas
Date: December 2025
"""

import argparse
import logging
import os
from typing import Tuple

import boto3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_from_s3(bucket_name: str, key: str) -> pd.DataFrame:
    """
    Load training data from S3 bucket.
    
    Args:
        bucket_name: Name of the S3 bucket
        key: S3 object key (path to the CSV file)
    
    Returns:
        DataFrame containing the training data
    
    Raises:
        Exception: If S3 download or CSV parsing fails
    """
    logger.info(f"Loading data from s3://{bucket_name}/{key}")
    
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        df = pd.read_csv(obj['Body'])
        
        logger.info(f"Successfully loaded {len(df)} rows")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to load data from S3: {str(e)}")
        raise


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Preprocess data for XGBoost Ranker training.
    
    The ranker requires:
    - Data sorted by query_group_id
    - Feature matrix (X)
    - Target labels (y)
    - Group sizes array for ranking
    
    Args:
        df: Raw DataFrame from S3
    
    Returns:
        Tuple of (X, y, groups)
        - X: Feature matrix
        - y: Target labels (relevance scores)
        - groups: Array of group sizes for ranking
    """
    logger.info("Preprocessing data for ranker...")
    
    # Sort by query group (required for ranker)
    df = df.sort_values(by='query_group_id')
    
    # Log unique categories for transparency
    logger.info(f"Unique categories: {df['category'].unique().tolist()}")
    
    # Feature Engineering
    # In production: Use One-Hot Encoding for 'category', TF-IDF for text, etc.
    # Here we use numeric features as baseline
    feature_columns = ['retail_price', 'cost']
    X = df[feature_columns]
    y = df['label']
    
    # Create groups array
    # Each element represents the number of items in a query group
    groups = df.groupby('query_group_id').size().to_numpy()
    
    logger.info(f"Features: {feature_columns}")
    logger.info(f"Number of query groups: {len(groups)}")
    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Label distribution:\n{y.value_counts()}")
    
    return X, y, groups


def train_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    gamma: float = 1.0,
    min_child_weight: float = 0.1
) -> xgb.XGBRanker:
    """
    Train XGBoost LambdaMART ranker model.
    
    Uses pairwise ranking objective to learn relative ordering of items
    within each query group.
    
    Args:
        X: Feature matrix
        y: Target labels (relevance scores)
        groups: Group sizes for ranking
        n_estimators: Number of boosting rounds
        learning_rate: Step size shrinkage
        gamma: Minimum loss reduction for split
        min_child_weight: Minimum sum of instance weight in child
    
    Returns:
        Trained XGBRanker model
    """
    logger.info("Initializing XGBoost Ranker with LambdaMART...")
    
    model = xgb.XGBRanker(
        objective='rank:pairwise',      # LambdaMART pairwise ranking
        learning_rate=learning_rate,
        gamma=gamma,
        min_child_weight=min_child_weight,
        n_estimators=n_estimators,
        eval_metric='ndcg',             # Normalized Discounted Cumulative Gain
        random_state=42
    )
    
    logger.info("Training model...")
    logger.info(f"Hyperparameters: n_estimators={n_estimators}, "
                f"learning_rate={learning_rate}, gamma={gamma}")
    
    model.fit(
        X,
        y,
        group=groups,
        verbose=True
    )
    
    logger.info("Training completed successfully!")
    
    return model


def save_model(model: xgb.XGBRanker, output_dir: str = '/opt/ml/model') -> str:
    """
    Save trained model in JSON format (SageMaker standard).
    
    Args:
        model: Trained XGBRanker model
        output_dir: Directory to save model (default: SageMaker model dir)
    
    Returns:
        Path to saved model file
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.json')
    
    logger.info(f"Saving model to {model_path}")
    model.save_model(model_path)
    
    logger.info("Model saved successfully!")
    return model_path


def main():
    """
    Main training pipeline.
    
    Orchestrates data loading, preprocessing, training, and model saving.
    """
    parser = argparse.ArgumentParser(
        description='Train XGBoost Learning-to-Rank model for product recommendations'
    )
    
    parser.add_argument(
        '--bucket',
        type=str,
        required=True,
        help='S3 bucket name containing training data'
    )
    
    parser.add_argument(
        '--key',
        type=str,
        required=True,
        help='S3 object key (path to CSV file)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/opt/ml/model',
        help='Directory to save trained model (default: /opt/ml/model)'
    )
    
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of boosting rounds (default: 100)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate (default: 0.1)'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.0,
        help='Minimum loss reduction for split (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Starting Training Pipeline: Two-Stage Ranking System")
    logger.info("=" * 80)
    
    # Step 1: Load data from S3
    df = load_data_from_s3(args.bucket, args.key)
    
    # Step 2: Preprocess data
    X, y, groups = preprocess_data(df)
    
    # Step 3: Train ranker
    model = train_ranker(
        X, y, groups,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )
    
    # Step 4: Save model
    model_path = save_model(model, args.output_dir)
    
    logger.info("=" * 80)
    logger.info(f"Training Pipeline Complete! Model saved at: {model_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
