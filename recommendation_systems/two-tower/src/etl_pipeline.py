"""
ETL Pipeline for Offline Item Embedding Generation

This script simulates an AWS SageMaker Processing Job that:
    1. Loads auction item data (in production: from S3/DynamoDB)
    2. Generates multimodal embeddings using the Two-Tower model
    3. Saves embeddings in OpenSearch-ready JSON format
    4. (In production) Pushes to Amazon OpenSearch Service k-NN index

In production, this runs as a scheduled batch job (e.g., nightly):
    - Input: New auction items from DynamoDB/S3
    - Processing: SageMaker Processing Job (ml.p3.2xlarge for GPU)
    - Output: Embeddings → S3 → OpenSearch bulk indexing
    
Demo: Uses synthetic dummy data to show end-to-end flow.
"""

import json
import torch
import numpy as np
from transformers import DistilBertTokenizer
from model import AuctionTwoTower


# ===== Synthetic Auction Data Generator =====
def generate_dummy_auction_items(num_items=10):
    """
    Generate synthetic auction items for demo purposes.
    
    In production, this data comes from:
        - DynamoDB: Real-time auction catalog
        - S3: Historical auction data
        - RDS: Product information database
    
    Returns:
        List of auction item dictionaries
    """
    categories = ['Watches', 'Electronics', 'Collectibles', 'Jewelry', 'Art']
    conditions = ['New', 'Like New', 'Good', 'Fair']
    
    items = []
    for i in range(num_items):
        item = {
            'item_id': f'ITEM-{1000 + i}',
            'title': f'{np.random.choice(["Vintage", "Rare", "Premium", "Limited Edition"])} '
                     f'{np.random.choice(["Rolex", "Omega", "PlayStation", "iPhone", "Artwork"])} '
                     f'- {np.random.choice(conditions)}',
            'description': f'Authentic {np.random.choice(categories).lower()} item in excellent condition. '
                          f'Perfect for collectors. Ships worldwide with tracking.',
            'category': np.random.choice(categories),
            'current_price': round(np.random.uniform(50, 5000), 2),
            'bid_count': np.random.randint(1, 100),
            'image_url': f'https://auctions.example.com/images/{1000 + i}.jpg',
            'seller_rating': round(np.random.uniform(3.5, 5.0), 1),
            'time_remaining': f'{np.random.randint(1, 72)} hours'
        }
        items.append(item)
    
    return items


# ===== Embedding Generation =====
def generate_item_embeddings(items, model, tokenizer, device='cpu'):
    """
    Generate embeddings for a batch of auction items.
    
    Args:
        items: List of item dictionaries
        model: AuctionTwoTower model instance
        tokenizer: DistilBERT tokenizer
        device: 'cpu' or 'cuda'
    
    Returns:
        List of dictionaries with item_id, embedding, and metadata
    """
    model.eval()
    model.to(device)
    
    embeddings_data = []
    
    # Process in batches for efficiency
    batch_size = 4
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        
        # Prepare text inputs
        texts = [f"{item['title']}. {item['description']}" for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        ).to(device)
        
        # Mock image inputs (in production: load real images from S3)
        # For demo: random tensors simulate preprocessed images
        batch_images = torch.randn(len(batch), 3, 224, 224).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                image_tensor=batch_images
            )
        
        # Convert to numpy and store with metadata
        embeddings_np = embeddings.cpu().numpy()
        
        for j, item in enumerate(batch):
            embedding_entry = {
                'item_id': item['item_id'],
                'embedding': embeddings_np[j].tolist(),  # 128-dim vector
                'metadata': {
                    'title': item['title'],
                    'category': item['category'],
                    'current_price': item['current_price'],
                    'bid_count': item['bid_count'],
                    'seller_rating': item['seller_rating'],
                    'image_url': item['image_url']
                }
            }
            embeddings_data.append(embedding_entry)
    
    return embeddings_data


# ===== OpenSearch Format Export =====
def save_to_opensearch_format(embeddings_data, output_file='item_embeddings.json'):
    """
    Save embeddings in OpenSearch bulk indexing format.
    
    In production:
        - Uploaded to S3: s3://auction-embeddings/daily/2025-12-21.json
        - Ingested via OpenSearch bulk API with k-NN index
    
    OpenSearch Index Mapping:
    {
        "mappings": {
            "properties": {
                "item_id": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 128,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib"
                    }
                },
                "metadata": {"type": "object"}
            }
        }
    }
    """
    with open(output_file, 'w') as f:
        for item in embeddings_data:
            # OpenSearch bulk format: index action + document
            index_action = {
                "index": {
                    "_index": "auction-items",
                    "_id": item['item_id']
                }
            }
            
            document = {
                "item_id": item['item_id'],
                "embedding": item['embedding'],
                "metadata": item['metadata']
            }
            
            # Write as newline-delimited JSON (NDJSON)
            f.write(json.dumps(index_action) + '\n')
            f.write(json.dumps(document) + '\n')
    
    print(f"✅ Saved {len(embeddings_data)} embeddings to {output_file}")
    print(f"   Format: OpenSearch bulk indexing (NDJSON)")


# ===== Main ETL Pipeline =====
def run_etl_pipeline(num_items=10, output_file='item_embeddings.json'):
    """
    Complete ETL pipeline for item embedding generation.
    
    Steps:
        1. Generate/Load auction items
        2. Initialize Two-Tower model
        3. Generate embeddings
        4. Save in OpenSearch format
    
    In production (AWS SageMaker Processing):
        - Input: s3://auction-data/items/{date}.parquet
        - Processing: This script on ml.p3.2xlarge (GPU)
        - Output: s3://auction-embeddings/{date}.json
    """
    print("="*70)
    print("AWS SageMaker Processing Job: Offline Item Embedding Pipeline")
    print("="*70)
    
    # Step 1: Generate dummy items
    print(f"\n📥 Step 1: Loading {num_items} auction items...")
    items = generate_dummy_auction_items(num_items)
    print(f"   Sample item: {items[0]['title']}")
    
    # Step 2: Initialize model
    print(f"\n🧠 Step 2: Initializing Two-Tower model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    model = AuctionTwoTower(embedding_dim=128, freeze_backbones=True)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # In production: Load fine-tuned model from S3
    # model_path = 's3://auction-models/two-tower/v1.2/model.pth'
    # model.load_state_dict(torch.load(model_path))
    
    # Step 3: Generate embeddings
    print(f"\n🔄 Step 3: Generating embeddings...")
    embeddings_data = generate_item_embeddings(items, model, tokenizer, device)
    print(f"   Generated {len(embeddings_data)} embeddings (128-dim each)")
    
    # Step 4: Save to OpenSearch format
    print(f"\n💾 Step 4: Saving to OpenSearch format...")
    save_to_opensearch_format(embeddings_data, output_file)
    
    # Step 5: Display sample embedding
    print(f"\n📊 Sample Embedding:")
    sample = embeddings_data[0]
    print(f"   Item: {sample['metadata']['title']}")
    print(f"   Embedding (first 10 dims): {sample['embedding'][:10]}")
    print(f"   L2 norm: {np.linalg.norm(sample['embedding']):.4f} (should be ~1.0)")
    
    print("\n" + "="*70)
    print("✅ ETL Pipeline Complete!")
    print("="*70)
    print(f"\n📍 Next Steps (Production):")
    print(f"   1. Upload to S3: aws s3 cp {output_file} s3://auction-embeddings/")
    print(f"   2. Bulk index to OpenSearch:")
    print(f"      curl -X POST 'https://opensearch:9200/_bulk' \\")
    print(f"           -H 'Content-Type: application/x-ndjson' \\")
    print(f"           --data-binary @{output_file}")
    
    return embeddings_data


# ===== CLI Entry Point =====
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate auction item embeddings')
    parser.add_argument('--num-items', type=int, default=10,
                       help='Number of items to generate (default: 10)')
    parser.add_argument('--output', type=str, default='item_embeddings.json',
                       help='Output file path (default: item_embeddings.json)')
    
    args = parser.parse_args()
    
    # Run the ETL pipeline
    embeddings = run_etl_pipeline(
        num_items=args.num_items,
        output_file=args.output
    )
    
    print(f"\n💡 Tip: Run 'python src/inference.py' to test similarity search!")
