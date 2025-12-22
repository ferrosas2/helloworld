"""
Multimodal Two-Tower Architecture for Auction Item Embeddings

This module implements a PyTorch model that encodes auction items
using both text (title + description) and image features into a 
shared 128-dimensional embedding space.

Architecture:
    - Text Tower: DistilBERT (pre-trained)
    - Image Tower: ResNet50 (pre-trained, ImageNet weights)
    - Fusion: Concatenate → Linear projection → L2 Normalization

In production, this model would be:
    1. Trained on AWS SageMaker Training Jobs (user-item interaction data)
    2. Deployed for batch inference on SageMaker Processing
    3. Embeddings indexed in Amazon OpenSearch Service (k-NN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer
from torchvision import models


class AuctionTwoTower(nn.Module):
    """
    Two-Tower model for multimodal auction item embeddings.
    
    Inputs:
        - text: Auction title + description (tokenized)
        - image: Product image (preprocessed tensor)
    
    Output:
        - embedding: 128-dim L2-normalized vector
    """
    
    def __init__(self, embedding_dim=128, freeze_backbones=True):
        """
        Initialize the two-tower architecture.
        
        Args:
            embedding_dim (int): Dimensionality of final embedding space
            freeze_backbones (bool): If True, freeze pre-trained weights
                                     (faster training, good for demos)
        """
        super(AuctionTwoTower, self).__init__()
        
        # ===== Text Tower: DistilBERT =====
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_hidden_dim = self.text_encoder.config.hidden_size  # 768
        
        if freeze_backbones:
            # Freeze DistilBERT weights (use only as feature extractor)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # ===== Image Tower: ResNet50 =====
        resnet = models.resnet50(pretrained=True)
        # Remove final classification layer
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_hidden_dim = 2048  # ResNet50 output dimension
        
        if freeze_backbones:
            # Freeze ResNet weights
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # ===== Fusion Layer =====
        # Concatenate text (768) + image (2048) = 2816 dims
        fusion_input_dim = self.text_hidden_dim + self.image_hidden_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def encode_text(self, input_ids, attention_mask):
        """
        Encode text using DistilBERT.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
        
        Returns:
            text_features: (batch_size, 768) tensor
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding (first token)
        text_features = outputs.last_hidden_state[:, 0, :]
        return text_features
    
    def encode_image(self, image_tensor):
        """
        Encode image using ResNet50.
        
        Args:
            image_tensor: (batch_size, 3, 224, 224) preprocessed image
        
        Returns:
            image_features: (batch_size, 2048) tensor
        """
        image_features = self.image_encoder(image_tensor)
        # Flatten spatial dimensions
        image_features = image_features.view(image_features.size(0), -1)
        return image_features
    
    def forward(self, input_ids, attention_mask, image_tensor):
        """
        Forward pass: Text + Image → 128-dim embedding.
        
        Args:
            input_ids: (batch_size, seq_len) tokenized text
            attention_mask: (batch_size, seq_len) attention mask
            image_tensor: (batch_size, 3, 224, 224) image
        
        Returns:
            embedding: (batch_size, 128) L2-normalized vector
        """
        # Encode text and image separately
        text_features = self.encode_text(input_ids, attention_mask)
        image_features = self.encode_image(image_tensor)
        
        # Concatenate features
        fused_features = torch.cat([text_features, image_features], dim=1)
        
        # Project to embedding space
        embedding = self.fusion_layer(fused_features)
        
        # L2 normalization (for cosine similarity)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class AuctionQueryEncoder(nn.Module):
    """
    Simplified encoder for user queries (text-only).
    
    In production:
        - This runs in real-time on AWS Lambda/SageMaker Endpoint
        - Query embedding is used to search OpenSearch k-NN index
    """
    
    def __init__(self, embedding_dim=128):
        super(AuctionQueryEncoder, self).__init__()
        
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_hidden_dim = self.text_encoder.config.hidden_size
        
        # Project text embedding to same space as item embeddings
        self.projection = nn.Sequential(
            nn.Linear(self.text_hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, input_ids, attention_mask):
        """
        Encode user query to 128-dim embedding.
        
        Args:
            input_ids: (batch_size, seq_len) tokenized query
            attention_mask: (batch_size, seq_len) attention mask
        
        Returns:
            embedding: (batch_size, 128) L2-normalized vector
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0, :]
        
        embedding = self.projection(text_features)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


def test_model():
    """
    Quick test to verify model runs correctly.
    """
    print("="*60)
    print("Testing AuctionTwoTower Model")
    print("="*60)
    
    # Initialize model
    model = AuctionTwoTower(embedding_dim=128, freeze_backbones=True)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Mock input data
    batch_size = 2
    sample_text = [
        "Vintage Rolex Watch - Rare 1960s Model",
        "Sony PlayStation 5 Console - Brand New"
    ]
    
    # Tokenize text
    encoded = tokenizer(
        sample_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Mock image input (random tensor for demo)
    mock_image = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        embeddings = model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            image_tensor=mock_image
        )
    
    print(f"\n✅ Model output shape: {embeddings.shape}")
    print(f"Expected: torch.Size([{batch_size}, 128])")
    
    # Verify L2 normalization
    norms = torch.norm(embeddings, p=2, dim=1)
    print(f"\n✅ Embedding L2 norms: {norms.tolist()}")
    print(f"Expected: ~[1.0, 1.0] (normalized)")
    
    # Test query encoder
    print("\n" + "="*60)
    print("Testing AuctionQueryEncoder")
    print("="*60)
    
    query_encoder = AuctionQueryEncoder(embedding_dim=128)
    query_encoder.eval()
    
    query_text = ["vintage watches"]
    encoded_query = tokenizer(
        query_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        query_embedding = query_encoder(
            input_ids=encoded_query['input_ids'],
            attention_mask=encoded_query['attention_mask']
        )
    
    print(f"\n✅ Query embedding shape: {query_embedding.shape}")
    
    # Calculate cosine similarity
    similarity = torch.matmul(query_embedding, embeddings.T)
    print(f"\n✅ Cosine similarity with items: {similarity[0].tolist()}")
    print(f"Higher score = more relevant item")
    
    print("\n" + "="*60)
    print("✅ Model test passed!")
    print("="*60)


if __name__ == "__main__":
    test_model()
