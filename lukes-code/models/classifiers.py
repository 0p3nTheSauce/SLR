import torch
import torch.nn as nn

class AttentionClassifier(nn.Module):
    def __init__(self, in_features, out_features, drop_p, 
                 in_linear=1, n_attention=5):
        super(AttentionClassifier, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Input projection layer (optional)
        if in_linear != 1:
            self.input_proj = nn.Linear(in_features, in_features * in_linear)
            self.feature_dim = in_features * in_linear
        else:
            self.input_proj = None
            self.feature_dim = in_features
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=8,  # You can make this configurable too
                dropout=drop_p,
                batch_first=True  # More convenient for most use cases
            ) for _ in range(n_attention)
        ])
        
        # Layer normalization for each attention layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.feature_dim) for _ in range(n_attention)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(drop_p)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(self.feature_dim // 2, out_features)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, in_features)
        
        # Optional input projection
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Apply attention layers with residual connections
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention (query, key, value all come from x)
            attn_out, _ = attention(x, x, x)
            
            # Residual connection + layer norm
            x = layer_norm(x + self.dropout(attn_out))
        
        # Global average pooling or use [CLS] token approach
        # Here using mean pooling across sequence dimension
        x = x.mean(dim=1)  # (batch_size, feature_dim)
        
        # Classification
        out = self.classifier(x)
        
        return out

# Example usage:
if __name__ == "__main__":
    # Example with feature extractor output
    batch_size, seq_len, feature_dim = 32, 100, 256
    num_classes = 10
    
    # Create model
    model = AttentionClassifier(
        in_features=feature_dim,
        out_features=num_classes,
        drop_p=0.1,
        in_linear=2,  # Project to 512 dim
        n_attention=3
    )
    
    # Dummy input from your feature extractor
    features = torch.randn(batch_size, seq_len, feature_dim)
    
    # Forward pass
    logits = model(features)
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {logits.shape}")  # (32, 10)