import torch
import torch.nn as nn

class Linear_Head(nn.Module):
    def __init__(self, in_features = 512, num_classes = 50):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    head = Linear_Head(in_features = 512, num_classes = 50)
    dummy_embedding = torch.randn(16, 512)
    logits = head(dummy_embedding)
    
    print(f"Embedding shape: {dummy_embedding.shape}")
    print(f"Logits shape: {logits.shape}")