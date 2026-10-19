import torch
import torch.nn as nn

class Projection_Head(nn.Module):
    def __init__(self, input_dim = 512, hidden_dim = 512, out_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__" :
    head = Projection_Head()
    dummy_embedding = torch.randn(4, 512)
    projected = head(dummy_embedding)

    print(f"Input embedding shape: {dummy_embedding.shape}")
    print(f"Projected ouput shape: {projected.shape}")