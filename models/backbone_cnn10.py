import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.pool(x)

class CNN_10_Backbone(nn.Module):
    def __init__(self, in_channels = 1, num_features = 512):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.block4 = ConvBlock(256, num_features)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x.mean(dim = (-2, -1))

if __name__ == "__main__":
    model = CNN_10_Backbone()
    dummy_input = torch.randn(4, 1, 64, 512)
    output = model(dummy_input)

    print(f"Full Forward Output Shape: {output.shape}")