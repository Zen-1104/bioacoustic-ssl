import torch
import torch.nn as nn

def evaluate_accuracy(model, backbone, data_loader, device = "cpu"):
    model.eval()
    backbone.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            feats = backbone(x)
            logits = model(feats)
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim = 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    if total == 0:
        raise RuntimeError("Linear probe evaluation processed zero examples.")
    
    avg_loss = val_loss / total
    accuracy = correct / total

    if not torch.isfinite(torch.tensor([avg_loss, accuracy])).all():
        raise FloatingPointError("Linear probe evaluation produced non finite metrics.")
    
    return avg_loss, accuracy

if __name__ == "__main__":
    class MockBackbone(nn.Module):
        def forward(self, x):
            return torch.flatten(x, 1)[:, :512]

    class MockHead(nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 50)

    backbone = MockBackbone()
    head = MockHead()
    dummy_x = torch.randn(8, 1, 64, 313)
    dummy_y = torch.randint(0, 50, (8,))
    dataset = torch.utils.data.TensorDataset(dummy_x, dummy_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 4)

    loss, acc = evaluate_accuracy(head, backbone, loader)
    print(f"Validation Loss: {loss:.4f}, Accuracy: {acc * 100:.2f}%")