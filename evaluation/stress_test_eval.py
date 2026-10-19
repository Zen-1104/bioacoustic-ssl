"""Stress test evaluation with per recording predictions for paired analysis."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def evaluate_at_snr(backbone, linear_head, dataloader, device):
    backbone.eval()
    linear_head.eval()
    criterion = nn.CrossEntropyLoss(reduction = "sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    predictions = []
    fallback_position = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                waveforms, labels, filenames = batch
            else:
                waveforms, labels = batch
                filenames = [f"sample_{fallback_position + i}" for i in range(len(labels))]
                
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            logits = linear_head(backbone(waveforms))

            if not torch.isfinite(logits).all():
                raise FloatingPointError("Model produced NaN or infinite logits during stress test evaluation.")
            
            pred = logits.argmax(dim = 1)
            loss_sum += float(criterion(logits, labels).item())
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))

            for filename, actual, guessed in zip(filenames, labels.cpu().tolist(), pred.cpu().tolist()):
                predictions.append({
                    "file": str(filename), "y_true": int(actual), "y_pred": int(guessed),
                    "correct": bool(actual == guessed),
                })

            fallback_position += len(labels)

    if total == 0:
        raise RuntimeError("Stress test evaluation processed zero records.")
        
    return {
        "accuracy": correct / total,
        "loss": loss_sum / total,
        "n": total,
        "predictions": predictions,
    }

def run_stress_test(backbone, linear_head, base_dataset, noise_mixer, snr_levels, device):
    """Compatibility helper, callers receive structured per SNR results."""
    results = {}

    for snr in snr_levels:
        base_dataset.snr_db = snr
        base_dataset.noise_mixer = noise_mixer

        if hasattr(base_dataset, "crop_mode"):
            base_dataset.crop_mode = "center"

        dataloader = DataLoader(base_dataset, batch_size=4, shuffle=False)
        results["Clean" if snr is None else f"{snr}dB"] = evaluate_at_snr(backbone, linear_head, dataloader, device)

    return results


if __name__ == "__main__":
    class MockBackbone(nn.Module):
        def forward(self, x):
            return torch.flatten(x, 1)[:, :512]
        
    class MockHead(nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 10, device = x.device)
        
    class MockDataset(Dataset):
        def __len__(self):
            return 8
        def __getitem__(self, idx):
            return torch.randn(1, 64, 512), torch.randint(0, 10, (1,)).item(), f"{idx}.wav"
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(evaluate_at_snr(MockBackbone().to(device), 
                          MockHead().to(device), 
                          DataLoader(MockDataset(), 
                                     batch_size = 4), 
                                     device)["accuracy"])