"""Verify PyTorch installation and selected accelerator numerics."""
import argparse
import torch


def main(argv = None):
    argparse.ArgumentParser(description = __doc__).parse_args(argv)
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    cpu = torch.tensor([1.0, 2.0, 3.0])
    device_result = (cpu.to(device) ** 2).cpu()
    cpu_result = cpu ** 2
    
    print(f"Selected device: {device}")
    print(f"CPU result: {cpu_result}")
    print(f"Device result: {device_result}")
    
    if not torch.equal(cpu_result, device_result):
        raise RuntimeError(f"CPU/device operation mismatch: {cpu_result} vs {device_result}.")
        
    print("Environment numerical check passed.")


if __name__ == "__main__":
    main()
