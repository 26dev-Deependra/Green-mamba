from models.baselines import get_resnet50, get_vit, get_mobilenet_v3
from models.green_mamba import GreenMamba
import argparse
import torch
import time
import tracemalloc
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def measure_efficiency(model, model_name, input_size):
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    # 1. Warmup
    print(f"[{model_name}] Warming up...")
    for _ in range(5):
        _ = model(dummy_input)

    # 2. Latency
    print(f"[{model_name}] Measuring Latency (50 runs)...")
    timings = []
    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            _ = model(dummy_input)
            timings.append((time.time() - start) * 1000)  # ms

    avg_latency = np.mean(timings)
    fps = 1000 / avg_latency

    # 3. Peak Memory
    tracemalloc.start()
    with torch.no_grad():
        _ = model(dummy_input)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    # 4. Params
    params = sum(p.numel() for p in model.parameters()) / 1e6

    # 5. Energy Estimation (Approx 5W for Pi 4 load)
    energy_j = 5.0 * (avg_latency / 1000)

    return {
        "Model": model_name,
        "Params (M)": round(params, 2),
        "Latency (ms)": round(avg_latency, 2),
        "FPS": round(fps, 2),
        "Memory (MB)": round(peak_mb, 2),
        "Energy (J)": round(energy_j, 4)
    }


def main():
    models_to_test = [
        ('Green-Mamba', GreenMamba(num_classes=3, use_cuda_kernel=False), 128),
        ('MobileNetV3-Small', get_mobilenet_v3(num_classes=3, pretrained=False), 224),
        ('ResNet50', get_resnet50(num_classes=3, pretrained=False), 180),
        ('ViT-Base', get_vit(num_classes=3, pretrained=False), 224)
    ]

    results = []
    print("Starting Hardware Benchmark on CPU...")
    print("-" * 60)

    for name, model, size in models_to_test:
        try:
            res = measure_efficiency(model, name, size)
            results.append(res)
            print(f"Result: {res}")
        except Exception as e:
            print(f"Failed to benchmark {name}: {e}")

    # Save
    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df)
    df.to_csv("benchmark_results.csv", index=False)


if __name__ == "__main__":
    main()
