import time
import torch
import torch.nn as nn
from torchvision import models

# 1. Tentukan Perangkat (CPU)
device = torch.device("cpu")
print(f"Mengukur latensi pada perangkat: {device}")

# Dummy input sesuai ukuran citra standar ImageNet (Batch Size = 1, Channel = 3, 224x224)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

def measure_latency(model, model_name, num_runs=100):
    model.eval()
    model.to(device)
    
    # Warm-up (pemanasan model)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    # Pengukuran waktu inferensi
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
    fps = 1000 / avg_latency_ms
    
    print(f"\n--- Hasil {model_name} ---")
    print(f"Rata-rata Latensi: {avg_latency_ms:.2f} ms")
    print(f"Estimasi Kecepatan: {fps:.2f} FPS")
    return avg_latency_ms

# 2. Load Model ResNet-18
resnet = models.resnet18()
resnet.fc = nn.Linear(resnet.fc.in_features, 2)

# 3. Load Model MobileNetV3-Small
mobilenet = models.mobilenet_v3_small()
mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, 2)

# 4. Jalankan Pengukuran
print("\nMemulai pengujian latensi...")
res_lat = measure_latency(resnet, "ResNet-18")
mob_lat = measure_latency(mobilenet, "MobileNetV3-Small")

print("\n==========================================")
print(f"Perbandingan: MobileNetV3-Small {res_lat / mob_lat:.2f}x lebih cepat dibanding ResNet-18.")
print("==========================================")