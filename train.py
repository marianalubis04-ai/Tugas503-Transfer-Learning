import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device: {device}")

# 2. Transformasi Data
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'dataset'
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True)
    for x in ['train', 'val']
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)

# 3. Fungsi Trainer dengan Mode Pendekatan
def train_model(mode_name='feature_extraction', num_epochs=10):
    print(f"\n==========================================")
    print(f" Memulai Training Mode: {mode_name.upper()}")
    print(f"==========================================")
    
    # Inisialisasi Model ResNet-18
    if mode_name == 'scratch':
        model = models.resnet18(weights=None) # Tanpa pretrained
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Pengaturan Freeze Layer
    if mode_name == 'feature_extraction':
        for param in model.parameters():
            param.requires_grad = False
    elif mode_name == 'partial':
        # Bekukan layer awal (layer1 & layer2), buka layer3 & layer4
        for name, param in model.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    start_time = time.time()

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = (running_corrects.double() / len(image_datasets[phase])).item()

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Val Acc: {history['val_acc'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}")

    total_time = time.time() - start_time
    return history, total_time

# 4. Jalankan 3 Pendekatan
modes = ['feature_extraction', 'partial', 'scratch']
results = {}

for m in modes:
    hist, duration = train_model(m, num_epochs=10)
    results[m] = {'history': hist, 'time': duration}

# 5. Cetak Tabel Hasil
print("\n" + "="*60)
print(f"{'Pendekatan':<20} | {'Best Val Acc':<15} | {'Waktu (detik)':<15}")
print("="*60)
for m in modes:
    best_acc = max(results[m]['history']['val_acc'])
    t_sec = results[m]['time']
    print(f"{m:<20} | {best_acc*100:>13.2f}% | {t_sec:>13.2f}s")
print("="*60)

# 6. Plot Grafik Akurasi & Loss per Epoch
plt.figure(figsize=(12, 5))

# Grafik Akurasi Validasi
plt.subplot(1, 2, 1)
for m in modes:
    plt.plot(range(1, 11), results[m]['history']['val_acc'], label=f"{m}")
plt.title('Akurasi Validasi per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)

# Grafik Loss Validasi
plt.subplot(1, 2, 2)
for m in modes:
    plt.plot(range(1, 11), results[m]['history']['val_loss'], label=f"{m}")
plt.title('Loss Validasi per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('grafik_hasil_3_pendekatan.png')
print("\nGrafik berhasil disimpan sebagai 'grafik_hasil_3_pendekatan.png'!")