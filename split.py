import os
import shutil
import random

# Direktori asal dan tujuan
RAW_DIR = "dataset_raw"
OUTPUT_DIR = "dataset"

# Rasio pembagian: Train 70%, Val 15%, Test 15%
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

random.seed(42)  # Agar pembagian konsisten

# Pastikan folder asal ada
if not os.path.exists(RAW_DIR):
    print(f"Error: Folder '{RAW_DIR}' tidak ditemukan!")
    exit()

classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]

if len(classes) < 2:
    print(f"Peringatan: Baru ditemukan {len(classes)} kelas ({classes}). Disarankan minimal 2 kelas!")

print("Mulai membagi dataset...")

for class_name in classes:
    class_path = os.path.join(RAW_DIR, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]
    
    splits = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }
    
    for split_name, split_files in splits.items():
        split_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for img in split_files:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_dir, img)
            shutil.copy(src, dst)
            
    print(f"- Kelas '{class_name}': Total {n_total} gambar -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

print("\nProses pemisahan dataset selesai! Folder 'dataset' siap digunakan untuk training.")