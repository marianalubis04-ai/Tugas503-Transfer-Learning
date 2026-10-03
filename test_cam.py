import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 1. Konfigurasi Device & Kelas
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['hp', 'mouse']  # Urutan abjad dari nama folder dataset

# 2. Load Model ResNet-18
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Load bobot yang sudah dilatih
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# 3. Transformasi Gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Buka Kamera RGB (CAM_ID = 2)
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print("Memulai deteksi real-time... Tekan [Q] untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Konversi frame BGR OpenCV ke PIL Image (RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    # Preprocessing & Prediksi
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probabilities, 1)

    label = class_names[preds[0].item()]
    confidence = conf[0].item() * 100

    # Tampilkan hasil di layar
    text = f"Prediksi: {label} ({confidence:.1f}%)"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Deteksi Real-time", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()