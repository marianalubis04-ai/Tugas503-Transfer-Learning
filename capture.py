import cv2
import os

# Pakai ID 2 untuk Kamera RGB Standar (bukan IR)
CAM_ID = 2

SAVE_DIR = "dataset_raw/mouse"  # Sesuaikan dengan kelas objek Anda
CLASS_NAME = "mouse"

os.makedirs(SAVE_DIR, exist_ok=True)

# Buka Kamera RGB
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)

# Setel format warna ke MJPG agar citra jernih
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
print(f"Mengambil gambar untuk kelas: {CLASS_NAME}")
print("Tekan [SPASI] untuk menyimpan gambar. Tekan [Q] untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Gagal mengambil frame dari kamera.")
        break

    display_frame = frame.copy()
    cv2.putText(display_frame, f"Kelas: {CLASS_NAME} | Tersimpan: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Capture Data", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Tekan Spasi
        img_name = f"{CLASS_NAME}_{count:03d}.png"
        cv2.imwrite(os.path.join(SAVE_DIR, img_name), frame)
        print(f"Gambar tersimpan: {img_name}")
        count += 1
    elif key == ord('q'):  # Tekan Q untuk keluar
        break

cap.release()
cv2.destroyAllWindows()