import cv2
import os
from PIL import Image
import numpy as np

# === Constants ===
SAVE_DIR = "captured_faces"

# === Create the directory if it doesn't exist ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === Ask for person's name ===
name = input("Enter the name of the person to capture: ").strip().lower()
save_path = os.path.join(SAVE_DIR, f"{name}.jpg")

if os.path.exists(save_path):
    confirm = input(f"⚠️ File for '{name}' already exists. Overwrite? (y/n): ").strip().lower()
    if confirm != "y":
        print("❌ Capture cancelled.")
        exit()

# === Initialize webcam ===
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(f"📸 Capturing face for: {name}. Press 'c' to capture, 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Failed to access webcam.")
        break

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        # Convert frame to RGB using PIL
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.save(save_path)
            print(f"✅ Saved face image in RGB: {save_path}")
        except Exception as e:
            print(f"❌ Error saving image: {e}")
        break
    elif key == ord("q"):
        print("❌ Capture cancelled by user.")
        break

# === Cleanup ===
video_capture.release()
cv2.destroyAllWindows()
