import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import csv
from PIL import Image

# === Load and convert image to RGB safely ===
def load_rgb_image(image_path):
    try:
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil).astype(np.uint8)
        return img_np
    except Exception as e:
        print(f"❌ Error loading {image_path}: {e}")
        return None

# === Load and encode all known faces from directory ===
def load_known_faces(directory):
    encodings = []
    names = []

    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".png")):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(directory, filename)
            img_rgb = load_rgb_image(image_path)

            if img_rgb is None:
                continue

            try:
                face_encoding = face_recognition.face_encodings(img_rgb)[0]
                encodings.append(face_encoding)
                names.append(name)
                print(f"✅ Loaded and encoded: {name}")
            except Exception as e:
                print(f"⚠️ Could not encode {name}: {e}")

    return encodings, names

# === Create attendance logs folder if not exists ===
attendance_dir = "attendance_logs"
os.makedirs(attendance_dir, exist_ok=True)

# === Initialize camera ===
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# === Load known faces ===
print("🔐 Loading known faces...")
known_face_encodings, known_face_names = load_known_faces("captured_faces")

if not known_face_encodings:
    print("❌ No known faces loaded.")
    exit()

# === Setup attendance ===
students = known_face_names.copy()
now = datetime.now()
current_date = now.strftime("%d-%m-%y")
csv_path = os.path.join(attendance_dir, f"{current_date}.csv")
csv_file = open(csv_path, "w+", newline="")
lnwriter = csv.writer(csv_file)

print("🟢 Attendance session started. Press 'q' to quit manually.")

# === Start video loop ===
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Failed to read from webcam.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        name = "Unknown"
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        if name in students:
            students.remove(name)
            current_time = datetime.now().strftime("%H:%M:%S")
            lnwriter.writerow([name, current_time])
            print(f"✅ Marked present: {name} at {current_time}")
            cv2.putText(frame, f"{name} PRESENT", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("📸 HMC Attendance", frame)

    # Exit if all students marked or user presses 'q'
    if not students:
        print("✅ All students marked present. Exiting...")
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("❌ Manual exit.")
        break

# === Cleanup ===
video_capture.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"📁 Attendance saved at: {csv_path}")
print("📁 Attendance saved and session closed.")
