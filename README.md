# 📸 Face Recognition Attendance System 🧑‍🤝‍🧑

Welcome to the **Face Recognition Attendance System** — an easy-to-use, real-time facial recognition-based attendance tracker built with Python! 🎉

---

## 🚀 Features

- 🤳 Capture faces of students/staff for attendance reference
- 🎥 Real-time webcam face detection & recognition
- 📝 Auto-generate daily attendance logs in CSV format
- ⏰ Mark attendance only once per session
- 🛠️ Simple and modular Python code — easy to extend!

---

## 🧰 Requirements

- Python 3.x 🐍
- `face_recognition` library 🧠
- `opencv-python` (cv2) 🎥
- `numpy` 🔢
- `Pillow` (PIL) 🖼️

---

## 💻 How to Use

1. **Capture Known Faces**  
   Run the capture script and save face images:
   ```bash
   python capture_faces.py
  Press c to capture your face, q to quit.

2. **Run Attendance**
Start the attendance system:
python main.py
The webcam will open, and recognized faces will be marked present!
Press q to stop.

4. **View Logs**
Attendance logs are saved daily in the attendance_logs/ folder as CSV files named by date.

Made with ❤️ by Divija

Happy Attendance! 🎉
