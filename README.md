# 🖐️ Mudra Recognition System

A real-time hand mudra recognition system using **MediaPipe**, **TensorFlow CNN**, and **OpenCV**, integrated with a **Flask backend** and **React frontend UI**.

---

## 🚀 Features

* 🎥 Real-time webcam-based hand gesture recognition
* 🤖 Deep Learning model (MobileNetV2 CNN)
* 🧠 MediaPipe hand tracking
* 🌐 Flask backend integration
* ⚛️ React + Vite frontend
* 🔄 Supports **HSV** and **YCrCb** segmentation

---

## 🏗️ Project Structure

```
.
├── app.py
├── original.py
├── skin_segmenation.py
├── hand_landmarker.task
│
├── models/
│   ├── mudra_mobilenetv2_final.keras
│   └── class_names.json
│
├── frontend/
│
└── README.md
```

---

## ⚙️ Requirements

* Python 3.10+
* Node.js 18+
* Webcam

---

## 🧪 Setup

### 1. Clone Repository

```
git clone https://github.com/your-username/mudra-recognition-system.git
cd mudra-recognition-system
```

---

### 2. Create Virtual Environment

```
python -m venv .venv
.\.venv\Scripts\activate
pip install flask opencv-python tensorflow mediapipe numpy
```

---

### 3. Run Backend

```
python app.py --camera 0
```

---

### 4. Run Frontend

```
cd frontend
npm install
npm run dev
```

---

### 5. Open Browser

```
http://localhost:5173
```

---

## 🎮 Usage

* Click **Try YCrCb Model**
* Click **Try HSV Model**
* Use **Stop** before switching

---

## 🔌 API Endpoints

* `/health` → Check backend status
* `/api/run-segmentation` → Start model
* `/api/stop-segmentation` → Stop model
* `/api/segmentation-status` → Check running status

---

## ⚠️ Troubleshooting

* Camera not working → Try `--camera 1`
* Backend not responding → Check `/health`
* TensorFlow logs → Can be ignored

---

## 🛠️ Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* MediaPipe
* Flask
* React + Vite

---

## 👩‍💻 Authors

* Sowmya P R
* Rishitha Rasineni

📌 Developed as an academic project for real-time gesture recognition.

---

## ⭐ Future Scope

* Deploy as web application
* Add more gestures
* Improve accuracy
* Mobile integration

---
