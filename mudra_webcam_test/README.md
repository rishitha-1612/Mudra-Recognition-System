# Mudra Recognition System

Real-time hand mudra recognition using MediaPipe + TensorFlow CNN, with:
- OpenCV live inference scripts
- Flask backend integration layer
- React model-selection UI

The CNN pipeline is unchanged; the web layer only integrates around it.

## Project Structure

```text
mudra_webcam_test/
  app.py                    # Flask backend launcher + API integration
  original.py               # Webcam inference (hand crop pipeline)
  skin_segmenation.py       # Webcam inference (HSV/YCrCb segmentation pipeline)
  hand_landmarker.task
  models/
    mudra_mobilenetv2_final.keras
    class_names.json
  frontend/                 # React + Vite UI
```

## Requirements

- Python 3.10+
- Node.js 18+ and npm (for frontend)
- Webcam

## Setup

Run from:

`C:\Users\RISHITHA\OneDrive\Desktop\mudra_webcam_test`

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r .\mudra_webcam_test\requirements.txt
```

## How To Run (Frontend Buttons + Backend)

Use two terminals.

### Terminal 1: Backend (Flask integration)

```powershell
cd C:\Users\RISHITHA\OneDrive\Desktop\mudra_webcam_test
.\.venv\Scripts\python .\mudra_webcam_test\app.py --camera 0
```

Optional:

```powershell
.\.venv\Scripts\python .\mudra_webcam_test\app.py --camera 1 --min-confidence 0.6
```

### Terminal 2: Frontend UI

```powershell
cd C:\Users\RISHITHA\OneDrive\Desktop\mudra_webcam_test\mudra_webcam_test\frontend
npm install
npm run dev
```

Open the URL printed by Vite (usually `http://localhost:5173`).

When you click:
- `Try YCrCb Model` -> backend launches:
  `.\.venv\Scripts\python .\mudra_webcam_test\skin_segmenation.py --camera 0 --method ycrcb`
- `Try HSV Model` -> backend launches:
  `.\.venv\Scripts\python .\mudra_webcam_test\skin_segmenation.py --camera 0 --method hsv`

## API Contract

### `GET /health`

Health status for backend engine.

### `POST /api/run-segmentation`

Launches `skin_segmenation.py` with selected method.

Request body:

```json
{
  "method": "ycrcb",
  "camera": 0
}
```

Success response:

```json
{
  "ok": true,
  "method": "ycrcb",
  "pid": 12345,
  "command": "C:\\...\\python.exe C:\\...\\skin_segmenation.py --camera 0 --method ycrcb"
}
```

### `GET /api/segmentation-status`

Returns current launched process status.

### `POST /api/stop-segmentation`

Stops the launched `skin_segmenation.py` process (if running).

### `GET /api/live-prediction?method=ycrcb|hsv` (optional compatibility endpoint)

Available for live API mode, but your current UI flow uses button-triggered script launch.

## Standalone Script Modes

If needed, you can still run the scripts directly:

```powershell
.\.venv\Scripts\python .\mudra_webcam_test\original.py --camera 0
.\.venv\Scripts\python .\mudra_webcam_test\skin_segmenation.py --camera 0 --method ycrcb
.\.venv\Scripts\python .\mudra_webcam_test\skin_segmenation.py --camera 0 --method hsv
```

## Troubleshooting

- Webcam not opening: try `--camera 1` or close other apps using the camera.
- Buttons not launching script: confirm backend is running on `127.0.0.1:5000`.
- TensorFlow oneDNN log is informational. To suppress in current terminal:

```powershell
$env:TF_ENABLE_ONEDNN_OPTS = "0"
```
