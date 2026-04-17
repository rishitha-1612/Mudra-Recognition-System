# MudraVision: Implementation-Centric IEEE Draft for a Real-Time Webcam Mudra Recognition System

## Abstract
This paper documents the implemented behavior of the MudraVision codebase in `mudra_webcam_test`. The system performs real-time webcam-based mudra classification using a MediaPipe hand detector, an optional skin-segmentation stage (YCrCb or HSV), and a TensorFlow/Keras classifier served through a Flask API and visualized with a React dashboard. The deployed classifier artifact (`mudra_mobilenetv2_final.keras`) has input shape `(None, 224, 224, 1)`, output shape `(None, 50)`, and `2,327,154` parameters. The backend exposes `/health` and `/api/live-prediction` endpoints, while the frontend polls every 500 ms and renders original, segmented, and model-input views with prediction metadata. This draft reports only repository-verifiable facts and explicitly avoids unverified performance claims.

## Index Terms
hand gesture recognition, mudra recognition, MediaPipe, TensorFlow, Flask, React, OpenCV, real-time inference

## I. Scope and Evidence Policy
This document is limited to facts verifiable from:
1. Source code in `mudra_webcam_test/*`.
2. Configuration and dependency files in the same repository.
3. Runtime inspection of the model artifact performed in this workspace.

No dataset statistics, training procedure details, benchmark accuracy, or latency benchmarks are claimed because those are not present as reproducible evaluation artifacts in the repository.

## II. System Artifacts
The repository includes the following runtime artifacts:
1. `app.py`: Flask bridge and live inference engine.
2. `original.py`: core recognizer, hand cropper, and standalone webcam pipeline.
3. `skin_segmenation.py`: segmentation-augmented pipeline and segmentation functions.
4. `models/mudra_mobilenetv2_final.keras`: deployed classifier model.
5. `models/class_names.json`: 50 class labels.
6. `hand_landmarker.task`: MediaPipe hand landmark model file.
7. `frontend/*`: React + Vite dashboard.

## III. Backend Processing Pipeline
### A. Frame Loop Behavior
In `app.py`, `LiveMudraEngine`:
1. Loads `MudraRecognizer`.
2. Builds `HandCropper` with:
   - `padding=0.18`
   - `vertical_padding_boost=0.24`
   - `horizontal_padding_boost=0.12`
   - `min_box_fraction=0.28`
   - `max_num_hands=2`
3. Opens webcam using `cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)` with fallback to default backend.
4. Reads frame, horizontal flips (`cv2.flip(frame, 1)`), crops hand ROI, and estimates FPS from `time.perf_counter()`.
5. Runs both segmentation methods (`ycrcb`, `hsv`) each frame.
6. Performs classification and temporal smoothing per method over a deque of length 5.
7. Encodes `original`, `segmented`, and `processed` images as JPEG (quality 85), base64.

### B. Output Payload Fields
Each method-specific payload includes:
1. `prediction` (string),
2. `confidence` (percentage value, rounded to 0.1),
3. `type` (`Samyuktha Hastha` or `Asamyuktha Hastha`),
4. `top3` (label-score list),
5. `images.original`, `images.segmented`, `images.processed`,
6. `timestamp` (local-time formatted string `%Y-%m-%dT%H:%M:%S`),
7. `fps`,
8. `error`.

When no hand crop exists, per-method score buffers are cleared and default payload values are returned.

## IV. API Surface
Implemented routes in `app.py`:
1. `GET /health`
   - Returns `{ ok: bool, error: <string|null> }`.
2. `GET /api/live-prediction?method=ycrcb|hsv`
   - Returns live payload for selected method.
   - Invalid/missing method normalizes to `ycrcb`.

## V. Hand Detection and Cropping Details
`original.py` defines `HandCropper` using MediaPipe Tasks `HandLandmarker`.

Verified behavior:
1. Supports up to two hands (`num_hands=max_num_hands`).
2. Uses video running mode for live stream (`VisionTaskRunningMode.VIDEO` by default).
3. Computes one fused bounding box from all detected landmark points.
4. Enforces minimum crop size as fractions of frame dimensions (`min_box_fraction`).
5. Applies temporal box smoothing with factor `smoothing=0.75`.
6. Returns `(crop, box, annotated_frame, hand_count)`.

If no landmarks are detected, it returns `None` crop and resets previous box state.

## VI. Segmentation Implementation Details
`skin_segmenation.py` implements two threshold branches.

### A. YCrCb Mask
1. Convert BGR -> YCrCb.
2. Threshold with:
   - lower `[0, 133, 77]`
   - upper `[255, 173, 127]`

### B. HSV Mask
1. Convert BGR -> HSV.
2. Threshold range 1:
   - lower `[0, 25, 40]`
   - upper `[25, 180, 255]`
3. Threshold range 2:
   - lower `[160, 25, 40]`
   - upper `[180, 180, 255]`
4. Merge both masks with bitwise OR.

### C. Shared Refinement
`refine_mask` applies:
1. median blur (`k=5`),
2. morphological open (`3x3` kernel),
3. morphological close (`5x5` kernel),
4. Gaussian blur (`5x5`),
5. binary threshold at `80`.

Segmented output is composited over constant gray background value `127` and then cropped to mask bounds.

## VII. Classifier Characteristics (Runtime-Verified)
Model properties, loaded through project code path:
1. model name: `mudra_mobilenetv2`,
2. input shape: `(None, 224, 224, 1)`,
3. output shape: `(None, 50)`,
4. parameter count: `2,327,154`,
5. expected channels in preprocessor: `1` (grayscale).

Preprocessing in `MudraRecognizer.preprocess`:
1. Convert BGR -> grayscale.
2. Resize to `224x224`.
3. Cast to `float32`.
4. Expand to model input batch shape.

Prediction output:
1. Uses `np.argmax` for top-1.
2. Retains full score vector.
3. Derives top-k using score sort descending.

## VIII. Class Taxonomy Facts
`models/class_names.json` contains exactly 50 classes.

`SAMYUKTHA_HASTHA_MUDRAS` in `original.py` contains 21 class labels, and all 21 are present in `class_names.json` (verified by runtime check).

`infer_hasta_category(label, detected_hands)` logic:
1. If label is in `SAMYUKTHA_HASTHA_MUDRAS`, return `Samyuktha Hastha`.
2. Else, if `detected_hands >= 2`, return `Samyuktha Hastha`.
3. Otherwise, return `Asamyuktha Hastha`.

## IX. Temporal Stabilization Logic
Both `app.py` and standalone scripts maintain `deque(maxlen=5)` for recent score vectors.
Smoothed score is:

\[
\bar{s} = \frac{1}{N}\sum_{i=1}^{N} s_i,\; N \leq 5
\]

Displayed class is `argmax(\bar{s})`. If no valid hand crop is present, deque is cleared.

## X. Frontend Behavior
### A. Stack and Build
From `frontend/package.json`:
1. React `^18.3.1`
2. Framer Motion `^11.3.19`
3. Vite `^5.4.8`
4. Tailwind CSS `^3.4.13`

### B. App Flow
`frontend/src/App.jsx`:
1. Intro screen duration fixed at `5000` ms.
2. Model selection page requires choosing `ycrcb` or `hsv`.
3. Dashboard loads with endpoint `/api/live-prediction?method=<selected>`.

### C. Data Polling
`useLiveMudraData(endpoint, intervalMs)`:
1. default interval in app usage is `500` ms,
2. fetches JSON with `Accept: application/json`,
3. normalizes images:
   - accepts `data:image...`,
   - accepts `http/https`,
   - accepts relative URLs,
   - otherwise treats value as raw base64 JPEG.

### D. Dev Proxy
`frontend/vite.config.js` proxies `/api` to `http://127.0.0.1:5000`.

## XI. Dependency and Runtime Facts
From `requirements.txt`:
1. `flask==3.0.3`
2. `mediapipe==0.10.33`
3. `numpy==2.2.6`
4. `opencv-python==4.13.0.92`
5. `tensorflow==2.21.0`

`original.py` sets:
1. `TF_ENABLE_ONEDNN_OPTS=0` (if unset),
2. `TF_CPP_MIN_LOG_LEVEL=2` (if unset).

Model-load fallback for legacy config:
1. If direct load fails with `quantization_config` issue, code rewrites `config.json` inside a temporary `.keras` archive by removing `quantization_config` keys recursively.
2. Then loads the sanitized archive via `tf.keras.models.load_model(..., compile=False)`.

## XII. What Is Not Claimed
This document does not claim:
1. classification accuracy,
2. precision/recall/F1,
3. confusion matrix,
4. latency percentile benchmarks,
5. dataset size or train/validation split,
6. training hyperparameters.

These items are not provided as reproducible evidence in the current repository.

## XIII. Conclusion
The repository implements a complete real-time mudra recognition product path:
1. webcam acquisition and hand localization,
2. optional segmentation with two color-space methods,
3. 50-class CNN inference with temporal smoothing,
4. Flask API transport,
5. React dashboard visualization.

All statements in this draft are constrained to code-level and runtime-verifiable facts from the present workspace.

## References (Repository-Backed)
[1] `mudra_webcam_test/app.py`  
[2] `mudra_webcam_test/original.py`  
[3] `mudra_webcam_test/skin_segmenation.py`  
[4] `mudra_webcam_test/models/class_names.json`  
[5] `mudra_webcam_test/models/mudra_mobilenetv2_final.keras`  
[6] `mudra_webcam_test/requirements.txt`  
[7] `mudra_webcam_test/frontend/src/App.jsx`  
[8] `mudra_webcam_test/frontend/src/hooks/useLiveMudraData.js`  
[9] `mudra_webcam_test/frontend/vite.config.js`  
[10] `mudra_webcam_test/frontend/package.json`
