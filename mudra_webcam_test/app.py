import argparse
import base64
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request

from original import (
    DEFAULT_CLASS_NAMES_PATH,
    DEFAULT_MODEL_PATH,
    HandCropper,
    MudraRecognizer,
    PredictionResult,
    infer_hasta_category,
)
from skin_segmenation import segment_hand


VALID_METHODS = {"ycrcb", "hsv"}


def _encode_image(image: Optional[np.ndarray]) -> Optional[str]:
    if image is None or image.size == 0:
        return None
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _normalize_method(raw: Optional[str]) -> str:
    if raw is None:
        return "ycrcb"
    value = raw.strip().lower()
    return value if value in VALID_METHODS else "ycrcb"


class LiveMudraEngine:
    def __init__(
        self,
        model_path: Path,
        class_names_path: Optional[Path],
        camera_index: int = 0,
        min_confidence: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.camera_index = camera_index
        self.min_confidence = min_confidence

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._latest: dict[str, dict] = {"ycrcb": self._empty_payload(), "hsv": self._empty_payload()}
        self._last_fps: Optional[float] = None
        self._error: Optional[str] = None

        self._recent_scores: dict[str, deque[np.ndarray]] = {
            "ycrcb": deque(maxlen=5),
            "hsv": deque(maxlen=5),
        }

    def _empty_payload(self) -> dict:
        return {
            "prediction": "Awaiting detection",
            "confidence": 0.0,
            "type": "Unknown",
            "top3": [],
            "images": {
                "original": None,
                "segmented": None,
                "processed": None,
            },
            "timestamp": None,
            "fps": None,
            "error": None,
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="live-mudra-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def latest(self, method: str) -> dict:
        selected = _normalize_method(method)
        with self._lock:
            payload = dict(self._latest[selected])
            payload["images"] = dict(self._latest[selected]["images"])
            payload["fps"] = self._last_fps
            payload["error"] = self._error
            return payload

    def _run_loop(self) -> None:
        recognizer = None
        cropper = None
        capture = None
        try:
            recognizer = MudraRecognizer(self.model_path, self.class_names_path)
            cropper = HandCropper(
                padding=0.18,
                vertical_padding_boost=0.24,
                horizontal_padding_boost=0.12,
                min_box_fraction=0.28,
                max_num_hands=2,
            )

            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                raise RuntimeError("Could not open webcam. Try a different camera index.")

            last_time = time.perf_counter()

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Failed to read frame from webcam.")

                frame = cv2.flip(frame, 1)
                hand_crop, box, annotated, hand_count = cropper.crop_hand(frame)

                if box is not None:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                now = time.perf_counter()
                elapsed = max(1e-6, now - last_time)
                last_time = now
                fps = 1.0 / elapsed

                timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                original_b64 = _encode_image(annotated)

                frame_payloads = {}
                for method in VALID_METHODS:
                    payload = self._empty_payload()
                    payload["timestamp"] = timestamp_iso
                    payload["images"]["original"] = original_b64

                    if hand_crop is None or hand_crop.size == 0:
                        self._recent_scores[method].clear()
                        frame_payloads[method] = payload
                        continue

                    segmented_preview, _ = segment_hand(hand_crop, method)
                    prediction, model_input_preview = recognizer.predict(
                        segmented_preview,
                        detected_hands=hand_count,
                    )
                    self._recent_scores[method].append(prediction.scores)

                    averaged_scores = np.mean(np.stack(self._recent_scores[method], axis=0), axis=0)
                    best_index = int(np.argmax(averaged_scores))
                    averaged_label = recognizer.class_names[best_index]
                    averaged_prediction = PredictionResult(
                        label=averaged_label,
                        hasta_category=infer_hasta_category(averaged_label, hand_count),
                        confidence=float(averaged_scores[best_index]),
                        scores=averaged_scores,
                        class_names=recognizer.class_names,
                    )

                    if averaged_prediction.confidence >= self.min_confidence:
                        payload["prediction"] = averaged_prediction.label
                        payload["confidence"] = round(averaged_prediction.confidence * 100, 1)
                        payload["type"] = averaged_prediction.hasta_category
                        payload["top3"] = [
                            {"label": label, "score": round(score * 100, 1)}
                            for label, score in averaged_prediction.top_k(3)
                        ]

                    payload["images"]["segmented"] = _encode_image(segmented_preview)
                    payload["images"]["processed"] = _encode_image(model_input_preview)
                    frame_payloads[method] = payload

                with self._lock:
                    self._latest = frame_payloads
                    self._last_fps = round(fps, 1)
                    self._error = None
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            if capture is not None:
                capture.release()
            if cropper is not None:
                cropper.close()


class SegmentationScriptRunner:
    def __init__(self, script_path: Path) -> None:
        self.script_path = script_path
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._last_method: Optional[str] = None
        self._last_error: Optional[str] = None

    def _cleanup_if_needed(self) -> None:
        if self._process is not None and self._process.poll() is not None:
            self._process = None

    def start(self, method: str, camera: int = 0) -> dict:
        selected_method = _normalize_method(method)
        with self._lock:
            self._cleanup_if_needed()
            if self._process is not None:
                self.stop()

            command = [
                sys.executable,
                str(self.script_path),
                "--camera",
                str(camera),
                "--method",
                selected_method,
            ]
            try:
                self._process = subprocess.Popen(command, cwd=str(self.script_path.parent))
                self._last_method = selected_method
                self._last_error = None
                return {
                    "ok": True,
                    "method": selected_method,
                    "pid": self._process.pid,
                    "command": " ".join(command),
                }
            except Exception as exc:
                self._process = None
                self._last_error = str(exc)
                return {"ok": False, "error": self._last_error}

    def stop(self) -> None:
        with self._lock:
            self._cleanup_if_needed()
            if self._process is None:
                return
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def status(self) -> dict:
        with self._lock:
            self._cleanup_if_needed()
            return {
                "running": self._process is not None,
                "pid": self._process.pid if self._process is not None else None,
                "method": self._last_method,
                "error": self._last_error,
            }

    def is_running(self) -> bool:
        return bool(self.status().get("running"))


def create_app(engine: LiveMudraEngine, runner: SegmentationScriptRunner) -> Flask:
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health() -> tuple[dict, int]:
        run_status = runner.status()
        if run_status["running"]:
            return {"ok": True, "mode": "script", "runner": run_status}, 200
        payload = engine.latest("ycrcb")
        return {"ok": payload.get("error") is None, "mode": "api", "error": payload.get("error")}, 200

    @app.route("/api/live-prediction", methods=["GET"])
    def live_prediction():
        if runner.is_running():
            return jsonify(
                {
                    "prediction": "Script mode active",
                    "confidence": 0.0,
                    "type": "Unknown",
                    "top3": [],
                    "images": {"original": None, "segmented": None, "processed": None},
                    "timestamp": None,
                    "fps": None,
                    "error": "skin_segmenation.py is running from button launch. Stop it to use API live feed.",
                }
            )
        if not engine.is_running():
            engine.start()
        method = _normalize_method(request.args.get("method"))
        return jsonify(engine.latest(method))

    @app.route("/api/run-segmentation", methods=["POST"])
    def run_segmentation():
        payload = request.get_json(silent=True) or {}
        method = _normalize_method(payload.get("method"))
        camera = int(payload.get("camera", 0))
        engine.stop()
        result = runner.start(method=method, camera=camera)
        status_code = 200 if result.get("ok") else 500
        return jsonify(result), status_code

    @app.route("/api/stop-segmentation", methods=["POST"])
    def stop_segmentation():
        runner.stop()
        return jsonify({"ok": True})

    @app.route("/api/segmentation-status", methods=["GET"])
    def segmentation_status():
        return jsonify(runner.status())

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flask bridge for Mudra live frontend integration.")
    parser.add_argument("--host", default="127.0.0.1", help="Flask host.")
    parser.add_argument("--port", type=int, default=5000, help="Flask port.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Prediction confidence threshold.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to Keras model.")
    parser.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES_PATH, help="Path to class_names.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names_path = args.class_names if args.class_names.exists() else None
    engine = LiveMudraEngine(
        model_path=args.model,
        class_names_path=class_names_path,
        camera_index=args.camera,
        min_confidence=args.min_confidence,
    )
    runner = SegmentationScriptRunner(script_path=Path(__file__).resolve().parent / "skin_segmenation.py")
    app = create_app(engine, runner)
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        engine.stop()
        runner.stop()


if __name__ == "__main__":
    main()
