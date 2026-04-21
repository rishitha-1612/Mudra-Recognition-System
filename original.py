import os
import argparse
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Optional
import time

import cv2
import mediapipe as mp
import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)


PROJECT_DIR = Path(__file__).resolve().parent


FALLBACK_CLASS_NAMES = [
    "Alapadmam",
    "Anjali",
    "Aralam",
    "Ardhachandran",
    "Ardhapathaka",
    "Berunda",
    "Bramaram",
    "Chakra",
    "Chandrakala",
    "Chaturam",
    "Garuda",
    "Hamsapaksha",
    "Hamsasyam",
    "Kangulam",
    "Kapith",
    "Kapotham",
    "Karkatta",
    "Kartariswastika",
    "Katakamukha_1",
    "Katakamukha_2",
    "Katakamukha_3",
    "Katakavardhana",
    "Katrimukha",
    "Khatva",
    "Kilaka",
    "Kurma",
    "Matsya",
    "Mayura",
    "Mrigasirsha",
    "Mukulam",
    "Mushti",
    "Nagabandha",
    "Padmakosha",
    "Pasha",
    "Pathaka",
    "Pushpaputa",
    "Sakata",
    "Samputa",
    "Sarpasirsha",
    "Shanka",
    "Shivalinga",
    "Shukatundam",
    "Sikharam",
    "Simhamukham",
    "Suchi",
    "Swastikam",
    "Tamarachudam",
    "Tripathaka",
    "Trishulam",
    "Varaha",
]

DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "mudra_mobilenetv2_final.keras"
DEFAULT_CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"
DEFAULT_HAND_LANDMARKER_PATH = PROJECT_DIR / "hand_landmarker.task"
IMAGE_SIZE = (224, 224)
SAMYUKTHA_HASTHA_MUDRAS = {
    "Anjali",
    "Berunda",
    "Chakra",
    "Garuda",
    "Kapotham",
    "Karkatta",
    "Kartariswastika",
    "Katakavardhana",
    "Khatva",
    "Kilaka",
    "Kurma",
    "Matsya",
    "Nagabandha",
    "Pasha",
    "Pushpaputa",
    "Sakata",
    "Samputa",
    "Shanka",
    "Shivalinga",
    "Swastikam",
    "Varaha",
}


def infer_hasta_category(label: str, detected_hands: Optional[int] = None) -> str:
    if label in SAMYUKTHA_HASTHA_MUDRAS:
        return "Samyuktha Hastha"
    if detected_hands is not None and detected_hands >= 2:
        return "Samyuktha Hastha"
    return "Asamyuktha Hastha"


def _strip_quantization_config(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "quantization_config":
                continue
            cleaned[key] = _strip_quantization_config(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_quantization_config(item) for item in value]
    return value


def _load_model_with_legacy_quantization_fix(model_path: Path):
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
        temp_model_path = Path(tmp_file.name)
    try:
        with zipfile.ZipFile(model_path, "r") as src_zip, zipfile.ZipFile(temp_model_path, "w") as dst_zip:
            for info in src_zip.infolist():
                data = src_zip.read(info.filename)
                if info.filename == "config.json":
                    config = json.loads(data.decode("utf-8"))
                    config = _strip_quantization_config(config)
                    data = json.dumps(config).encode("utf-8")
                dst_zip.writestr(info, data)
        return tf.keras.models.load_model(temp_model_path, compile=False)
    finally:
        if temp_model_path.exists():
            temp_model_path.unlink()


@dataclass
class PredictionResult:
    label: str
    hasta_category: str
    confidence: float
    scores: np.ndarray
    class_names: list[str]

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        indices = np.argsort(self.scores)[::-1][:k]
        return [(self.class_names[int(index)], float(self.scores[int(index)])) for index in indices]


class MudraRecognizer:
    def __init__(self, model_path: Path, class_names_path: Optional[Path] = None) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
        except TypeError as error:
            if "quantization_config" not in str(error):
                raise
            self.model = _load_model_with_legacy_quantization_fix(model_path)
        self.input_shape = self.model.input_shape
        self.class_names = self._load_class_names(model_path, class_names_path)

        if isinstance(self.input_shape, list):
            self.input_shape = self.input_shape[0]

        self.expected_channels = self._get_expected_channels()
        output_shape = self.model.output_shape
        class_count = output_shape[-1] if isinstance(output_shape, tuple) else output_shape[0][-1]

        if class_count != len(self.class_names):
            raise ValueError(
                f"Model outputs {class_count} classes, but {len(self.class_names)} class names were provided."
            )

    def _load_class_names(self, model_path: Path, class_names_path: Optional[Path]) -> list[str]:
        candidate_paths = []
        if class_names_path is not None:
            candidate_paths.append(class_names_path)
        candidate_paths.append(model_path.with_name("class_names.json"))
        candidate_paths.append(DEFAULT_CLASS_NAMES_PATH)

        for path in candidate_paths:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
                    raise ValueError(f"Invalid class names file: {path}")
                return data

        return FALLBACK_CLASS_NAMES

    def _get_expected_channels(self) -> int:
        if len(self.input_shape) != 4:
            raise ValueError(f"Unexpected model input shape: {self.input_shape}")

        channels = self.input_shape[-1]
        if channels not in (1, 3):
            raise ValueError(f"Expected 1 or 3 channels, got {channels}")

        return channels

    def preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        model_input = resized.astype(np.float32)

        if self.expected_channels == 1:
            batch = model_input[np.newaxis, ..., np.newaxis]
        else:
            rgb_like = np.stack([model_input] * 3, axis=-1)
            batch = rgb_like[np.newaxis, ...]

        return batch, resized

    def predict(self, image_bgr: np.ndarray, detected_hands: Optional[int] = None) -> tuple[PredictionResult, np.ndarray]:
        batch, processed_preview = self.preprocess(image_bgr)
        scores = self.model.predict(batch, verbose=0)[0]
        best_index = int(np.argmax(scores))
        confidence = float(scores[best_index])
        label = self.class_names[best_index]
        return PredictionResult(
            label=label,
            hasta_category=infer_hasta_category(label, detected_hands),
            confidence=confidence,
            scores=scores,
            class_names=self.class_names,
        ), processed_preview


class HandCropper:
    def __init__(
        self,
        model_path: Path = DEFAULT_HAND_LANDMARKER_PATH,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        padding: float = 0.10,
        vertical_padding_boost: float = 0.10,
        horizontal_padding_boost: float = 0.08,
        min_box_fraction: float = 0.20,
        smoothing: float = 0.75,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Hand detector model not found: {model_path}")

        self.padding = padding
        self.vertical_padding_boost = vertical_padding_boost
        self.horizontal_padding_boost = horizontal_padding_boost
        self.min_box_fraction = min_box_fraction
        self.smoothing = smoothing
        self.previous_box: Optional[tuple[int, int, int, int]] = None
        running_mode = VisionTaskRunningMode.IMAGE if static_image_mode else VisionTaskRunningMode.VIDEO
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = HandLandmarker.create_from_options(options)
        self.connections = HandLandmarksConnections.HAND_CONNECTIONS
        self.running_mode = running_mode

    def _draw_landmarks(self, frame: np.ndarray, landmarks) -> None:
        height, width = frame.shape[:2]

        for connection in self.connections:
            start = landmarks[connection.start]
            end = landmarks[connection.end]
            start_point = (int(start.x * width), int(start.y * height))
            end_point = (int(end.x * width), int(end.y * height))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

        for landmark in landmarks:
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(frame, point, 3, (0, 255, 255), -1)

    def crop_hand(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]], np.ndarray, int]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self.running_mode == VisionTaskRunningMode.VIDEO:
            timestamp_ms = int(time.time() * 1000)
            result = self.hands.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self.hands.detect(mp_image)
        annotated = frame_bgr.copy()

        if not result.hand_landmarks:
            self.previous_box = None
            return None, None, annotated, 0
        hand_count = len(result.hand_landmarks)

        height, width = frame_bgr.shape[:2]
        all_xs = []
        all_ys = []

        for hand_landmarks in result.hand_landmarks:
            self._draw_landmarks(annotated, hand_landmarks)
            all_xs.extend(landmark.x for landmark in hand_landmarks)
            all_ys.extend(landmark.y for landmark in hand_landmarks)

        x_min = max(0, int(min(all_xs) * width))
        y_min = max(0, int(min(all_ys) * height))
        x_max = min(width, int(max(all_xs) * width))
        y_max = min(height, int(max(all_ys) * height))

        box_width = x_max - x_min
        box_height = y_max - y_min

        # Give extra room around fingertip-heavy poses and two-hand signs.
        min_box_width = int(width * self.min_box_fraction)
        min_box_height = int(height * self.min_box_fraction)
        if box_width < min_box_width:
            extra_width = (min_box_width - box_width) // 2
            x_min = max(0, x_min - extra_width)
            x_max = min(width, x_max + extra_width)
        if box_height < min_box_height:
            extra_height = (min_box_height - box_height) // 2
            y_min = max(0, y_min - extra_height)
            y_max = min(height, y_max + extra_height)

        box_width = x_max - x_min
        box_height = y_max - y_min
        pad_x = int(box_width * (self.padding + self.horizontal_padding_boost))
        pad_y = int(box_height * (self.padding + self.vertical_padding_boost))

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(width, x_max + pad_x)
        y2 = min(height, y_max + pad_y)

        current_box = (x1, y1, x2, y2)
        if self.previous_box is not None:
            px1, py1, px2, py2 = self.previous_box
            x1 = int(px1 * self.smoothing + x1 * (1.0 - self.smoothing))
            y1 = int(py1 * self.smoothing + y1 * (1.0 - self.smoothing))
            x2 = int(px2 * self.smoothing + x2 * (1.0 - self.smoothing))
            y2 = int(py2 * self.smoothing + y2 * (1.0 - self.smoothing))
            current_box = (x1, y1, x2, y2)
        self.previous_box = current_box

        if x2 <= x1 or y2 <= y1:
            return None, None, annotated, hand_count

        crop = frame_bgr[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2), annotated, hand_count

    def close(self) -> None:
        self.hands.close()


def draw_prediction(
    frame: np.ndarray,
    prediction: Optional[PredictionResult],
    box: Optional[tuple[int, int, int, int]],
    hand_crop: Optional[np.ndarray],
    model_input_preview: Optional[np.ndarray],
) -> np.ndarray:
    display = frame.copy()
    display_height, display_width = display.shape[:2]

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if prediction is not None:
        text = f"{prediction.label} ({prediction.confidence * 100:.1f}%)"
        cv2.putText(
            display,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"Type: {prediction.hasta_category}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 215, 0),
            2,
            cv2.LINE_AA,
        )
        for i, (label, score) in enumerate(prediction.top_k(3), start=1):
            cv2.putText(
                display,
                f"{i}. {label}: {score * 100:.1f}%",
                (20, 72 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(
            display,
            "Show one or two hands to the camera",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    if hand_crop is not None and hand_crop.size > 0:
        preview_size = min(160, max(80, min(display_width, display_height) // 4))
        preview = cv2.resize(hand_crop, (preview_size, preview_size), interpolation=cv2.INTER_AREA)
        h, w = preview.shape[:2]
        x_start = max(0, display_width - w - 20)
        x_end = min(display_width, x_start + w)
        y_start = 20
        y_end = min(display_height, y_start + h)
        preview = preview[: y_end - y_start, : x_end - x_start]
        display[y_start:y_end, x_start:x_end] = preview
        cv2.rectangle(
            display,
            (x_start, y_start),
            (x_end, y_end),
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            "Cropped hand",
            (x_start, min(display_height - 10, y_end + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if model_input_preview is not None and model_input_preview.size > 0:
        preview_size = min(160, max(80, min(display_width, display_height) // 4))
        input_preview = cv2.resize(model_input_preview, (preview_size, preview_size), interpolation=cv2.INTER_AREA)
        input_preview = cv2.cvtColor(input_preview, cv2.COLOR_GRAY2BGR)
        x_start = max(0, display_width - preview_size - 20)
        y_start = min(display_height - preview_size - 40, 220)
        y_start = max(20, y_start)
        y_end = min(display_height, y_start + preview_size)
        x_end = min(display_width, x_start + preview_size)
        input_preview = input_preview[: y_end - y_start, : x_end - x_start]
        display[y_start:y_end, x_start:x_end] = input_preview
        cv2.rectangle(display, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)
        cv2.putText(
            display,
            "CNN input",
            (x_start, min(display_height - 10, y_end + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        display,
        "Press Q to quit",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live mudra recognition from webcam.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained Keras .h5 model.")
    parser.add_argument("--class-names", type=Path, default=None, help="Optional path to class_names.json.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index. Use 0 for default camera.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only show predictions at or above this confidence threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recognizer = MudraRecognizer(args.model, args.class_names)
    cropper = HandCropper()
    recent_scores: deque[np.ndarray] = deque(maxlen=5)

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera index.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Failed to read frame from webcam.")

            frame = cv2.flip(frame, 1)
            hand_crop, box, annotated, hand_count = cropper.crop_hand(frame)

            prediction = None
            model_input_preview = None
            if hand_crop is not None:
                current_prediction, model_input_preview = recognizer.predict(hand_crop, detected_hands=hand_count)
                recent_scores.append(current_prediction.scores)
                averaged_scores = np.mean(np.stack(recent_scores, axis=0), axis=0)
                best_index = int(np.argmax(averaged_scores))
                averaged_label = recognizer.class_names[best_index]
                current_prediction = PredictionResult(
                    label=averaged_label,
                    hasta_category=infer_hasta_category(averaged_label, hand_count),
                    confidence=float(averaged_scores[best_index]),
                    scores=averaged_scores,
                    class_names=recognizer.class_names,
                )
                if current_prediction.confidence >= args.min_confidence:
                    prediction = current_prediction
            else:
                recent_scores.clear()

            output = draw_prediction(annotated, prediction, box, hand_crop, model_input_preview)
            cv2.imshow("Mudra Recognition", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cropper.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
