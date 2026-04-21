import argparse
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from original import (
    DEFAULT_CLASS_NAMES_PATH,
    DEFAULT_HAND_LANDMARKER_PATH,
    DEFAULT_MODEL_PATH,
    HandCropper,
    MudraRecognizer,
    PredictionResult,
    infer_hasta_category,
)


def segment_skin_ycrcb(image_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper)


def segment_skin_hsv(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_1 = np.array([0, 25, 40], dtype=np.uint8)
    upper_1 = np.array([25, 180, 255], dtype=np.uint8)
    lower_2 = np.array([160, 25, 40], dtype=np.uint8)
    upper_2 = np.array([180, 180, 255], dtype=np.uint8)
    mask_1 = cv2.inRange(hsv, lower_1, upper_1)
    mask_2 = cv2.inRange(hsv, lower_2, upper_2)
    return cv2.bitwise_or(mask_1, mask_2)


def refine_mask(mask: np.ndarray) -> np.ndarray:
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)
    return mask


def crop_to_mask(image_bgr: np.ndarray, mask: np.ndarray, padding: float = 0.18) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.findNonZero(mask)
    if points is None:
        return image_bgr, mask

    x, y, w, h = cv2.boundingRect(points)
    image_h, image_w = image_bgr.shape[:2]
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_w, x + w + pad_x)
    y2 = min(image_h, y + h + pad_y)
    return image_bgr[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def segment_hand(image_bgr: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    raw_mask = segment_skin_hsv(image_bgr) if method == "hsv" else segment_skin_ycrcb(image_bgr)
    mask = refine_mask(raw_mask)

    gray_background = np.full_like(image_bgr, 127)
    segmented = gray_background.copy()
    segmented[mask > 0] = image_bgr[mask > 0]

    cropped_segmented, cropped_mask = crop_to_mask(segmented, mask)
    return cropped_segmented, cropped_mask


def resize_for_panel(image: np.ndarray, width: int, height: int, grayscale: bool = False) -> np.ndarray:
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    canvas = np.full((height, width, 3), 30, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    new_w = max(1, int(image.shape[1] * scale))
    new_h = max(1, int(image.shape[0] * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def compose_display(
    original_frame: np.ndarray,
    prediction: Optional[PredictionResult],
    segmented_preview: Optional[np.ndarray],
    model_preview: Optional[np.ndarray],
    method: str,
) -> np.ndarray:
    frame_h, frame_w = original_frame.shape[:2]
    side_width = max(260, frame_w // 3)
    panel_height = frame_h // 2

    canvas = np.full((frame_h, frame_w + side_width, 3), 25, dtype=np.uint8)
    canvas[:, :frame_w] = original_frame

    cv2.putText(
        canvas,
        "Original",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if segmented_preview is not None and segmented_preview.size > 0:
        segmented_panel = resize_for_panel(segmented_preview, side_width, panel_height)
        canvas[:panel_height, frame_w:] = segmented_panel
        cv2.putText(
            canvas,
            f"Skin Segmentation ({method.upper()})",
            (frame_w + 12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "Skin Segmentation",
            (frame_w + 12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Waiting for hand",
            (frame_w + 20, panel_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )

    if model_preview is not None and model_preview.size > 0:
        model_panel = resize_for_panel(model_preview, side_width, frame_h - panel_height, grayscale=True)
        canvas[panel_height:, frame_w:] = model_panel
        cv2.putText(
            canvas,
            "Model Input",
            (frame_w + 12, panel_height + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "Model Input",
            (frame_w + 12, panel_height + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if prediction is not None:
        cv2.putText(
            canvas,
            f"{prediction.label} ({prediction.confidence * 100:.1f}%)",
            (20, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Type: {prediction.hasta_category}",
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 215, 0),
            2,
            cv2.LINE_AA,
        )
        for i, (label, score) in enumerate(prediction.top_k(3), start=1):
            cv2.putText(
                canvas,
                f"{i}. {label}: {score * 100:.1f}%",
                (20, 92 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(
            canvas,
            "Show one or two hands to the camera",
            (20, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Press Q to quit",
        (20, frame_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.line(canvas, (frame_w, 0), (frame_w, frame_h), (90, 90, 90), 2)
    cv2.line(canvas, (frame_w, panel_height), (frame_w + side_width, panel_height), (90, 90, 90), 2)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mudra recognition with skin segmentation and gray background.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained Keras model.")
    parser.add_argument("--class-names", type=Path, default=None, help="Optional path to class_names.json.")
    parser.add_argument("--hand-landmarker", type=Path, default=DEFAULT_HAND_LANDMARKER_PATH, help="Path to hand_landmarker.task.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Prediction confidence threshold.")
    parser.add_argument("--method", choices=["hsv", "ycrcb"], default="ycrcb", help="Skin segmentation method.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recognizer = MudraRecognizer(args.model, args.class_names)
    cropper = HandCropper(
        model_path=args.hand_landmarker,
        padding=0.18,
        vertical_padding_boost=0.24,
        horizontal_padding_boost=0.12,
        min_box_fraction=0.28,
        max_num_hands=2,
    )
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

            segmented_preview = None
            model_input_preview = None
            prediction = None

            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if hand_crop is not None and hand_crop.size > 0:
                segmented_preview, _ = segment_hand(hand_crop, args.method)
                current_prediction, model_input_preview = recognizer.predict(
                    segmented_preview,
                    detected_hands=hand_count,
                )
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

            output = compose_display(
                annotated,
                prediction,
                segmented_preview,
                model_input_preview,
                args.method,
            )
            cv2.imshow("Mudra Skin Segmentation", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cropper.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
