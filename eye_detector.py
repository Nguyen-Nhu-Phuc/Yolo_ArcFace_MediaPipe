"""Eye Aspect Ratio (EAR) bằng MediaPipe Face Landmarker trên crop mặt.

EAR gần 0 = mắt nhắm; mắt mở thường > 0.2.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FACE_MODEL = Path(__file__).resolve().parent / "models" / "mediapipe" / "face_landmarker.task"

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [13, 14, 78, 308]  # môi trên, môi dưới, khóe trái, khóe phải


def calculate_ear_from_landmarks(eye_points: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)."""
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    if h < 1e-6:
        return 1.0
    return float((v1 + v2) / (2.0 * h))


class EyeDetector:
    """Không thread-safe — mỗi luồng phân tích giữ 1 instance riêng."""

    def __init__(self):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as base

        if not FACE_MODEL.exists():
            raise FileNotFoundError(f"Thiếu model MediaPipe: {FACE_MODEL}")
        options = vision.FaceLandmarkerOptions(
            base_options=base.BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def measure(self, face_image) -> dict | None:
        """{"ear": độ mở mắt, "mar": độ mở miệng (khoảng cách môi / bề rộng miệng)}."""
        import mediapipe as mp

        if face_image is None or face_image.size == 0:
            return None
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w = face_image.shape[:2]

        def points(indices):
            return np.array(
                [[landmarks[i].x * w, landmarks[i].y * h] for i in indices],
                dtype=np.float32,
            )

        left = calculate_ear_from_landmarks(points(LEFT_EYE_INDICES))
        right = calculate_ear_from_landmarks(points(RIGHT_EYE_INDICES))
        upper, lower, corner_l, corner_r = points(MOUTH_INDICES)
        width = float(np.linalg.norm(corner_l - corner_r))
        mar = float(np.linalg.norm(upper - lower)) / width if width > 1e-6 else None
        return {"ear": (left + right) / 2.0, "mar": mar}

    def get_ear(self, face_image) -> float | None:
        m = self.measure(face_image)
        return m["ear"] if m else None

    def close(self) -> None:
        try:
            self.face_landmarker.close()
        except Exception:
            pass
