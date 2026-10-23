"""Keypoint COCO-17 cho từng người bằng YOLO11-pose chạy trên crop.

Chạy trên crop (thay vì toàn khung) để học sinh ngồi xa vẫn đủ lớn cho
model pose, và keypoint gắn thẳng vào đúng track YOLOv12.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

PAD_X = 0.08
PAD_Y = 0.05


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class PoseEstimator:
    """Không thread-safe (predictor Ultralytics) — 1 instance / luồng phân tích."""

    def __init__(self, weights: str, imgsz: int = 320, conf: float = 0.25):
        self.model = YOLO(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)

    def estimate(self, frame: np.ndarray, boxes: list[tuple[int, int, int, int]]
                 ) -> list[np.ndarray | None]:
        """boxes: [(x1,y1,x2,y2)] theo pixel khung → keypoints (17,3) theo pixel khung."""
        if not boxes:
            return []
        h, w = frame.shape[:2]
        crops, origins, targets = [], [], []
        for x1, y1, x2, y2 in boxes:
            bw, bh = x2 - x1, y2 - y1
            cx1 = max(0, int(x1 - bw * PAD_X))
            cy1 = max(0, int(y1 - bh * PAD_Y))
            cx2 = min(w, int(x2 + bw * PAD_X))
            cy2 = min(h, int(y2 + bh * PAD_Y))
            crops.append(frame[cy1:cy2, cx1:cx2])
            origins.append((cx1, cy1))
            targets.append((x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1))

        results = self.model.predict(
            crops, imgsz=self.imgsz, conf=self.conf, verbose=False
        )

        out: list[np.ndarray | None] = []
        for res, (ox, oy), target in zip(results, origins, targets):
            kp = res.keypoints
            if kp is None or kp.data is None or len(kp.data) == 0:
                out.append(None)
                continue
            data = kp.data.cpu().numpy()
            pboxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else None
            best = 0
            if pboxes is not None and len(pboxes) > 1:
                best = int(np.argmax([_iou(b, target) for b in pboxes]))
            kpts = data[best].astype(np.float32).copy()
            if kpts.shape[1] == 2:
                kpts = np.concatenate([kpts, np.ones((17, 1), np.float32)], axis=1)
            kpts[:, 0] += ox
            kpts[:, 1] += oy
            out.append(kpts)
        return out
