"""Vẽ box + nhãn (tên HS, điểm tập trung, lý do) lên khung hình."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from behavior import REASON_SHORT

# cv2.putText chỉ vẽ được ASCII → nhãn tiếng Việt có dấu vẽ bằng Pillow
_FONT_CANDIDATES = (
    "arial.ttf",
    "segoeui.ttf",
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
)

COLOR_FOCUSED = (80, 200, 40)
COLOR_UNFOCUSED = (60, 60, 235)
COLOR_NO_POSE = (200, 160, 70)


def main_reason(reasons: list) -> str:
    """Tên đầy đủ của hành vi chính, bỏ phần số giây: 'Cúi đầu lâu (3.2s)' → 'Cúi đầu lâu'."""
    for reason in reasons:
        for name, _short in REASON_SHORT:
            if name in reason:
                return name
    return ""


def _label_for(det: dict) -> str:
    who = det.get("student_id") if det.get("known") else f"ID{det['track_id']}"
    score = det.get("score")
    if score is None:
        return who
    label = f"{who} {score}"
    tag = main_reason(det.get("reasons") or [])
    if tag:
        label = f"{label} · {tag}"
    return label


@lru_cache(maxsize=8)
def _get_font(size: int):
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_annotations(frame, detections: list, show_labels: bool = True) -> None:
    """Vẽ tại chỗ. Xanh = tập trung, đỏ = mất tập trung, vàng nhạt = chưa có pose."""
    labels = []
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        if det.get("score") is None:
            color = COLOR_NO_POSE
        elif det.get("is_focused", True):
            color = COLOR_FOCUSED
        else:
            color = COLOR_UNFOCUSED
        reasons = det.get("reasons") or []
        thickness = 4 if any("Cúi đầu lâu" in r for r in reasons) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if show_labels:
            labels.append((_label_for(det), x1, y1, color))
    if labels:
        _draw_labels(frame, labels)


@lru_cache(maxsize=1024)
def _label_patch(label: str, color: tuple, size: int) -> np.ndarray:
    """Pillow vẽ chữ chậm (~7 ms/nhãn) nên cache patch theo nội dung nhãn."""
    font = _get_font(size)
    left, top, right, bottom = font.getbbox(label)
    img = Image.new("RGB", (right - left + 8, bottom - top + 6), (16, 12, 10))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 3, img.height], fill=color)
    draw.text((6 - left, 3 - top), label, font=font, fill=(255, 255, 255))
    return np.asarray(img)


def _draw_labels(frame, labels: list) -> None:
    size = max(13, frame.shape[0] // 55)
    frame_h, frame_w = frame.shape[:2]
    for label, x1, y1, color in labels:
        patch = _label_patch(label, color, size)
        h, w = patch.shape[:2]
        x, y = x1, y1 - h - 2
        if y < 0:
            y = y1 + 2
        px0, py0 = max(0, x), max(0, y)
        px1, py1 = min(frame_w, x + w), min(frame_h, y + h)
        if px0 < px1 and py0 < py1:
            frame[py0:py1, px0:px1] = patch[py0 - y:py1 - y, px0 - x:px1 - x]
