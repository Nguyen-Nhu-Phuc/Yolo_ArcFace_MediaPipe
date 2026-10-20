"""Augment ảnh mặt (offline gallery) + crop/zoom mặt từ box người (runtime)."""

from __future__ import annotations

import cv2
import numpy as np

# Cạnh dài tối thiểu trước khi đưa vào ArcFace (video xa)
FACE_MIN_SIDE = 224

FAR_DOWNSCALE = (0.35, 0.5)
JPEG_QUALITIES = (55, 70)
AUG_WEIGHT_MULT = 0.6


def upscale_min_side(img_bgr: np.ndarray, min_side: int = FACE_MIN_SIDE) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    if side >= min_side:
        return img_bgr
    scale = min_side / float(side)
    return cv2.resize(
        img_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def augment_far_variants(img_bgr: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Biến thể giả lập mặt xa / nén / mờ — trả (ảnh, hệ số trọng số)."""
    if img_bgr is None or img_bgr.size == 0:
        return []
    out: list[tuple[np.ndarray, float]] = []
    h, w = img_bgr.shape[:2]
    for scale in FAR_DOWNSCALE:
        nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
        small = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        out.append((back, AUG_WEIGHT_MULT))
    for q in JPEG_QUALITIES:
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if dec is not None:
                out.append((dec, AUG_WEIGHT_MULT * 0.95))
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 0.85)
    out.append((blurred, AUG_WEIGHT_MULT * 0.9))
    return out


def crop_bbox_with_margin(
    img: np.ndarray, x1: float, y1: float, x2: float, y2: float, margin: float = 0.22
) -> np.ndarray | None:
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None
    mx, my = bw * margin, bh * margin
    ix1 = max(0, int(x1 - mx))
    iy1 = max(0, int(y1 - my))
    ix2 = min(w, int(x2 + mx))
    iy2 = min(h, int(y2 + my))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return img[iy1:iy2, ix1:ix2].copy()


def face_rois_from_person(person_bgr: np.ndarray, face_app) -> list[np.ndarray]:
    """
    Detect mặt trên crop người → zoom ROI; fallback vùng đầu nếu không detect được.
    """
    if person_bgr is None or person_bgr.size == 0:
        return []
    h, w = person_bgr.shape[:2]
    rois: list[np.ndarray] = []

    faces = face_app.get(person_bgr) or []
    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    for face in faces[:2]:
        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        crop = crop_bbox_with_margin(person_bgr, x1, y1, x2, y2)
        if crop is not None and crop.size > 0:
            rois.append(upscale_min_side(crop))

    if not rois:
        head_rois = [
            person_bgr[0 : max(int(h * 0.48), 16), int(w * 0.06) : int(w * 0.94)],
            person_bgr[0 : max(int(h * 0.62), 20), :],
        ]
        for roi in head_rois:
            if roi is None or roi.size == 0:
                continue
            rois.append(upscale_min_side(roi))

    return rois
