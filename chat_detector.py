"""Phát hiện 2 học sinh ngồi cạnh quay vào nhau (THỬ NGHIỆM, mặc định tắt).

Cặp bị gắn cờ khi box sát nhau và góc vai lệch nhau nhiều.
"""

from __future__ import annotations

import math


def _compute_shoulder_angle(keypoints):
    if keypoints is None or len(keypoints) < 17:
        return None
    left = keypoints[5]
    right = keypoints[6]
    if left[2] < 0.5 or right[2] < 0.5:
        return None
    return math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))


def _horizontal_gap(det_a, det_b) -> float:
    """Khoảng hở ngang giữa 2 box, chuẩn hóa theo chiều cao box trung bình."""
    a_center = (det_a["x1"] + det_a["x2"]) / 2
    b_center = (det_b["x1"] + det_b["x2"]) / 2
    if a_center <= b_center:
        gap = det_b["x1"] - det_a["x2"]
    else:
        gap = det_a["x1"] - det_b["x2"]
    avg_height = ((det_a["y2"] - det_a["y1"]) + (det_b["y2"] - det_b["y1"])) / 2
    if avg_height <= 0:
        return float("inf")
    return gap / avg_height


def detect_chat(detections, config) -> list[tuple]:
    ch = config.behavior
    if not ch.is_on("Nói chuyện") or len(detections) < 2:
        return []

    pairs = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            det_a, det_b = detections[i], detections[j]
            if _horizontal_gap(det_a, det_b) > ch.chat_distance_threshold:
                continue
            angle_a = _compute_shoulder_angle(det_a.get("keypoints"))
            angle_b = _compute_shoulder_angle(det_b.get("keypoints"))
            if angle_a is None or angle_b is None:
                continue
            diff = abs(angle_a - angle_b)
            if diff > 180:
                diff = 360 - diff
            if diff >= ch.chat_angle_threshold:
                pairs.append((det_a["track_id"], det_b["track_id"]))
    return pairs
