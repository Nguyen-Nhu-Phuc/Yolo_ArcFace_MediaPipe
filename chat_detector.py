"""Phát hiện 2 học sinh ngồi cạnh quay mặt vào nhau (THỬ NGHIỆM, mặc định tắt).

Cặp bị gắn cờ khi ngồi sát nhau cùng hàng, em bên trái quay sang phải và em bên phải
quay sang trái, giữ đủ lâu. Thấy được miệng mà miệng không động thì không tính.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from behavior import Debounce

MAR_HISTORY = 8
MAR_MIN_SAMPLES = 3


def _horizontal_gap(det_a, det_b) -> float:
    """Khoảng hở ngang giữa 2 box (a bên trái), chuẩn hóa theo chiều cao box trung bình."""
    avg_height = ((det_a["y2"] - det_a["y1"]) + (det_b["y2"] - det_b["y1"])) / 2
    if avg_height <= 0:
        return float("inf")
    return (det_b["x1"] - det_a["x2"]) / avg_height


def _row_offset(det_a, det_b) -> float:
    avg_height = ((det_a["y2"] - det_a["y1"]) + (det_b["y2"] - det_b["y1"])) / 2
    if avg_height <= 0:
        return float("inf")
    return abs((det_a["y1"] + det_a["y2"]) - (det_b["y1"] + det_b["y2"])) / 2 / avg_height


class ChatTracker:
    """Giữ thời gian từng cặp quay vào nhau qua nhiều lần phân tích (1 instance / phiên)."""

    def __init__(self):
        self._timers: dict[tuple[int, int], Debounce] = {}
        self._mar: dict[int, deque] = {}

    def _mouth_still(self, tid: int, bh) -> bool | None:
        hist = self._mar.get(tid)
        if hist is None or len(hist) < MAR_MIN_SAMPLES:
            return None
        return float(np.std(hist)) < bh.chat_mouth_std

    def update(self, detections: list[dict], config, dt: float) -> list[tuple[int, int]]:
        """detections cần x1..y2, track_id, mar (có thể None) và yaw: độ quay so với hướng
        nhìn thường ngày của HS (StudentStateTracker.last_turn), dương = sang phải ảnh."""
        bh = config.behavior
        present = {d["track_id"] for d in detections}
        for tid in [t for t in self._mar if t not in present]:
            del self._mar[tid]
        for d in detections:
            if d.get("mar") is not None:
                self._mar.setdefault(d["track_id"], deque(maxlen=MAR_HISTORY)).append(d["mar"])
        if not bh.is_on("Nói chuyện"):
            self._timers.clear()
            return []

        ordered = sorted(detections, key=lambda d: d["x1"] + d["x2"])
        facing: set[tuple[int, int]] = set()
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                if (_horizontal_gap(left, right) > bh.chat_distance_threshold
                        or _row_offset(left, right) > bh.chat_row_tolerance):
                    continue
                ya, yb = left.get("yaw"), right.get("yaw")
                if ya is None or yb is None:
                    continue
                if ya < bh.chat_yaw_threshold or yb > -bh.chat_yaw_threshold:
                    continue
                still = [s for s in (self._mouth_still(left["track_id"], bh),
                                     self._mouth_still(right["track_id"], bh)) if s is not None]
                if still and all(still):
                    continue
                facing.add((left["track_id"], right["track_id"]))

        pairs = []
        for key in set(self._timers) | facing:
            t = self._timers.setdefault(key, Debounce()).step(key in facing, dt, bh.miss_tolerance_sec)
            if t <= 0:
                del self._timers[key]
            elif t >= bh.chat_duration:
                pairs.append(key)
        return pairs
