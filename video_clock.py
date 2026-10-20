"""Ngày giờ thật của từng khung: đọc dòng ngày giờ camera in trên hình (OSD) bằng EasyOCR.

Lần đầu quét toàn khung để tìm vị trí dòng ngày giờ; sau đó chỉ đọc lại vùng đó
định kỳ để chỉnh mốc (video ghi đứt quãng vẫn đúng giờ).
Không đọc được → camera dùng đồng hồ máy, file dùng vị trí trong video.
"""

from __future__ import annotations

import bisect
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import numpy as np

# Ngày dạng 10-07-2026 không phân biệt được ngày/tháng → thứ tự mặc định
DEFAULT_DATE_ORDER = (os.environ.get("OSD_DATE_ORDER") or "DMY").strip().upper()
RESYNC_SEC = 60.0
# Lệch quá mức này so với dự đoán thì phải đọc lại khớp 2 lần mới đổi mốc
MAX_DRIFT_SEC = 3.0

_reader = None
_reader_lock = threading.Lock()


def _get_reader():
    global _reader
    with _reader_lock:
        if _reader is None:
            import easyocr

            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return _reader


def preload_ocr() -> None:
    _get_reader()


@dataclass
class OsdReading:
    dt: datetime
    date_fmt: str  # strftime để hiển thị ngày đúng như trên video
    box: tuple[int, int, int, int] | None = None  # vùng chữ trên khung (x1,y1,x2,y2)


def parse_osd_text(text: str) -> OsdReading | None:
    """'10-07-2026 09:41:20' / '2026/07/10 09:41:20' → OsdReading."""
    digits = re.sub(r"\D", "", text)
    if len(digits) != 14:
        return None
    sep_m = re.search(r"\d([-/.])\d", text)
    sep = sep_m.group(1) if sep_m else "-"
    try:
        if digits[:2] in ("19", "20") and int(digits[4:6]) <= 12:
            year, a, b = int(digits[:4]), int(digits[4:6]), int(digits[6:8])
            month, day = a, b
            date_fmt = f"%Y{sep}%m{sep}%d"
        else:
            a, b, year = int(digits[:2]), int(digits[2:4]), int(digits[4:8])
            if a > 12:
                order = "DMY"
            elif b > 12:
                order = "MDY"
            else:
                order = DEFAULT_DATE_ORDER
            if order == "MDY":
                month, day = a, b
                date_fmt = f"%m{sep}%d{sep}%Y"
            else:
                day, month = a, b
                date_fmt = f"%d{sep}%m{sep}%Y"
        hh, mm, ss = int(digits[8:10]), int(digits[10:12]), int(digits[12:14])
        return OsdReading(datetime(year, month, day, hh, mm, ss), date_fmt)
    except ValueError:
        return None


def _lines(results) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Gom các mảnh chữ EasyOCR cùng hàng thành 1 dòng (ngày và giờ hay bị tách)."""
    items = []
    for box, text, _conf in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append((min(xs), min(ys), max(xs), max(ys), text))
    items.sort(key=lambda t: (0.5 * (t[1] + t[3]), t[0]))
    lines: list[list] = []
    for it in items:
        cy, h = 0.5 * (it[1] + it[3]), it[3] - it[1]
        if lines:
            last = lines[-1]
            lcy = 0.5 * (last[0][1] + last[0][3])
            if abs(cy - lcy) <= 0.6 * max(h, 1):
                last.append(it)
                continue
        lines.append([it])
    out = []
    for ln in lines:
        ln.sort(key=lambda t: t[0])
        text = " ".join(t[4] for t in ln)
        box = (
            int(min(t[0] for t in ln)), int(min(t[1] for t in ln)),
            int(max(t[2] for t in ln)), int(max(t[3] for t in ln)),
        )
        out.append((text, box))
    return out


def read_osd(frame: np.ndarray, region: tuple[int, int, int, int] | None = None
             ) -> OsdReading | None:
    """Đọc ngày giờ trên khung; region = chỉ đọc vùng đã biết (nhanh)."""
    reader = _get_reader()
    ox = oy = 0
    img = frame
    scale = 1.0
    if region is not None:
        x1, y1, x2, y2 = region
        img = frame[y1:y2, x1:x2]
        ox, oy = x1, y1
        if img.size == 0:
            return None
        scale = 2.0
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(img, allowlist="0123456789-:/. ")
    for text, (bx1, by1, bx2, by2) in _lines(results):
        reading = parse_osd_text(text)
        if reading is None:
            continue
        reading.box = (
            int(ox + bx1 / scale), int(oy + by1 / scale),
            int(ox + bx2 / scale), int(oy + by2 / scale),
        )
        return reading
    return None


class VideoClock:
    """Ánh xạ chỉ số khung → ngày giờ theo các mốc đọc được từ OSD."""

    def __init__(self, fps: float, is_camera: bool):
        self.fps = fps
        self.is_camera = is_camera
        self._lock = threading.Lock()
        self._anchor_frames: list[int] = []
        self._anchor_dts: list[datetime] = []
        self.date_fmt = "%d-%m-%Y"
        self.region: tuple[int, int, int, int] | None = None
        self._pending: tuple[int, datetime] | None = None

    @property
    def source(self) -> str:
        if self._anchor_frames:
            return "osd"
        return "system" if self.is_camera else "video"

    def _predict(self, frame_idx: int) -> datetime | None:
        if not self._anchor_frames:
            return None
        i = bisect.bisect_right(self._anchor_frames, frame_idx) - 1
        i = max(i, 0)
        return self._anchor_dts[i] + timedelta(
            seconds=(frame_idx - self._anchor_frames[i]) / self.fps
        )

    def add_reading(self, frame_idx: int, reading: OsdReading, frame_shape) -> None:
        with self._lock:
            pred = self._predict(frame_idx)
            if pred is not None and abs((reading.dt - pred).total_seconds()) > MAX_DRIFT_SEC:
                p = self._pending
                consistent = p is not None and abs(
                    (reading.dt - p[1]).total_seconds() - (frame_idx - p[0]) / self.fps
                ) <= MAX_DRIFT_SEC
                if not consistent:
                    self._pending = (frame_idx, reading.dt)
                    return
            self._pending = None
            i = bisect.bisect_right(self._anchor_frames, frame_idx)
            self._anchor_frames.insert(i, frame_idx)
            self._anchor_dts.insert(i, reading.dt)
            self.date_fmt = reading.date_fmt
            if reading.box is not None:
                h, w = frame_shape[:2]
                x1, y1, x2, y2 = reading.box
                mx, my = int(0.25 * (x2 - x1)) + 8, int(1.0 * (y2 - y1)) + 6
                self.region = (max(0, x1 - mx), max(0, y1 - my),
                               min(w, x2 + mx), min(h, y2 + my))

    def at(self, frame_idx: int, wall: float | None = None) -> datetime | None:
        """None khi chỉ có vị trí trong video (file không có OSD)."""
        with self._lock:
            dt = self._predict(frame_idx)
        if dt is not None:
            return dt
        if self.is_camera and wall is not None:
            return datetime.fromtimestamp(wall)
        return None

    def format_date(self, dt: datetime | None) -> str:
        return dt.strftime(self.date_fmt) if dt is not None else ""

    def current_label(self, frame_idx: int) -> str | None:
        dt = self.at(frame_idx)
        if dt is None:
            return None
        return f"{self.format_date(dt)} {dt:%H:%M:%S}"
