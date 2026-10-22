"""Ghi keypoint + tín hiệu mặt của từng lần phân tích ra JSONL để chạy lại offline.

Dòng đầu: header (nguồn, fps, kích thước khung, ngưỡng hành vi lúc ghi).
Mỗi dòng sau: {"idx", "t" (giây trong video), "fps_eff", "tracks": [{tid, sid, key, box, kpts, signals}]}.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from config import BehaviorConfig

LOG_DIR = Path(__file__).resolve().parent / "logs" / "keypoints"
FLUSH_EVERY = 20


def behavior_to_dict(bh: BehaviorConfig) -> dict:
    data = asdict(bh)
    data["disabled"] = sorted(bh.disabled)
    return data


class KeypointLogger:
    def __init__(self, path: Path, source: str, fps: float, width: int, height: int,
                 behavior: BehaviorConfig):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._f = self.path.open("w", encoding="utf-8")
        self._n = 0
        self._write({
            "type": "header",
            "version": 1,
            "source": source,
            "fps": fps,
            "width": width,
            "height": height,
            "started": datetime.now().isoformat(timespec="seconds"),
            "behavior": behavior_to_dict(behavior),
        })

    def _write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def log(self, frame_idx: int, fps_eff: float, detections: list[dict]) -> None:
        if self._f.closed:
            return
        tracks = [
            {
                "tid": d["track_id"],
                "sid": d.get("student_id"),
                "key": d["key"],
                "box": [d["x1"], d["y1"], d["x2"], d["y2"]],
                "kpts": [[round(float(v), 1 if i < 2 else 3) for i, v in enumerate(p)]
                         for p in d["keypoints"]],
                "signals": d.get("signals") or {},
            }
            for d in detections
            if d.get("keypoints") is not None and d.get("key")
        ]
        self._write({
            "idx": frame_idx,
            "t": round(frame_idx / self.fps, 3),
            "fps_eff": round(fps_eff, 4),
            "tracks": tracks,
        })
        self._n += 1
        if self._n % FLUSH_EVERY == 0:
            self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


def read_log(path: Path) -> tuple[dict, list[dict]]:
    """→ (header, các dòng mẫu)."""
    header: dict = {}
    samples: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "header":
                header = row
            else:
                samples.append(row)
    return header, samples
