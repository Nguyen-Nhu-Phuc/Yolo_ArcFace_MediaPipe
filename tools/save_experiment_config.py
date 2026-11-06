"""Lưu cấu hình thực nghiệm (mô hình, ngưỡng, thư viện) để tái lập kết quả.

Chạy từ thư mục gốc dự án:
  python -m tools.save_experiment_config
  python -m tools.save_experiment_config --out evidence/config/experiment_config_stage1.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from core.config import load_config
from core.face_gallery import FaceRecognizer

ROOT = Path(__file__).resolve().parents[1]


def sha256(path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "Not installed"


def build_report() -> dict:
    cfg = load_config()
    rec = FaceRecognizer()
    return {
        "created_at": datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: version(p)
            for p in [
                "ultralytics", "torch", "opencv-python",
                "insightface", "onnxruntime", "numpy",
            ]
        },
        "yolo": {
            "path": str(cfg.yolo_model),
            "sha256": sha256(cfg.yolo_model),
            "imgsz": cfg.yolo_imgsz,
            "conf": cfg.confidence_threshold,
            "iou": cfg.iou_threshold,
        },
        "pose": {
            "path": str(cfg.pose_model),
            "sha256": sha256(cfg.pose_model),
            "imgsz": cfg.pose_imgsz,
            "conf": cfg.pose_conf,
            "max_per_frame": cfg.pose_max_per_frame,
        },
        "arcface": {
            "gallery_path": str(rec.gallery_path),
            "gallery_sha256": sha256(rec.gallery_path),
            "match_threshold": rec.match_threshold,
            "min_margin": rec.min_margin,
            "lock_sim": rec.lock_sim,
            "hard_lock_sim": rec.hard_lock_sim,
            "hard_lock_margin": rec.hard_lock_margin,
            "note": "Điểm tương đồng quy đổi (%) = cosine × 100, không phải xác suất nhận diện đúng.",
        },
        "behavior": {
            **asdict(cfg.behavior),
            "disabled": sorted(cfg.behavior.disabled),
        },
        "ernu": {
            **{k: v for k, v in asdict(cfg.ernu).items()},
            "rules_file": str(
                ROOT / "evidence" / "config" / "ernu_rules_v1.json"
            ),
            "note": (
                "E/R/N/U theo cửa sổ 10s + A1–A5; BEEI chỉ số phụ; "
                "không dùng pred làm GT của chính hệ thống."
            ),
        },
        "algorithm_notes": {
            "iou_transfer_lock": False,
            "arcface_margin_rule": "accept if lock_sim or margin >= min_margin (single-gallery exempt)",
            "missing_keypoints": "decay + 'Không xác định' when pose attempted",
            "episode_identity": "per-event student_id; no majority-vote reassignment of history",
            "ernu": "core/ernu.py; ACTIVITY_CONTEXT via ERNU_ACTIVITY_CONTEXT; rules ernu-v1",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lưu cấu hình thực nghiệm Giai đoạn 1")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "evidence" / "config" / "experiment_config_stage1.json",
    )
    args = parser.parse_args()
    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã tạo {args.out.resolve()}")


if __name__ == "__main__":
    main()
