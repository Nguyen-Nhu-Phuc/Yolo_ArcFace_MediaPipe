"""Chạy phân tích offline trên 1 file video theo bước thời gian cố định → log keypoint JSONL.

Cùng pipeline với web (analysis.FrameAnalyzer) nhưng không bỏ khung theo tốc độ CPU,
ArcFace chạy đồng bộ sau mỗi bước. Dùng log này để gán nhãn + chạy eval_behavior.py.

    python tools/extract_keypoints.py --video D:\video\tiet3.mp4 --start 0 --duration 600
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))  # chạy được bằng python tools/<tên>.py

from core.analysis import FrameAnalyzer, effective_fps, fit_width, work_size
from core.config import load_config
from core.identity import IdentityTracker
from core.keypoint_log import LOG_DIR, KeypointLogger
from core.video_source import capture_fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Trích keypoint + tín hiệu mặt từ video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", help="File JSONL (mặc định logs/keypoints/<tên video>.jsonl)")
    parser.add_argument("--step", type=float, default=0.5, help="Giây giữa 2 lần phân tích")
    parser.add_argument("--start", type=float, default=0.0, help="Bắt đầu từ giây thứ N")
    parser.add_argument("--duration", type=float, default=0.0, help="Số giây phân tích (0 = hết video)")
    parser.add_argument("--no-arcface", action="store_true", help="Bỏ nhận diện tên (nhanh hơn)")
    args = parser.parse_args()

    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {video}")
    fps = capture_fps(cap, default=30.0)
    width, height = work_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    stride = max(1, round(args.step * fps))
    start_idx = int(args.start * fps)
    end_idx = start_idx + int(args.duration * fps) if args.duration > 0 else None
    if start_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    cfg = load_config()
    recognizer = None
    if not args.no_arcface:
        from core.face_gallery import FaceRecognizer

        recognizer = FaceRecognizer(force_rebuild=False)
    identity = IdentityTracker(recognizer)
    analyzer = FrameAnalyzer(cfg, identity)
    out = Path(args.out) if args.out else LOG_DIR / f"{video.stem}.jsonl"
    logger = KeypointLogger(out, video.name, fps, width, height, cfg.behavior)
    print(f"{video.name}: {fps:g} fps, phân tích mỗi {stride} khung ({stride / fps:.2f}s) → {out}")

    idx = start_idx
    n = 0
    t0 = time.perf_counter()
    fps_eff = effective_fps(fps, stride)
    try:
        while end_idx is None or idx < end_idx:
            ok, frame = cap.read()
            if not ok:
                break
            frame = fit_width(frame)
            detections, parsed = analyzer.analyze(frame, fps_eff)
            logger.log(idx, fps_eff, detections)
            if recognizer is not None:
                identity.recognize(frame, parsed)
            n += 1
            if n % 20 == 0:
                ms = (time.perf_counter() - t0) / n * 1000
                print(f"  {idx / fps:7.1f}s · {len(detections)} người · "
                      f"{identity.identified_count()} đã nhận diện · {ms:.0f} ms/bước")
            for _ in range(stride - 1):
                if not cap.grab():
                    break
            idx += stride
    finally:
        logger.close()
        analyzer.close()
        cap.release()
    print(f"Xong: {n} bước, {(idx - start_idx) / fps:.0f}s video → {out}")


if __name__ == "__main__":
    main()
