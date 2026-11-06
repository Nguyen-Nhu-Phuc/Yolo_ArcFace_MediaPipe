"""Phân tích 1 khung: YOLOv12 track → YOLO11-pose → hướng mặt → hành vi.

Dùng chung cho web (app.py) và chạy offline (extract_keypoints.py), nên log keypoint
của 2 nơi giống hệt nhau.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from core.behavior import (
    UNKNOWN_BEHAVIOR,
    StudentStateTracker,
    detect_behaviors,
    observation_group,
)
from core.config import Config
from core.identity import IdentityTracker
from core.pose_estimator import PoseEstimator

WORK_MAX_WIDTH = 1280  # khung được thu về độ rộng này trước khi phân tích + hiển thị
# Ba loại thời gian (không trộn):
#   1) Thời gian video — chỉ số khung / FPS nguồn → mốc bắt đầu/kết thúc sự kiện trên Excel
#   2) Thời gian quan sát hợp lệ — dt = 1/fps_eff đưa vào bộ đếm hành vi (bằng chứng duy trì)
#   3) Thời gian xử lý — ms phân tích / khung (hiệu năng), không dùng cho điểm danh hay hành vi
# Phân tích chậm (CPU) → mỗi lần chỉ cộng tối đa N giây vào bộ đếm hành vi,
# để 1 lần quan sát không đủ vượt ngưỡng thời gian.
MAX_BEHAVIOR_STEP_SEC = 1.0


def fit_width(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= WORK_MAX_WIDTH:
        return frame
    s = WORK_MAX_WIDTH / float(w)
    return cv2.resize(frame, (WORK_MAX_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)


def work_size(width: int, height: int) -> tuple[int, int]:
    """Kích thước khung sau fit_width."""
    if width <= WORK_MAX_WIDTH:
        return width, height
    return WORK_MAX_WIDTH, int(height * WORK_MAX_WIDTH / width)


def effective_fps(fps: float, frames_elapsed: int) -> float:
    """FPS hiệu dụng cho thời gian quan sát hợp lệ (bộ đếm hành vi).

    Khi bỏ khung, fps/frames_elapsed giảm; sàn 1/MAX_BEHAVIOR_STEP_SEC tránh cộng dt quá lớn
    (hành vi 10s video có thể ghi nhận < 10s quan sát hợp lệ — đúng theo thiết kế thận trọng).
    """
    return max(fps / max(1, frames_elapsed), 1.0 / MAX_BEHAVIOR_STEP_SEC)


class FrameAnalyzer:
    """Không thread-safe (predictor Ultralytics) — 1 instance / luồng phân tích."""

    def __init__(self, cfg: Config, identity: IdentityTracker):
        self.cfg = cfg
        self.identity = identity
        self.yolo = YOLO(cfg.yolo_model)
        self.pose = PoseEstimator(cfg.pose_model, cfg.pose_imgsz, cfg.pose_conf)
        self.state_tracker = StudentStateTracker()

    def reset_tracks(self) -> None:
        """Phiên mới: xóa tư thế đã nhớ và ID theo dõi, model YOLO giữ nguyên."""
        self.state_tracker = StudentStateTracker()
        predictor = getattr(self.yolo, "predictor", None)
        if predictor is not None:
            predictor.trackers = []

    def _face_signals(self, tid: int) -> dict:
        cfg = self.cfg
        signals: dict = {"yaw": None, "pitch": None}
        pose = self.identity.head_pose(tid, cfg.head_pose_max_age)
        if pose is not None:
            signals["yaw"], signals["pitch"] = round(pose[0], 1), round(pose[1], 1)
        return signals

    def analyze(self, frame: np.ndarray, fps_eff: float) -> tuple[list[dict], list]:
        """→ (detections, parsed tracks cho ArcFace)."""
        cfg = self.cfg
        h, w = frame.shape[:2]
        results = self.yolo.track(
            source=frame,
            conf=cfg.confidence_threshold,
            iou=cfg.iou_threshold,
            classes=[0],
            persist=True,
            verbose=False,
            imgsz=cfg.yolo_imgsz,
            tracker="bytetrack.yaml",
        )
        raw = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            tids = boxes.id.cpu().numpy().astype(int)
            raw = [(int(t), *map(int, b)) for b, t in zip(xyxy, tids)]

        parsed = self.identity.update_tracks(raw, w, h)
        pose_targets = sorted(
            parsed, key=lambda t: (t[3] - t[1]) * (t[4] - t[2]), reverse=True
        )[: cfg.pose_max_per_frame]
        pose_tid_set = {t[0] for t in pose_targets}
        kpts_by_tid = dict(
            zip(
                [t[0] for t in pose_targets],
                self.pose.estimate(frame, [t[1:] for t in pose_targets]),
            )
        )

        detections: list[dict] = []
        dt = 1.0 / max(float(fps_eff), 1e-6)
        for tid, x1, y1, x2, y2 in parsed:
            info = self.identity.describe(tid)
            kpts = kpts_by_tid.get(tid)
            # Nghi ngờ không phải người (OSD…) và không được đưa vào pose → bỏ qua hiển thị
            if kpts is None and info["no_face_suspect"] and tid not in pose_tid_set:
                continue
            key = info["student_id"] or f"t{tid}"
            det = {
                "track_id": tid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "known": info["known"],
                "student_id": info["student_id"],
                "name": info["name"],
                "match_pct": info["match_pct"],
                "hard_locked": info["hard_locked"],
                "face_match_events": info.get("face_match_events", 0),
                "keypoints": kpts,
                "key": key,
                "posed": False,
                "pose_status": "ok",
                "reasons": [],
                "group": "unknown",
            }
            if kpts is not None:
                signals = self._face_signals(tid)
                reasons = detect_behaviors(
                    kpts, cfg, self.state_tracker, key, fps_eff, signals=signals,
                )
                det.update(
                    signals=signals, yaw=self.state_tracker.last_turn.get(key),
                    posed=True, reasons=reasons,
                    group=observation_group(reasons),
                    pose_status="ok",
                )
            else:
                # Thiếu keypoint: không coi là bình thường; decay bộ đếm, ghi Không xác định
                # khi đã gửi sang YOLO-Pose nhưng không có điểm khớp.
                self.state_tracker.decay(key, dt, cfg.behavior)
                if tid in pose_tid_set:
                    det.update(
                        posed=True,  # vẫn ghi nhật ký / Excel như một quan sát
                        reasons=[UNKNOWN_BEHAVIOR],
                        group="unknown",
                        pose_status="no_keypoints",
                    )
                else:
                    det.update(
                        posed=False,
                        reasons=[],
                        group="unknown",
                        pose_status="skipped_limit",
                    )
            detections.append(det)

        return detections, parsed
