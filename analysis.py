"""Phân tích 1 khung: YOLOv12 track → YOLO11-pose → tín hiệu mặt → chấm điểm → nói chuyện.

Dùng chung cho web (app.py) và chạy offline (extract_keypoints.py), nên log keypoint
của 2 nơi giống hệt nhau.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from behavior import StudentStateTracker, calculate_attention_score, is_head_down
from chat_detector import ChatTracker
from config import Config
from identity import IdentityTracker
from pose_estimator import PoseEstimator

HEAD_CROP_FRAC = 0.42  # phần trên box người dùng làm crop mặt cho EAR / MAR
WORK_MAX_WIDTH = 1280  # khung được thu về độ rộng này trước khi phân tích + hiển thị
# Phân tích chậm (CPU) → mỗi lần chỉ cộng tối đa N giây vào bộ đếm hành vi,
# để 1 lần quan sát không đủ vượt ngưỡng thời gian (vd. nhắm mắt 2s)
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
    return max(fps / max(1, frames_elapsed), 1.0 / MAX_BEHAVIOR_STEP_SEC)


class FrameAnalyzer:
    """Không thread-safe (predictor Ultralytics, MediaPipe) — 1 instance / luồng phân tích."""

    def __init__(self, cfg: Config, identity: IdentityTracker):
        self.cfg = cfg
        self.identity = identity
        self.yolo = YOLO(cfg.yolo_model)
        self.pose = PoseEstimator(cfg.pose_model, cfg.pose_imgsz, cfg.pose_conf)
        self.eye_detector = None
        try:
            from eye_detector import EyeDetector

            self.eye_detector = EyeDetector()
        except Exception as exc:  # noqa: BLE001
            print(f"Không bật được EyeDetector (không đo nhắm mắt / miệng): {exc}")
        self.state_tracker = StudentStateTracker()
        self.chat = ChatTracker()

    def close(self) -> None:
        if self.eye_detector is not None:
            self.eye_detector.close()

    def _face_signals(self, frame: np.ndarray, tid: int, box, kpts) -> dict:
        cfg = self.cfg
        x1, y1, x2, y2 = box
        signals: dict = {"ear": None, "mar": None, "yaw": None, "pitch": None}
        pose = self.identity.head_pose(tid, cfg.head_pose_max_age)
        if pose is not None:
            signals["yaw"], signals["pitch"] = round(pose[0], 1), round(pose[1], 1)
        # Cúi nhìn xuống làm mắt trông như nhắm → không đo EAR (ngủ gục đã có luật cúi đầu lâu)
        if (self.eye_detector is not None
                and (cfg.behavior.is_on("Nhắm mắt") or cfg.behavior.is_on("Nói chuyện"))
                and x2 - x1 >= cfg.eye_min_box_width
                and not is_head_down(kpts, cfg.behavior)):
            face = self.eye_detector.measure(frame[y1:y1 + int((y2 - y1) * HEAD_CROP_FRAC), x1:x2])
            if face:
                signals["ear"] = round(face["ear"], 4)
                signals["mar"] = round(face["mar"], 4) if face["mar"] is not None else None
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
        kpts_by_tid = dict(
            zip(
                [t[0] for t in pose_targets],
                self.pose.estimate(frame, [t[1:] for t in pose_targets]),
            )
        )

        detections: list[dict] = []
        for tid, x1, y1, x2, y2 in parsed:
            info = self.identity.describe(tid)
            kpts = kpts_by_tid.get(tid)
            if kpts is None and info["no_face_suspect"]:
                continue
            det = {
                "track_id": tid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "known": info["known"],
                "student_id": info["student_id"],
                "name": info["name"],
                "match_pct": info["match_pct"],
                "hard_locked": info["hard_locked"],
                "keypoints": kpts,
                "score": None,
                "reasons": [],
                "is_focused": None,
            }
            if kpts is not None:
                key = info["student_id"] or f"t{tid}"
                signals = self._face_signals(frame, tid, (x1, y1, x2, y2), kpts)
                score, reasons = calculate_attention_score(
                    kpts, cfg, self.state_tracker, key, fps_eff, signals=signals,
                )
                det.update(
                    key=key, signals=signals, yaw=self.state_tracker.last_turn.get(key),
                    mar=signals["mar"],
                    score=int(score), reasons=reasons,
                    is_focused=score >= cfg.attention_threshold,
                )
            detections.append(det)

        apply_chat(self.chat, [d for d in detections if d["score"] is not None], cfg, fps_eff)
        return detections, parsed


def apply_chat(chat: ChatTracker, scored: list[dict], cfg: Config, fps_eff: float) -> None:
    for tid_a, tid_b in chat.update(scored, cfg, 1.0 / fps_eff):
        for det in scored:
            if det["track_id"] in (tid_a, tid_b):
                det["score"] = max(0, det["score"] - cfg.behavior.chat_penalty)
                if "Nói chuyện" not in det["reasons"]:
                    det["reasons"].append("Nói chuyện")
                det["is_focused"] = det["score"] >= cfg.attention_threshold
