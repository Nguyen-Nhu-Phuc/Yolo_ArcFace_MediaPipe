"""Cấu hình chấm điểm tập trung + model (giá trị mặc định, ghi đè bằng biến môi trường)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CLASSROOM_YOLO = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
# Bật/tắt hành vi từ giao diện được lưu ở đây (giữ qua các lần chạy)
BEHAVIOR_SETTINGS = BASE_DIR / "behavior_settings.json"
DEFAULT_DISABLED = frozenset({"Nói chuyện"})


@dataclass
class BehaviorConfig:
    """Ngưỡng các luật hành vi (giống classroom_attention)."""
    head_down_threshold: float = 0.08
    head_down_duration: float = 3.0
    head_down_penalty: int = 80
    eye_closed_threshold: float = 0.18
    eye_closed_duration: float = 2.0
    eye_closed_penalty: int = 70
    stillness_threshold: float = 25.0
    stillness_duration: float = 4.0
    stillness_penalty: int = 50
    shoulder_tilt_threshold: int = 25
    shoulder_tilt_penalty: int = 20
    hand_below_hip_threshold: float = 0.02
    hand_below_hip_penalty: int = 15
    short_head_down_penalty: int = 30
    writing_min_drop: float = 0.4
    writing_max_drop: float = 2.5
    writing_max_offset: float = 0.6
    head_rotation_threshold: float = 30.0
    head_rotation_duration: float = 2.0
    head_rotation_penalty: int = 25
    chat_distance_threshold: float = 0.15
    chat_angle_threshold: float = 40.0
    chat_penalty: int = 15
    # Ngủ gục: nằm úp / nằm nghiêng lên bàn (đo theo bề rộng vai, góc theo độ)
    desk_sleep_duration: float = 5.0  # khi có dấu hiệu nằm nghiêng / thân nằm ngang
    desk_sleep_face_down_duration: float = 15.0  # chỉ úp mặt lên tay: dễ lẫn cúi viết sát bàn
    desk_sleep_penalty: int = 90
    desk_sleep_shoulder_angle: float = 45.0  # vai dựng đứng → người nằm nghiêng
    desk_sleep_head_side: float = 0.6  # đầu lệch ngang khỏi giữa vai ≥ N bề rộng vai
    desk_sleep_torso_angle: float = 55.0  # hông → vai lệch khỏi phương thẳng đứng
    desk_sleep_head_drop: float = 0.15  # đầu thấp hơn vai ≥ N bề rộng vai (úp mặt lên tay)
    desk_sleep_wrist_dy: float = 0.3  # cổ tay ngang tầm đầu (đầu gối lên tay)
    desk_sleep_max_move: float = 0.6  # đầu dịch > N bề rộng vai giữa 2 lần phân tích → không phải ngủ
    # Tên hành vi đang tắt (theo behavior.REASON_SHORT). Gán frozenset mới, không sửa tại chỗ:
    # luồng phân tích đọc song song khi giao diện bật/tắt.
    disabled: frozenset[str] = DEFAULT_DISABLED

    def is_on(self, name: str) -> bool:
        return name not in self.disabled


def _resolve_yolo_weights() -> str:
    """
    YOLO_WEIGHTS=classroom (mặc định best.pt — YOLOv12n train cho lớp học)
    | coco (yolo12n.pt). Hoặc YOLO_MODEL_PATH=đường dẫn .pt
    """
    override = (os.environ.get("YOLO_MODEL_PATH") or "").strip()
    if override:
        return override
    mode = (os.environ.get("YOLO_WEIGHTS") or "classroom").strip().lower()
    coco = (os.environ.get("YOLO_COCO_WEIGHTS") or "yolo12n.pt").strip()
    if mode in ("coco", "general", "pretrained"):
        return coco
    if CLASSROOM_YOLO.exists():
        return str(CLASSROOM_YOLO)
    print(f"Cảnh báo: không thấy best.pt — dùng {coco} (COCO).")
    return coco


@dataclass
class Config:
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    attention_threshold: int = 50

    yolo_model: str = field(default_factory=_resolve_yolo_weights)
    yolo_imgsz: int = 640
    confidence_threshold: float = 0.15
    iou_threshold: float = 0.45

    # YOLOv12 không có trọng số pose chính thức → dùng YOLO11-pose
    pose_model: str = field(
        default_factory=lambda: (os.environ.get("POSE_MODEL") or "yolo11n-pose.pt").strip()
    )
    pose_imgsz: int = 320
    pose_conf: float = 0.25
    pose_max_per_frame: int = 40

    # EAR chỉ đo được khi mặt đủ lớn; người nhỏ hơn dùng độ tin cậy keypoint mắt
    eye_min_box_width: int = 90

    show_labels: bool = True


def load_config() -> Config:
    cfg = Config()
    if BEHAVIOR_SETTINGS.exists():
        try:
            data = json.loads(BEHAVIOR_SETTINGS.read_text(encoding="utf-8"))
            cfg.behavior.disabled = frozenset(data.get("disabled", []))
        except Exception as exc:  # noqa: BLE001
            print(f"Không đọc được {BEHAVIOR_SETTINGS.name}, dùng mặc định: {exc}")
    return cfg


def save_behavior_settings(cfg: Config) -> None:
    BEHAVIOR_SETTINGS.write_text(
        json.dumps({"disabled": sorted(cfg.behavior.disabled)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
