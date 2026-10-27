"""Cấu hình nhận biết hành vi + model (giá trị mặc định, ghi đè bằng biến môi trường)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # 3.Thesis
DATA_DIR = BASE_DIR.parent / "2.Data"
CLASSROOM_YOLO = BASE_DIR / "models" / "yolo12_classroom" / "best.pt"
POSE_WEIGHTS = BASE_DIR / "models" / "yolo11n-pose.pt"
# Bật/tắt hành vi từ giao diện được lưu ở đây (giữ qua các lần chạy)
BEHAVIOR_SETTINGS = DATA_DIR / "behavior_settings.json"
# Ẩn khỏi giao diện và luôn tắt (vẫn chạy được khi đo offline bằng tools/eval_behavior.py)
HIDDEN_BEHAVIORS = frozenset({"Nói chuyện", "Tay bất thường", "Cúi đầu thoáng qua", "Nhắm mắt"})
DEFAULT_DISABLED = HIDDEN_BEHAVIORS


@dataclass
class BehaviorConfig:
    """Ngưỡng hậu xử lý. Khoảng cách theo bề rộng vai (sw), thời gian theo giây.

    Đây là ngưỡng khởi đầu, chưa phải cấu hình đã chứng minh trên video thật.
    Chỉnh lại bằng eval_behavior.py sau khi có nhãn.
    """
    # Điểm khớp dưới mức này không dùng để kết luận. EMA: hệ số của giá trị mới.
    kp_min_conf: float = 0.5
    ema_alpha: float = 0.6
    # Tư thế nền chỉ nhận khung đầu gần như đứng yên và không lệch xa mức hiện có.
    baseline_max_speed: float = 0.2  # sw / giây
    baseline_yaw_band: float = 15.0  # độ, lệch tối đa còn được đưa vào hướng nhìn nền
    # Cúi đầu: độ cao mũi trên đường vai (sw). Ngồi thẳng thường ~0.4–0.7.
    head_down_ratio: float = 0.25  # ngưỡng chung khi chưa đủ lịch sử tư thế của HS
    head_down_drop: float = 0.35  # cúi = thấp hơn p75 độ cao đầu của chính HS ≥ N sw
    head_baseline_min_samples: int = 20
    head_baseline_window: int = 300
    head_down_pitch: float = 25.0  # độ, khi InsightFace thấy mặt
    head_down_duration: float = 5.0  # chỉ khi không có bằng chứng viết bài
    eye_closed_threshold: float = 0.18
    eye_closed_duration: float = 2.0
    stillness_speed: float = 0.08  # sw / giây
    stillness_window: float = 2.0
    stillness_duration: float = 4.0
    shoulder_tilt_threshold: float = 25.0
    shoulder_tilt_duration: float = 2.0
    hand_below_hip_ratio: float = 0.2  # cổ tay thấp hơn hông ≥ N sw
    hand_hip_min_conf: float = 0.6  # hông thường bị bàn che → pose đoán, đòi conf cao
    hand_low_duration: float = 2.0
    writing_min_drop: float = 0.4
    writing_max_drop: float = 2.5
    writing_max_offset: float = 0.6
    writing_min_speed: float = 0.15  # sw / giây — tay phải chuyển động, không chỉ nằm trên bàn
    writing_move_window: float = 1.0
    hand_raise_duration: float = 1.0  # cổ tay + khuỷu + vai, giữ đủ lâu mới gắn nhãn
    head_rotation_threshold: float = 30.0
    head_rotation_duration: float = 2.0
    # Mất tín hiệu ngắn không xóa đợt, và cũng không cộng thêm thời gian cho hành vi
    miss_tolerance_sec: float = 1.0
    chat_distance_threshold: float = 0.15  # khoảng hở ngang giữa 2 box / chiều cao box
    chat_row_tolerance: float = 0.5  # lệch tâm dọc / chiều cao box → cùng hàng
    chat_yaw_threshold: float = 25.0
    chat_duration: float = 2.0
    chat_mouth_std: float = 0.02  # độ lệch chuẩn MAR dưới mức này = miệng không động
    # Gục xuống bàn: cần đồng thời vị trí đầu và thân nghiêng, giữ đủ lâu.
    # Không kết luận học sinh đang ngủ. desk_sleep_face_down_duration giữ lại cho lần dò ngưỡng cũ.
    desk_sleep_duration: float = 5.0
    desk_sleep_face_down_duration: float = 15.0
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

    yolo_model: str = field(default_factory=_resolve_yolo_weights)
    yolo_imgsz: int = 640
    confidence_threshold: float = 0.15
    iou_threshold: float = 0.45

    # YOLOv12 không có trọng số pose chính thức → dùng YOLO11-pose
    pose_model: str = field(
        default_factory=lambda: (os.environ.get("POSE_MODEL") or str(POSE_WEIGHTS)).strip()
    )
    pose_imgsz: int = 320
    pose_conf: float = 0.25
    pose_max_per_frame: int = 40

    # EAR chỉ đo được khi mặt đủ lớn; người nhỏ hơn dùng độ tin cậy keypoint mắt
    eye_min_box_width: int = 90
    # Góc đầu từ InsightFace chỉ dùng khi đo trong N lần phân tích gần nhất
    head_pose_max_age: int = 3

    show_labels: bool = True
    # Ghi keypoint + tín hiệu mặt từng lần phân tích ra logs/keypoints/ (chạy lại bằng eval_behavior.py)
    record_keypoints: bool = field(
        default_factory=lambda: os.environ.get("RECORD_KEYPOINTS", "1") != "0"
    )


def load_config() -> Config:
    cfg = Config()
    if BEHAVIOR_SETTINGS.exists():
        try:
            data = json.loads(BEHAVIOR_SETTINGS.read_text(encoding="utf-8"))
            cfg.behavior.disabled = frozenset(data.get("disabled", [])) | HIDDEN_BEHAVIORS
        except Exception as exc:  # noqa: BLE001
            print(f"Không đọc được {BEHAVIOR_SETTINGS.name}, dùng mặc định: {exc}")
    return cfg


def save_behavior_settings(cfg: Config) -> None:
    BEHAVIOR_SETTINGS.write_text(
        json.dumps({"disabled": sorted(cfg.behavior.disabled)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
