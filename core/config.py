"""Cấu hình nhận biết hành vi + model (giá trị mặc định, ghi đè bằng biến môi trường)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CLASSROOM_YOLO = BASE_DIR / "models" / "yolo12_classroom" / "best.pt"
POSE_WEIGHTS = BASE_DIR / "models" / "yolo11n-pose.pt"
# Bật/tắt hành vi từ giao diện được lưu ở đây (giữ qua các lần chạy)
BEHAVIOR_SETTINGS = DATA_DIR / "behavior_settings.json"
# Quy tắc ERNU khóa trước kiểm thử cuối (có thể ghi đè bằng ERNU_RULES_PATH)
DEFAULT_ERNU_RULES = BASE_DIR / "evidence" / "config" / "ernu_rules_v1.json"


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
    head_down_drop: float = 0.45  # cúi = thấp hơn p75 độ cao đầu của chính HS ≥ N sw
    head_baseline_min_samples: int = 20
    head_baseline_window: int = 300
    head_down_pitch: float = 35.0  # độ, khi InsightFace thấy mặt
    head_down_duration: float = 5.0  # chỉ khi không có bằng chứng viết bài
    stillness_window: float = 2.0  # cửa sổ đo tốc độ đầu, giây
    shoulder_tilt_threshold: float = 25.0
    shoulder_tilt_duration: float = 2.0
    writing_min_drop: float = 0.4
    writing_max_drop: float = 2.5
    writing_max_offset: float = 0.85  # cổ tay lệch khỏi giữa vai, vẫn tính khi viết lệch sang một bên
    # Bề rộng vai trên khung phân tích (rộng tối đa 1280). Hàng đầu gần camera vai rộng,
    # tay trên bàn nằm gần ống kính hơn vai nên trong ảnh thấp và xòe rộng hơn.
    writing_far_sw: float = 70.0
    writing_near_sw: float = 220.0
    writing_kp_min_conf: float = 0.35  # cổ tay trên bàn thường bị che, thấp hơn ngưỡng khớp chung
    writing_min_speed: float = 0.05  # sw / giây — tay phải chuyển động, không chỉ nằm trên bàn
    writing_move_window: float = 2.0  # nhịp tay trong cửa sổ này vẫn là đang bắt được viết bài
    hand_raise_duration: float = 1.0  # cổ tay + khuỷu + vai, giữ đủ lâu mới gắn nhãn
    head_rotation_threshold: float = 30.0
    head_rotation_duration: float = 2.0
    # Mất tín hiệu ngắn không xóa đợt, và cũng không cộng thêm thời gian cho hành vi
    miss_tolerance_sec: float = 1.0
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
    disabled: frozenset[str] = field(default_factory=frozenset)

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


def _default_ernu_matrix() -> dict[str, dict[str, str]]:
    acts = ("A1", "A2", "A3", "A4", "A5")
    return {
        "B1": {a: "E" for a in acts},
        "B2": {a: "E" for a in acts},  # tự động: viết bài = bằng chứng tập trung, không cần A2
        "B3": {a: "N" for a in acts},
        "B4": {a: "N" for a in acts},  # cúi đầu lâu: không tính không tập trung (có thể đang viết/đọc)
        "B5": {a: "R" for a in acts},
        "NORMAL": {a: "N" for a in acts},
        "MISSING": {a: "U" for a in acts},
    }


def _default_proxy_mapping() -> dict[str, str]:
    return {"E": "ON", "R": "OFF", "N": "UNCERTAIN", "U": "NO_DATA"}


@dataclass
class ErnuConfig:
    """Ngưỡng diễn giải E/R/N/U theo cửa sổ — giá trị khởi tạo để thử nghiệm, chưa kiểm chứng.

    Khóa trước tập kiểm thử cuối (Phụ lục B). C_min và min_duration_* xuất phát từ
    bộ tiêu chí ERNU; không tinh chỉnh theo video kiểm thử cuối.
    """

    window_sec: float = 10.0
    c_min: float = 0.70  # coverage tối thiểu cửa sổ; dưới mức này → U
    # Ngưỡng thời lượng tín hiệu B (giây) khi gán nhãn cửa sổ — khớp BehaviorConfig khởi tạo
    min_duration_b1: float = 1.0   # giơ tay
    min_duration_b2: float = 0.0   # viết bài: detector đã yêu cầu chuyển động; không cắt thêm
    min_duration_b3: float = 2.0   # quay đầu
    min_duration_b4: float = 5.0   # cúi đầu lâu
    min_duration_b5: float = 5.0   # gục xuống bàn
    # Công bố thống kê lớp/HS chỉ khi C ≥ publish_c_min và T_valid > 0
    publish_c_min: float = 0.70
    rules_version: str = "ernu-v1"
    default_activity: str = "A1"
    # BEEI phụ: (w_E, w_R, w_N); xuất thêm w_N=0 và w_N=1 để phân tích độ nhạy
    beei_w_e: float = 1.0
    beei_w_r: float = 0.0
    beei_w_n: float = 0.5
    # Đường dẫn ACTIVITY_CONTEXT (.xlsx/.json); rỗng → mặc định A1 cả phiên
    activity_context_path: str = field(
        default_factory=lambda: (os.environ.get("ERNU_ACTIVITY_CONTEXT") or "").strip()
    )
    rules_path: str = ""
    matrix: dict[str, dict[str, str]] = field(default_factory=_default_ernu_matrix)
    proxy_mapping: dict[str, str] = field(default_factory=_default_proxy_mapping)
    # Ngưỡng điểm hết phiên: >= focus_threshold → "Tập trung"
    focus_threshold: float = 50.0
    # Kết luận tập trung khi đã quan sát hợp lệ tối thiểu (giây) — không đòi bao phủ cả video
    focus_min_valid_sec: float = 20.0

    def min_duration(self, b_code: str) -> float:
        return {
            "B1": self.min_duration_b1,
            "B2": self.min_duration_b2,
            "B3": self.min_duration_b3,
            "B4": self.min_duration_b4,
            "B5": self.min_duration_b5,
        }.get(b_code, 0.0)

    @classmethod
    def load_rules(cls, path: str | Path | None = None) -> "ErnuConfig":
        """Nạp ernu_rules_*.json; thiếu file → mặc định cứng trong code."""
        cfg = cls()
        raw_path = path or (os.environ.get("ERNU_RULES_PATH") or "").strip() or str(DEFAULT_ERNU_RULES)
        p = Path(raw_path)
        cfg.rules_path = str(p)
        if not p.exists():
            print(f"Cảnh báo: không thấy {p} — dùng ErnuConfig mặc định.")
            return cfg
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"Cảnh báo: không đọc được {p}: {exc} — dùng mặc định.")
            return cfg

        cfg.rules_version = str(data.get("rules_version") or cfg.rules_version)
        if "window_sec" in data:
            cfg.window_sec = float(data["window_sec"])
        if "c_min" in data:
            cfg.c_min = float(data["c_min"])
        if "publish_c_min" in data:
            cfg.publish_c_min = float(data["publish_c_min"])
        if "default_activity" in data:
            cfg.default_activity = str(data["default_activity"]).strip().upper() or "A1"
        md = data.get("min_duration_sec") or {}
        if "B1" in md:
            cfg.min_duration_b1 = float(md["B1"])
        if "B2" in md:
            cfg.min_duration_b2 = float(md["B2"])
        if "B3" in md:
            cfg.min_duration_b3 = float(md["B3"])
        if "B4" in md:
            cfg.min_duration_b4 = float(md["B4"])
        if "B5" in md:
            cfg.min_duration_b5 = float(md["B5"])
        beei = data.get("beei") or {}
        if "w_E" in beei:
            cfg.beei_w_e = float(beei["w_E"])
        if "w_R" in beei:
            cfg.beei_w_r = float(beei["w_R"])
        if "w_N" in beei:
            cfg.beei_w_n = float(beei["w_N"])
        mat = data.get("matrix")
        if isinstance(mat, dict) and mat:
            cfg.matrix = {str(k): {str(a): str(lab) for a, lab in row.items()} for k, row in mat.items() if isinstance(row, dict)}
        pm = data.get("proxy_mapping")
        if isinstance(pm, dict) and pm:
            cfg.proxy_mapping = {str(k).upper(): str(v).upper() for k, v in pm.items()}
        if "focus_threshold" in data:
            cfg.focus_threshold = float(data["focus_threshold"])
        if "focus_min_valid_sec" in data:
            cfg.focus_min_valid_sec = float(data["focus_min_valid_sec"])
        # Chế độ tự động: luôn ưu tiên R→OFF nếu JSON chưa cập nhật
        if cfg.proxy_mapping.get("R") == "PENDING":
            cfg.proxy_mapping["R"] = "OFF"
        return cfg


@dataclass
class Config:
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    ernu: ErnuConfig = field(default_factory=ErnuConfig)

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

    # Góc đầu từ InsightFace chỉ dùng khi đo trong N lần phân tích gần nhất
    head_pose_max_age: int = 3

    show_labels: bool = True
    # Ghi keypoint + tín hiệu mặt từng lần phân tích ra logs/keypoints/ (chạy lại bằng eval_behavior.py)
    record_keypoints: bool = field(
        default_factory=lambda: os.environ.get("RECORD_KEYPOINTS", "1") != "0"
    )


def load_config() -> Config:
    cfg = Config()
    cfg.ernu = ErnuConfig.load_rules()
    # Giữ ACTIVITY_CONTEXT từ env nếu rules JSON không ghi đè path
    env_act = (os.environ.get("ERNU_ACTIVITY_CONTEXT") or "").strip()
    if env_act:
        cfg.ernu.activity_context_path = env_act
    disabled = _behavior_disabled()
    if disabled is None and BEHAVIOR_SETTINGS.exists():
        try:
            data = json.loads(BEHAVIOR_SETTINGS.read_text(encoding="utf-8"))
            disabled = list(data.get("disabled", []))
            _store_behavior_disabled(disabled)
        except Exception as exc:  # noqa: BLE001
            print(f"Không đọc được {BEHAVIOR_SETTINGS.name}, dùng mặc định: {exc}")
    if disabled is not None:
        cfg.behavior.disabled = frozenset(disabled)
    return cfg


def _behavior_disabled() -> list | None:
    from core.db import enabled, get_db
    if not enabled():
        return None
    doc = get_db().settings.find_one({"_id": "behavior"})
    if doc is None:
        return None
    return list(doc.get("disabled") or [])


def _store_behavior_disabled(disabled: list) -> None:
    from core.db import enabled, get_db
    if not enabled():
        return
    get_db().settings.replace_one(
        {"_id": "behavior"},
        {"_id": "behavior", "disabled": list(disabled)},
        upsert=True,
    )


def save_behavior_settings(cfg: Config) -> None:
    disabled = sorted(cfg.behavior.disabled)
    from core.db import enabled
    if enabled():
        _store_behavior_disabled(disabled)
        return
    BEHAVIOR_SETTINGS.write_text(
        json.dumps({"disabled": disabled}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
