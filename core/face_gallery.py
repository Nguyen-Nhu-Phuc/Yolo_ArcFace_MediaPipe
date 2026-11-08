"""
Nhận diện học sinh bằng ArcFace (InsightFace) + gallery đã XUẤT.

Quy trình đúng:
  1) python tools/export_arcface_model.py   ← xuất embedding từ dataset
  2) App load models/arcface_*/gallery.npz rồi so khớp cosine

Không train lại backbone ArcFace trên 33 ảnh — chỉ xuất gallery embeddings.
"""

from __future__ import annotations

import csv
import io
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from core.face_augment import face_rois_from_person

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def _existing_gallery_dirs() -> tuple[Path, Path]:
    """Chọn dataset/gallery đang có trên máy. Biến môi trường được ưu tiên."""
    env_data = os.environ.get("CLASSROOM_FACES_DIR")
    env_model = os.environ.get("ARCFACE_MODEL_DIR")
    if env_data or env_model:
        dataset = Path(env_data) if env_data else DATA_DIR / "2_dataset_arcface"
        if not dataset.is_absolute():
            dataset = (BASE_DIR / dataset).resolve()
        model = Path(env_model) if env_model else BASE_DIR / "models" / f"arcface_{dataset.name}"
        if not model.is_absolute():
            model = (BASE_DIR / model).resolve()
        return dataset, model

    candidates = [
        (DATA_DIR / "2_dataset_arcface", BASE_DIR / "models" / "arcface_2_dataset_arcface"),
        (DATA_DIR / "classroom-faces-lop07", BASE_DIR / "models" / "arcface_classroom-faces-lop07"),
    ]
    for dataset, model in candidates:
        if (model / "gallery.npz").exists() or (dataset / "arcface_gallery.npz").exists():
            return dataset, model
    found = sorted((BASE_DIR / "models").glob("arcface_*/gallery.npz"))
    if found:
        model = found[0].parent
        return DATA_DIR / model.name[len("arcface_"):], model
    return candidates[0]


DATASET_DIR, MODEL_DIR = _existing_gallery_dirs()
GALLERY_NPZ = MODEL_DIR / "gallery.npz"
META_JSON = MODEL_DIR / "meta.json"
# Cache cũ (fallback)
LEGACY_NPZ = DATASET_DIR / "arcface_gallery.npz"
STUDENTS_CSV = DATASET_DIR / "students.csv"
# Danh sách họ tên để xuất báo cáo. Có file này thì dùng thay tên placeholder trong students.csv.
ROSTER_CSV = DATASET_DIR / "danh_sach_hoc_sinh.csv"

# Ngưỡng mặc định cho InsightFace ArcFace (sẽ ghi đè từ meta.json nếu có)
# Quy tắc: % tương thích = cosine_sim * 100. Dưới MIN_COMPAT_PCT → không gán / gán lại.
MATCH_THRESHOLD = 0.45
MIN_COMPAT_PCT = 45.0
MIN_MARGIN = 0.004  # lớp đông: top-1 / top-2 thường sát nhau
LOCK_SIM = 0.45
# Khóa cứng: khớp ArcFace cao + tách hẳn người khớp nhì → chỉ đổi khi HS khác có % cao hơn.
# Đo trên video lớp: top-1 trung vị ~47%, chênh với top-2 trung vị ~17%.
HARD_LOCK_PCT = float(os.environ.get("HARD_LOCK_PCT", "45"))
HARD_LOCK_MARGIN_PCT = float(os.environ.get("HARD_LOCK_MARGIN_PCT", "10"))
# Các khóa ngưỡng trong meta.json — giữ nguyên khi dựng lại gallery.
THRESHOLD_KEYS = (
    "match_threshold", "min_compat_pct", "min_margin", "lock_sim",
    "hard_lock_pct", "hard_lock_margin_pct",
)

_face_app = None
_gallery_write_lock = threading.Lock()


def read_meta() -> dict:
    if not META_JSON.exists():
        return {}
    try:
        return json.loads(META_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_meta(meta: dict) -> None:
    META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def derive_lock_sim(match_threshold: float, hard_lock_sim: float) -> float:
    """Mức 'khớp chắc' (dừng thử thêm crop, nhận lại mã khi track đổi): trên ngưỡng nhận diện
    8 điểm %, nhưng không vượt ngưỡng khóa cứng."""
    return max(match_threshold, min(match_threshold + 0.08, hard_lock_sim))


def _load_thresholds_from_meta() -> None:
    global MATCH_THRESHOLD, MIN_COMPAT_PCT, MIN_MARGIN, LOCK_SIM, HARD_LOCK_PCT, HARD_LOCK_MARGIN_PCT
    if not META_JSON.exists():
        return
    try:
        meta = json.loads(META_JSON.read_text(encoding="utf-8"))
        HARD_LOCK_PCT = float(meta.get("hard_lock_pct", HARD_LOCK_PCT))
        HARD_LOCK_MARGIN_PCT = float(meta.get("hard_lock_margin_pct", HARD_LOCK_MARGIN_PCT))
        MATCH_THRESHOLD = float(meta.get("match_threshold", MATCH_THRESHOLD))
        MIN_COMPAT_PCT = float(
            meta.get("min_compat_pct", meta.get("match_threshold", MATCH_THRESHOLD) * 100.0)
        )
        # Đồng bộ threshold với % tương thích tối thiểu
        MATCH_THRESHOLD = max(MATCH_THRESHOLD, MIN_COMPAT_PCT / 100.0)
        MIN_COMPAT_PCT = max(MIN_COMPAT_PCT, MATCH_THRESHOLD * 100.0)
        MIN_MARGIN = float(meta.get("min_margin", MIN_MARGIN))
        LOCK_SIM = float(meta.get("lock_sim", LOCK_SIM))
        LOCK_SIM = max(LOCK_SIM, MATCH_THRESHOLD)
    except Exception:
        pass


def sim_to_pct(sim: float) -> float:
    """Cosine similarity → điểm tương đồng quy đổi (%) 0–100.

    Đây là phép quy đổi tuyến tính, không phải xác suất nhận diện đúng.
    """
    return max(0.0, min(100.0, float(sim) * 100.0))


def _roster_path() -> Path:
    return ROSTER_CSV if ROSTER_CSV.exists() else STUDENTS_CSV


def _read_students_csv() -> list[dict]:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import roster_rows
        return roster_rows()
    path = _roster_path()
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_student_classes() -> dict[str, str]:
    """Mã HS → lớp, đọc từ danh sách họ tên (thiếu file/cột thì trả rỗng)."""
    return {
        sid: (row.get("class_name") or "").strip()
        for row in _read_students_csv()
        if (sid := (row.get("student_id") or "").strip())
    }


def load_student_names() -> dict[str, str]:
    """Mã HS → họ tên trong danh sách lớp. Bỏ ô trống và ô chỉ lặp lại mã."""
    names: dict[str, str] = {}
    for row in _read_students_csv():
        sid = (row.get("student_id") or "").strip()
        full = (row.get("full_name") or "").strip()
        if sid and full and full != sid:
            names[sid] = full
    return names


_display_codes: dict[str, str] | None = None


def display_codes() -> dict[str, str]:
    """Mã gallery → số trên khung, từ 1 theo đúng thứ tự danh sách học sinh."""
    global _display_codes
    if _display_codes is None:
        codes: dict[str, str] = {}
        n = 1
        for row in _read_students_csv():
            sid = (row.get("student_id") or "").strip()
            if sid and sid not in codes:
                codes[sid] = str(n)
                n += 1
        _display_codes = codes
    return _display_codes


def display_code(student_id: str | None) -> str:
    if not student_id:
        return ""
    return display_codes().get(student_id, student_id)


def reset_display_codes() -> None:
    """Xóa cache số trên khung sau khi danh sách lớp đổi thứ tự hoặc thêm học sinh."""
    global _display_codes
    _display_codes = None


def roster_names(ids: list[str], gallery_names: list[str]) -> dict[str, str]:
    """Danh sách lớp để xuất Excel: họ tên trong students.csv ghi đè tên lưu trong gallery."""
    names: dict[str, str] = {}
    for sid, name in zip(ids, gallery_names):
        sid = (sid or "").strip()
        name = (name or "").strip()
        if sid:
            names[sid] = "" if not name or name == sid else name
    for sid, full in load_student_names().items():
        names[sid] = full
    return names


def get_face_app():
    """InsightFace buffalo_l (ArcFace + detector)."""
    global _face_app
    if _face_app is None:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        print("Nạp InsightFace ArcFace (buffalo_l)...")
        # Giới hạn thread: quá nhiều → YOLO/pose chậm; quá ít → gắn tên trễ
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ARCFACE_THREADS", "4"))
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        # 256 nhanh hơn 320 trên CPU; đặt ARCFACE_DET_SIZE=320/640 nếu cần mặt xa hơn
        det = int(os.environ.get("ARCFACE_DET_SIZE", "256"))
        det = max(160, min(640, det))
        if det % 32:
            det = max(160, (det // 32) * 32)
        app = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"], sess_options=so
        )
        app.prepare(ctx_id=-1, det_size=(det, det))
        _face_app = app
        print(f"InsightFace sẵn sàng (det_size={det}, threads={so.intra_op_num_threads}).")
    return _face_app


def embed_bgr(img_bgr: np.ndarray) -> np.ndarray | None:
    if img_bgr is None or img_bgr.size == 0:
        return None
    app = get_face_app()
    faces = app.get(img_bgr)
    if faces:
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return np.asarray(faces[0].normed_embedding, dtype=np.float32)

    h, w = img_bgr.shape[:2]
    pad = max(8, max(h, w) // 5)
    padded = cv2.copyMakeBorder(
        img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    faces = app.get(padded)
    if faces:
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return np.asarray(faces[0].normed_embedding, dtype=np.float32)

    if max(h, w) < 160:
        scale = 160.0 / max(h, w)
        up = cv2.resize(
            img_bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        return embed_bgr(up)
    return None


def _npz_ids(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "ids" not in getattr(data, "files", []):
        return None
    return [str(x) for x in data["ids"].tolist()]


def resolve_gallery_path() -> Path:
    """Gallery trong models/ được dùng khi cùng mã với danh sách lớp.

    Thư mục dataset vừa thay (arcface_gallery.npz) được ưu tiên nếu bản trong
    models/ còn mã học sinh của lần xuất cũ.
    """
    roster = [
        sid for row in _read_students_csv()
        if (sid := (row.get("student_id") or "").strip())
    ]
    legacy_ids = _npz_ids(LEGACY_NPZ)
    model_ids = _npz_ids(GALLERY_NPZ)
    if roster and legacy_ids == roster and model_ids != roster:
        return LEGACY_NPZ
    if GALLERY_NPZ.exists():
        return GALLERY_NPZ
    if LEGACY_NPZ.exists():
        return LEGACY_NPZ
    found = sorted((BASE_DIR / "models").glob("arcface_*/gallery.npz"))
    if found:
        return found[0]
    raise FileNotFoundError(
        f"Chưa có gallery ArcFace.\n"
        f"Chạy trước: python tools/export_arcface_model.py --dataset {DATASET_DIR.name}\n"
        f"Kỳ vọng file: {GALLERY_NPZ}"
    )


class FaceRecognizer:
    """Load gallery đã xuất + so khớp ArcFace InsightFace."""

    def __init__(self, force_rebuild: bool = False):
        from core.db import enabled

        raw = None
        source = ""
        if enabled():
            from core.gallery_store import load_bytes, rebuild_from_roster
            from core.mongo_roster import ensure_ready

            ensure_ready()
            raw = rebuild_from_roster() if force_rebuild else load_bytes()
            source = "MongoDB Atlas"
        _load_thresholds_from_meta()
        if raw is not None:
            data = np.load(io.BytesIO(raw), allow_pickle=True)
            self.gallery_path = source
        else:
            if force_rebuild:
                from tools.export_arcface_model import export_gallery

                export_gallery(DATASET_DIR, MODEL_DIR)
            path = resolve_gallery_path()
            data = np.load(path, allow_pickle=True)
            self.gallery_path = path
        self.ids: list[str] = data["ids"].tolist()
        self.names: list[str] = data["names"].tolist()
        self.embeddings: np.ndarray = data["embeddings"].astype(np.float32)
        # chuẩn hóa lại
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings = self.embeddings / norms
        self.match_threshold = MATCH_THRESHOLD
        self.min_compat_pct = MIN_COMPAT_PCT
        self.min_margin = MIN_MARGIN
        self.lock_sim = LOCK_SIM
        self.hard_lock_sim = HARD_LOCK_PCT / 100.0
        self.hard_lock_margin = HARD_LOCK_MARGIN_PCT / 100.0

        # warmup
        get_face_app()
        print(
            f"Gallery ArcFace: {len(self.ids)} HS từ {self.gallery_path} | "
            f"thr={self.match_threshold:.2f} (≥{self.min_compat_pct:.0f}%) "
            f"margin={self.min_margin} | khóa cứng ≥{HARD_LOCK_PCT:.0f}% (+{HARD_LOCK_MARGIN_PCT:.0f}%)"
        )

    def for_ids(self, allowed_ids: set[str] | list[str]) -> "FaceRecognizer":
        """Bản gallery chỉ gồm mã trong allowed_ids — dùng cho nhận diện theo lớp.

        Không nạp lại InsightFace; ngưỡng lấy từ bản gốc tại thời điểm gọi.
        """
        allowed = {str(x).strip() for x in allowed_ids if str(x).strip()}
        idx = [i for i, sid in enumerate(self.ids) if sid in allowed]
        clone = object.__new__(FaceRecognizer)
        clone.gallery_path = f"{self.gallery_path} · {len(idx)}/{len(self.ids)} HS"
        clone.ids = [self.ids[i] for i in idx]
        clone.names = [self.names[i] for i in idx]
        if idx:
            clone.embeddings = self.embeddings[np.asarray(idx, dtype=np.int64)]
        else:
            dim = int(self.embeddings.shape[1]) if getattr(self.embeddings, "ndim", 0) == 2 else 512
            clone.embeddings = np.zeros((0, dim), dtype=np.float32)
        clone.match_threshold = self.match_threshold
        clone.min_compat_pct = self.min_compat_pct
        clone.min_margin = self.min_margin
        clone.lock_sim = self.lock_sim
        clone.hard_lock_sim = self.hard_lock_sim
        clone.hard_lock_margin = self.hard_lock_margin
        return clone

    def for_class(self, class_id: str, class_map: dict[str, str] | None = None) -> "FaceRecognizer":
        """Gallery chỉ học sinh thuộc lớp class_id."""
        code = (class_id or "").strip()
        if not code:
            return self
        classes = class_map if class_map is not None else load_student_classes()
        allowed = {sid for sid, c in classes.items() if (c or "").strip() == code}
        return self.for_ids(allowed)

    def _threshold_meta(self) -> dict:
        return {
            "match_threshold": round(self.match_threshold, 4),
            "min_compat_pct": round(self.min_compat_pct, 2),
            "min_margin": self.min_margin,
            "lock_sim": round(self.lock_sim, 4),
            "hard_lock_pct": round(self.hard_lock_sim * 100, 1),
            "hard_lock_margin_pct": round(self.hard_lock_margin * 100, 1),
        }

    def thresholds(self) -> dict:
        """Ngưỡng đang dùng (theo %) + độ giống giữa các HS trong gallery để tham khảo."""
        meta = read_meta()
        off_mean, off_max = meta.get("offdiag_sim_mean"), meta.get("offdiag_sim_max")
        return {
            "match_pct": round(self.min_compat_pct, 1),
            "hard_lock_pct": round(self.hard_lock_sim * 100, 1),
            "hard_lock_margin_pct": round(self.hard_lock_margin * 100, 1),
            "lock_pct": round(self.lock_sim * 100, 1),
            "gallery_size": len(self.ids),
            "offdiag_mean_pct": round(off_mean * 100, 1) if off_mean is not None else None,
            "offdiag_max_pct": round(off_max * 100, 1) if off_max is not None else None,
        }

    def set_thresholds(self, match_pct: float, hard_lock_pct: float, hard_lock_margin_pct: float) -> None:
        """Đổi ngưỡng (áp dụng ngay cho mọi lần so khớp sau) và lưu vào meta.json."""
        if not 5 <= match_pct <= 95:
            raise ValueError("Ngưỡng nhận diện phải trong khoảng 5–95%.")
        if not match_pct <= hard_lock_pct <= 99:
            raise ValueError("Ngưỡng khóa cứng phải ≥ ngưỡng nhận diện và ≤ 99%.")
        if not 0 <= hard_lock_margin_pct <= 50:
            raise ValueError("Chênh lệch khóa cứng phải trong khoảng 0–50%.")
        self.match_threshold = match_pct / 100.0
        self.min_compat_pct = float(match_pct)
        self.hard_lock_sim = hard_lock_pct / 100.0
        self.hard_lock_margin = hard_lock_margin_pct / 100.0
        self.lock_sim = derive_lock_sim(self.match_threshold, self.hard_lock_sim)
        with _gallery_write_lock:
            meta = read_meta()
            meta.update(self._threshold_meta())
            _write_meta(meta)

    def is_compatible(self, sim: float) -> bool:
        """True nếu % tương thích ≥ MIN_COMPAT_PCT (và ≥ match_threshold)."""
        return sim_to_pct(sim) >= self.min_compat_pct and float(sim) >= self.match_threshold

    def _accept_match(self, best_sim: float, margin: float, n_gallery: int) -> bool:
        """Chấp nhận top-1 khi đủ tương thích và không mơ hồ với top-2.

        Gallery 1 người: không có đối thủ → chỉ cần vượt ngưỡng tương thích.
        Gallery ≥2: cần điểm ≥ lock_sim hoặc chênh lệch ≥ min_margin
        (không dùng margin >= 0.0 — điều kiện đó làm mất tác dụng kiểm tra margin).
        """
        if not self.is_compatible(best_sim):
            return False
        if n_gallery <= 1:
            return True
        return best_sim >= self.lock_sim or margin >= self.min_margin

    def match(
        self, face_bgr: np.ndarray, detector: str | None = None
    ) -> tuple[str | None, str | None, float, float]:
        emb = embed_bgr(face_bgr)
        if emb is None:
            return None, None, 0.0, 0.0
        sims = self.embeddings @ emb
        order = np.argsort(sims)[::-1]
        best_i = int(order[0])
        best_sim = float(sims[best_i])
        second = float(sims[order[1]]) if len(order) > 1 else 0.0
        margin = best_sim - second
        if self._accept_match(best_sim, margin, len(order)):
            return self.ids[best_i], self.names[best_i], best_sim, margin
        return None, None, best_sim, margin

    def _match_embedding(
        self, emb: np.ndarray
    ) -> tuple[str | None, str | None, float, float]:
        if emb is None or self.embeddings.size == 0:
            return None, None, 0.0, 0.0
        emb = np.asarray(emb, dtype=np.float32)
        n = float(np.linalg.norm(emb) + 1e-8)
        emb = emb / n
        sims = self.embeddings @ emb
        order = np.argsort(sims)[::-1]
        best_i = int(order[0])
        best_sim = float(sims[best_i])
        second = float(sims[order[1]]) if len(order) > 1 else 0.0
        margin = best_sim - second
        if self._accept_match(best_sim, margin, len(order)):
            return self.ids[best_i], self.names[best_i], best_sim, margin
        return None, None, best_sim, margin

    def match_person_crop(
        self, person_bgr: np.ndarray
    ) -> tuple[str | None, str | None, float, float]:
        if person_bgr is None or person_bgr.size == 0:
            return None, None, 0.0, 0.0
        h, w = person_bgr.shape[:2]
        if h < 32 or w < 20:
            return None, None, 0.0, 0.0

        app = get_face_app()
        # Upscale crop người nhỏ trước khi detect mặt (cuối lớp)
        work = person_bgr
        if max(h, w) < 160:
            scale = 160.0 / max(h, w)
            work = cv2.resize(
                person_bgr,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        best_sid = best_name = None
        best_sim = 0.0
        best_margin = 0.0

        # Ưu tiên embedding InsightFace trực tiếp (không detect 2 lần)
        faces = app.get(work) or []
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        for face in faces[:1]:
            emb = np.asarray(face.normed_embedding, dtype=np.float32)
            sid, name, sim, margin = self._match_embedding(emb)
            if sid is not None and sim > best_sim:
                best_sid, best_name, best_sim, best_margin = sid, name, sim, margin
                if sim >= self.lock_sim:
                    return best_sid, best_name, best_sim, best_margin
            elif sim > best_sim:
                best_sim, best_margin = sim, margin

        if best_sid is not None:
            return best_sid, best_name, best_sim, best_margin

        # Fallback ROI tốn thêm 1–2 lần detect — tắt mặc định để tăng tốc
        if os.environ.get("ARCFACE_ROI_FALLBACK", "0") != "1":
            return best_sid, best_name, best_sim, best_margin

        rois = face_rois_from_person(work, app)
        for roi in rois:
            if roi is None or roi.size == 0:
                continue
            sid, name, sim, margin = self.match(roi)
            if sid is not None and sim > best_sim:
                best_sid, best_name, best_sim, best_margin = sid, name, sim, margin
                if sim >= self.lock_sim:
                    break
            elif sim > best_sim:
                best_sim, best_margin = sim, margin
        return best_sid, best_name, best_sim, best_margin


# Tương thích import cũ từ app.py
def build_gallery(force: bool = False) -> dict:
    if force or not GALLERY_NPZ.exists():
        from tools.export_arcface_model import export_gallery

        export_gallery(DATASET_DIR, MODEL_DIR)
    data = np.load(resolve_gallery_path(), allow_pickle=True)
    return {
        "ids": data["ids"].tolist(),
        "names": data["names"].tolist(),
        "embeddings": data["embeddings"].astype(np.float32),
    }


if __name__ == "__main__":
    build_gallery(force=True)
    r = FaceRecognizer(force_rebuild=False)
    print("ready", len(r.ids))
