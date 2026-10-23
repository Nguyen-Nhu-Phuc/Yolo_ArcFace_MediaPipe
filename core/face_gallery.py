"""
Nhận diện học sinh bằng ArcFace (InsightFace) + gallery đã XUẤT.

Quy trình đúng:
  1) python tools/export_arcface_model.py   ← xuất embedding từ dataset
  2) App load models/arcface_*/gallery.npz rồi so khớp cosine

Không train lại backbone ArcFace trên 33 ảnh — chỉ xuất gallery embeddings.
"""

from __future__ import annotations

import csv
import json
import os
import re
import threading
from pathlib import Path

import cv2
import numpy as np

from core.face_augment import face_rois_from_person

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(
    os.environ.get("CLASSROOM_FACES_DIR", str(BASE_DIR / "data" / "2_dataset_arcface"))
)
# Model đã xuất (ưu tiên). Bản cũ: models/arcface_classroom-faces-lop07/
MODEL_DIR = Path(
    os.environ.get(
        "ARCFACE_MODEL_DIR",
        str(BASE_DIR / "models" / f"arcface_{DATASET_DIR.name}"),
    )
)
GALLERY_NPZ = MODEL_DIR / "gallery.npz"
META_JSON = MODEL_DIR / "meta.json"
# Cache cũ (fallback)
LEGACY_NPZ = DATASET_DIR / "arcface_gallery.npz"
STUDENTS_CSV = DATASET_DIR / "students.csv"

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
# Các khóa ngưỡng trong meta.json — giữ nguyên khi dựng lại gallery / nạp HS mới
THRESHOLD_KEYS = (
    "match_threshold", "min_compat_pct", "min_margin", "lock_sim",
    "hard_lock_pct", "hard_lock_margin_pct",
)

ANGLE_WEIGHTS = {
    "front": 1.00,
    "down": 0.95,
    "up": 0.40,
    "left": 0.55,
    "right": 0.55,
}

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
    """Cosine similarity → % tương thích (0–100)."""
    return max(0.0, min(100.0, float(sim) * 100.0))


def load_student_classes() -> dict[str, str]:
    """Mã HS → lớp, đọc từ students.csv (thiếu file/cột thì trả rỗng)."""
    if not STUDENTS_CSV.exists():
        return {}
    with STUDENTS_CSV.open(encoding="utf-8-sig", newline="") as f:
        return {
            (row.get("student_id") or "").strip(): (row.get("class_name") or "").strip()
            for row in csv.DictReader(f)
            if (row.get("student_id") or "").strip()
        }


def get_face_app():
    """InsightFace buffalo_l (ArcFace + detector)."""
    global _face_app
    if _face_app is None:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        print("Nạp InsightFace ArcFace (buffalo_l)...")
        # Mặc định onnxruntime chiếm mọi nhân + spin-wait → YOLO/pose chạy song song bị chậm ~5x
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ARCFACE_THREADS", "2"))
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        app = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"], sess_options=so
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_app = app
        print("InsightFace sẵn sàng.")
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


def resolve_gallery_path() -> Path:
    if GALLERY_NPZ.exists():
        return GALLERY_NPZ
    if LEGACY_NPZ.exists():
        return LEGACY_NPZ
    raise FileNotFoundError(
        f"Chưa có gallery ArcFace.\n"
        f"Chạy trước: python tools/export_arcface_model.py --dataset {DATASET_DIR.name}\n"
        f"Kỳ vọng file: {GALLERY_NPZ}"
    )


class FaceRecognizer:
    """Load gallery đã xuất + so khớp ArcFace InsightFace."""

    def __init__(self, force_rebuild: bool = False):
        if force_rebuild:
            # Gọi lại exporter
            from tools.export_arcface_model import export_gallery

            export_gallery(DATASET_DIR, MODEL_DIR)

        _load_thresholds_from_meta()
        path = resolve_gallery_path()
        data = np.load(path, allow_pickle=True)
        self.ids: list[str] = data["ids"].tolist()
        self.names: list[str] = data["names"].tolist()
        self.embeddings: np.ndarray = data["embeddings"].astype(np.float32)
        # chuẩn hóa lại
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings = self.embeddings / norms
        self.gallery_path = path
        self.match_threshold = MATCH_THRESHOLD
        self.min_compat_pct = MIN_COMPAT_PCT
        self.min_margin = MIN_MARGIN
        self.lock_sim = LOCK_SIM
        self.hard_lock_sim = HARD_LOCK_PCT / 100.0
        self.hard_lock_margin = HARD_LOCK_MARGIN_PCT / 100.0

        # warmup
        get_face_app()
        print(
            f"Gallery ArcFace: {len(self.ids)} HS từ {path} | "
            f"thr={self.match_threshold:.2f} (≥{self.min_compat_pct:.0f}%) "
            f"margin={self.min_margin} | khóa cứng ≥{HARD_LOCK_PCT:.0f}% (+{HARD_LOCK_MARGIN_PCT:.0f}%)"
        )

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
        # Gán khi ≥ ngưỡng %; margin chỉ nới khi điểm còn sát ngưỡng
        if self.is_compatible(best_sim):
            strong = best_sim >= self.lock_sim or margin >= self.min_margin
            near_ok = best_sim >= (self.match_threshold + 0.04) or margin >= 0.0
            if strong or near_ok:
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
        if self.is_compatible(best_sim):
            strong = best_sim >= self.lock_sim or margin >= self.min_margin
            near_ok = best_sim >= (self.match_threshold + 0.04) or margin >= 0.0
            if strong or near_ok:
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
        for face in faces[:2]:
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

        # Fallback: zoom ROI đầu rồi embed lại
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

    def detect_and_match(
        self, frame_bgr: np.ndarray, max_faces: int = 5
    ) -> list[dict]:
        """Luồng webcam riêng: InsightFace detect + ArcFace match (không dùng YOLO)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        app = get_face_app()
        faces = app.get(frame_bgr) or []
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )[:max_faces]

        out: list[dict] = []
        for face in faces:
            x1, y1, x2, y2 = [float(v) for v in face.bbox]
            emb = np.asarray(face.normed_embedding, dtype=np.float32)
            sims = self.embeddings @ emb
            order = np.argsort(sims)[::-1]
            best_i = int(order[0])
            best_sim = float(sims[best_i])
            second = float(sims[order[1]]) if len(order) > 1 else 0.0
            margin = best_sim - second
            known = self.is_compatible(best_sim) and margin >= self.min_margin
            pct = round(sim_to_pct(best_sim), 1)
            out.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "known": known,
                    "student_id": self.ids[best_i] if known else None,
                    "name": self.names[best_i] if known else "Unknown",
                    "sim": round(best_sim, 4),
                    "match_pct": pct,
                    "margin": round(margin, 4),
                }
            )
        return out

    @staticmethod
    def _parse_hs_num(sid: str) -> int | None:
        """Quy tắc ID: HS001, HS002, … (chỉ chữ số sau HS)."""
        m = re.match(r"^HS(\d+)$", str(sid or "").strip(), re.I)
        return int(m.group(1)) if m else None

    def next_student_id(self) -> str:
        """ID tiếp theo theo quy tắc HS001…HS999 (liên tục từ max hiện có + 1)."""
        best = 0
        for sid in self.ids:
            n = self._parse_hs_num(sid)
            if n is not None:
                best = max(best, n)
        if STUDENTS_CSV.exists():
            with STUDENTS_CSV.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    n = self._parse_hs_num(row.get("student_id") or "")
                    if n is not None:
                        best = max(best, n)
        faces_root = DATASET_DIR / "faces"
        if faces_root.exists():
            for p in faces_root.rglob("HS*"):
                if p.is_dir():
                    n = self._parse_hs_num(p.name)
                    if n is not None:
                        best = max(best, n)
        return f"HS{best + 1:03d}"

    def list_students(self) -> list[dict]:
        return [
            {"student_id": sid, "full_name": name}
            for sid, name in zip(self.ids, self.names)
        ]

    def _save_gallery_files(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        emb = self.embeddings.astype(np.float32)
        payload = dict(
            ids=np.array(self.ids, dtype=object),
            names=np.array(self.names, dtype=object),
            embeddings=emb,
        )
        np.savez(GALLERY_NPZ, **payload)
        np.savez(LEGACY_NPZ, **payload)

        meta = read_meta()
        meta.update(
            {
                "backend": "insightface",
                "model": "buffalo_l",
                "task": "arcface_gallery",
                "dataset": str(DATASET_DIR.resolve()),
                "num_students": len(self.ids),
                "embedding_dim": int(emb.shape[1]) if len(emb) else 512,
                **self._threshold_meta(),
            }
        )
        _write_meta(meta)

    def _upsert_students_csv(
        self, student_id: str, full_name: str, class_name: str
    ) -> None:
        rows: list[dict] = []
        fieldnames = ["student_id", "full_name", "class_name"]
        if STUDENTS_CSV.exists():
            with STUDENTS_CSV.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    fieldnames = list(reader.fieldnames)
                    for fn in ("student_id", "full_name", "class_name"):
                        if fn not in fieldnames:
                            fieldnames.append(fn)
                rows = [dict(r) for r in reader]

        updated = False
        for row in rows:
            if (row.get("student_id") or "").strip() == student_id:
                row["full_name"] = full_name
                row["class_name"] = class_name
                updated = True
                break
        if not updated:
            rows.append(
                {
                    "student_id": student_id,
                    "full_name": full_name,
                    "class_name": class_name,
                }
            )

        STUDENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        with STUDENTS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def enroll_student(
        self,
        student_id: str,
        full_name: str,
        captures: dict[str, np.ndarray],
        class_name: str = "LOP07",
    ) -> dict:
        """
        Nạp / cập nhật 1 học sinh vào gallery (luồng riêng).
        captures: {angle: bgr_image} — angle in front/left/right/up/down
        """
        student_id = (student_id or "").strip().upper()
        full_name = (full_name or student_id).strip()
        class_name = (class_name or "LOP07").strip() or "LOP07"
        if not re.match(r"^HS\d{2,4}$", student_id, re.I):
            raise ValueError("student_id phải dạng HS001, HS082, …")
        if not captures:
            raise ValueError("Cần ít nhất 1 ảnh khuôn mặt.")

        weighted: list[tuple[np.ndarray, float]] = []
        used: list[str] = []
        saved_paths: list[str] = []

        enroll_dir = DATASET_DIR / "faces" / class_name / student_id / "enroll"
        enroll_dir.mkdir(parents=True, exist_ok=True)

        with _gallery_write_lock:
            for angle, img in captures.items():
                angle = (angle or "front").strip().lower()
                if angle not in ANGLE_WEIGHTS and angle != "other":
                    angle = "front"
                if img is None or img.size == 0:
                    continue
                emb = embed_bgr(img)
                if emb is None:
                    raise ValueError(
                        f"Không detect được mặt ở góc '{angle}'. Hãy chụp lại gần hơn / sáng hơn."
                    )
                w = ANGLE_WEIGHTS.get(angle, 0.3)
                weighted.append((emb, w))
                fname = f"{angle}_01.jpg"
                out_path = enroll_dir / fname
                if not cv2.imwrite(str(out_path), img):
                    raise RuntimeError(f"Không ghi được ảnh: {out_path}")
                used.append(fname)
                saved_paths.append(str(out_path.relative_to(BASE_DIR)))

            if not weighted:
                raise ValueError("Không tạo được embedding từ ảnh đã chụp.")

            total_w = sum(w for _, w in weighted)
            mean = sum(e * w for e, w in weighted) / total_w
            mean = (mean / (np.linalg.norm(mean) + 1e-8)).astype(np.float32)

            if student_id in self.ids:
                idx = self.ids.index(student_id)
                self.names[idx] = full_name
                self.embeddings[idx] = mean
                action = "updated"
            else:
                self.ids.append(student_id)
                self.names.append(full_name)
                if self.embeddings.size == 0:
                    self.embeddings = mean.reshape(1, -1)
                else:
                    self.embeddings = np.vstack([self.embeddings, mean])
                action = "added"

            self._upsert_students_csv(student_id, full_name, class_name)
            self._save_gallery_files()

        return {
            "action": action,
            "student_id": student_id,
            "full_name": full_name,
            "class_name": class_name,
            "images": used,
            "saved": saved_paths,
            "total_students": len(self.ids),
        }


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
