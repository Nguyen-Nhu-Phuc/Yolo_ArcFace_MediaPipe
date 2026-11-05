"""Gallery ArcFace lưu trên MongoDB Atlas. Thư mục tạm chỉ tồn tại lúc dựng embedding rồi bị xóa."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
from bson.binary import Binary

from core.db import enabled, get_db
from core.face_gallery import GALLERY_NPZ


def stored_ids() -> list[str]:
    raw = load_bytes()
    if not raw:
        return []
    data = np.load(io.BytesIO(raw), allow_pickle=True)
    if "ids" not in getattr(data, "files", []):
        return []
    return [str(x) for x in data["ids"].tolist()]


def load_bytes() -> bytes | None:
    if not enabled():
        return None
    db = get_db()
    doc = db.gallery.find_one({"_id": "current"}, {"npz": 1})
    if doc and doc.get("npz"):
        return bytes(doc["npz"])
    if GALLERY_NPZ.exists():
        raw = GALLERY_NPZ.read_bytes()
        db.gallery.replace_one({"_id": "current"}, {"_id": "current", "npz": Binary(raw)}, upsert=True)
        return raw
    return None


def save_bytes(raw: bytes) -> None:
    db = get_db()
    if db is None:
        raise RuntimeError("Chưa có MONGODB_URI")
    db.gallery.replace_one({"_id": "current"}, {"_id": "current", "npz": Binary(raw)}, upsert=True)


def rebuild_from_roster() -> bytes:
    """Dựng embedding từ ảnh trên Atlas và ghi lại gallery trên Atlas."""
    from core.mongo_roster import ensure_ready, photo_files, roster_rows
    from tools.export_arcface_model import export_gallery

    ensure_ready()
    rows = roster_rows()
    with tempfile.TemporaryDirectory(prefix="ernu-gallery-") as tmp:
        root = Path(tmp)
        faces = root / "faces"
        lines = ["student_id,full_name,class_name"]
        for row in rows:
            images = photo_files(row["student_id"])
            if not images:
                continue
            klass = row["class_name"] or "LOP"
            dest = faces / klass / row["student_id"] / "enroll"
            dest.mkdir(parents=True, exist_ok=True)
            for i, raw in enumerate(images, start=1):
                (dest / f"enroll_{i:02d}.jpg").write_bytes(raw)
            name = (row["full_name"] or row["student_id"]).replace(",", " ")
            lines.append(f"{row['student_id']},{name},{klass}")
        (root / "students.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not any(faces.rglob("*.jpg")):
            raise RuntimeError("Chưa có ảnh học sinh trên Atlas để dựng gallery.")
        export_gallery(root, root / "model")
        raw = (root / "model" / "gallery.npz").read_bytes()
    save_bytes(raw)
    return raw
