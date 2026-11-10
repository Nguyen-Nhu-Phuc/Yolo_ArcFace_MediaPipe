"""Lớp, học sinh, ảnh nhận diện và báo cáo Excel trên MongoDB Atlas."""

from __future__ import annotations

import io
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from bson.binary import Binary

from core.db import get_db
from core.roster_store import (
    IMAGE_SUFFIXES,
    MAX_PHOTOS,
    RosterError,
    _attendance_counts,
    _xlsx_lock,
    _check_class_id,
    _check_student_fields,
    _decode_photos,
    _enroll_dir,
    _next_student_id,
    _read_csv,
    _roster_rows,
    classes_csv,
)

_ready = False
_ready_lock = threading.Lock()
_thumb_cache: dict[str, bytes] = {}
_thumb_lock = threading.Lock()
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def ensure_ready() -> None:
    global _ready
    with _ready_lock:
        if _ready:
            return
        db = get_db()
        if db is None:
            raise RosterError("Chưa cấu hình MONGODB_URI trong file .env")
        try:
            db.command("ping")
        except Exception as exc:  # noqa: BLE001
            raise RosterError("Không kết nối được MongoDB Atlas. Kiểm tra MONGODB_URI.") from exc
        db.students.create_index("student_id", unique=True)
        db.classes.create_index("class_id", unique=True)
        db.photos.create_index([("student_id", 1), ("seq", 1)])
        db.reports.create_index("name", unique=True)
        if db.students.count_documents({}) == 0:
            _import_local(db)
        if db.reports.count_documents({}) == 0:
            _import_reports(db)
        _ready = True


def roster_rows() -> list[dict]:
    """Danh sách cho điểm danh và số trên khung. Không đọc CSV."""
    ensure_ready()
    db = get_db()
    rows = []
    for doc in db.students.find({}, {"_id": 0}).sort("student_id", 1):
        rows.append({
            "student_id": doc.get("student_id") or "",
            "full_name": doc.get("full_name") or "",
            "class_name": doc.get("class_name") or "",
            "gioi_tinh": doc.get("gioi_tinh") or "",
            "ngay_sinh": doc.get("ngay_sinh") or "",
        })
    return rows


def _photo_counts() -> dict[str, int]:
    db = get_db()
    counts: dict[str, int] = {}
    for row in db.photos.aggregate([{"$group": {"_id": "$student_id", "n": {"$sum": 1}}}]):
        if row.get("_id"):
            counts[str(row["_id"])] = int(row["n"])
    return counts


def _gallery_ids() -> set[str]:
    from core.gallery_store import stored_ids
    return set(stored_ids())


def _public(row: dict, photos: int, gallery: set[str]) -> dict:
    sid = row["student_id"]
    return {
        **row,
        "enroll_count": photos,
        "in_gallery": sid in gallery,
        "photo_url": f"/api/admin/students/{sid}/photo" if photos else "",
    }


def list_classes() -> list[dict]:
    ensure_ready()
    db = get_db()
    students = roster_rows()
    counts = _photo_counts()
    gallery = _gallery_ids()
    notes = {d["class_id"]: d.get("note") or "" for d in db.classes.find({}, {"_id": 0})}
    codes = list(notes)
    for row in students:
        if row["class_name"] and row["class_name"] not in notes:
            codes.append(row["class_name"])
            notes[row["class_name"]] = ""
    out = []
    for code in codes:
        members = [r for r in students if r["class_name"] == code]
        with_photo = sum(1 for r in members if counts.get(r["student_id"], 0) > 0)
        out.append({
            "class_id": code,
            "note": notes.get(code, ""),
            "students": len(members),
            "with_photo": with_photo,
            "missing_photo": len(members) - with_photo,
            "in_gallery": sum(1 for r in members if r["student_id"] in gallery),
        })
    return out


def list_students(class_id: str = "") -> list[dict]:
    ensure_ready()
    rows = roster_rows()
    code = (class_id or "").strip()
    if code:
        rows = [r for r in rows if r["class_name"] == code]
    counts = _photo_counts()
    gallery = _gallery_ids()
    return [_public(r, counts.get(r["student_id"], 0), gallery) for r in rows]


def overview() -> dict:
    from core.db import database_name
    classes = list_classes()
    students = list_students()
    gallery_ids = _gallery_ids()
    roster_ids = {s["student_id"] for s in students}
    missing_gallery = [s["student_id"] for s in students if s["enroll_count"] and not s["in_gallery"]]
    extra_gallery = [sid for sid in gallery_ids if sid not in roster_ids]
    return {
        "storage": "MongoDB Atlas",
        "database": database_name(),
        "classes": len(classes),
        "students": len(students),
        "with_photo": sum(1 for s in students if s["enroll_count"]),
        "missing_photo": sum(1 for s in students if not s["enroll_count"]),
        "enroll_images": sum(s["enroll_count"] for s in students),
        "gallery_size": len(gallery_ids),
        "missing_from_gallery": missing_gallery,
        "extra_in_gallery": extra_gallery,
        "gallery_ready": not missing_gallery and not extra_gallery,
        "by_class": classes,
    }


def add_class(class_id: str, note: str = "") -> dict:
    ensure_ready()
    code = _check_class_id(class_id)
    note = " ".join((note or "").split())
    if len(note) > 80:
        raise RosterError("Ghi chú lớp tối đa 80 ký tự.")
    db = get_db()
    if db.classes.find_one({"class_id": {"$regex": f"^{code}$", "$options": "i"}}):
        raise RosterError(f"Lớp {code} đã có.")
    db.classes.insert_one({"class_id": code, "note": note})
    return {"class_id": code, "note": note}


def delete_class(class_id: str) -> None:
    ensure_ready()
    code = _check_class_id(class_id)
    db = get_db()
    if db.students.count_documents({"class_name": code}):
        raise RosterError("Lớp còn học sinh. Chuyển hoặc xóa học sinh trước.")
    result = db.classes.delete_one({"class_id": code})
    if result.deleted_count == 0:
        raise RosterError("Không thấy lớp.")


def _insert_photos(student_id: str, images: list[np.ndarray]) -> None:
    if not images:
        return
    db = get_db()
    last = db.photos.find_one({"student_id": student_id}, sort=[("seq", -1)])
    seq = int(last["seq"]) + 1 if last else 1
    docs = []
    for offset, img in enumerate(images):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RosterError("Không lưu được ảnh khuôn mặt.")
        docs.append({"student_id": student_id, "seq": seq + offset, "image": Binary(buf.tobytes())})
    if len(docs) > MAX_PHOTOS:
        raise RosterError(f"Tối đa {MAX_PHOTOS} ảnh mỗi lần.")
    db.photos.insert_many(docs)


def add_student(
    *,
    class_name: str,
    full_name: str,
    gioi_tinh: str = "",
    ngay_sinh: str = "",
    student_id: str = "",
    photos: list[tuple[str, bytes]] | None = None,
) -> dict:
    ensure_ready()
    code = _check_class_id(class_name)
    name, gender, born = _check_student_fields(full_name, gioi_tinh, ngay_sinh)
    images = _decode_photos(photos or [])
    db = get_db()
    if not db.classes.find_one({"class_id": code}):
        raise RosterError(f"Chưa có lớp {code}. Hãy tạo lớp trước.")
    rows = roster_rows()
    from core.roster_store import STUDENT_ID_RE
    sid = (student_id or "").strip() or _next_student_id(code, rows)
    if not STUDENT_ID_RE.match(sid):
        raise RosterError("Mã học sinh chỉ gồm chữ, số, gạch dưới.")
    if any(r["student_id"].lower() == sid.lower() for r in rows):
        raise RosterError(f"Mã {sid} đã tồn tại.")
    db.students.insert_one({
        "student_id": sid,
        "full_name": name,
        "class_name": code,
        "gioi_tinh": gender,
        "ngay_sinh": born,
    })
    if images:
        _insert_photos(sid, images)
    gallery = _gallery_ids()
    return _public({
        "student_id": sid, "full_name": name, "class_name": code,
        "gioi_tinh": gender, "ngay_sinh": born,
    }, len(images), gallery)


def update_student(
    student_id: str,
    *,
    full_name: str | None = None,
    class_name: str | None = None,
    gioi_tinh: str | None = None,
    ngay_sinh: str | None = None,
) -> dict:
    ensure_ready()
    sid = (student_id or "").strip()
    db = get_db()
    row = db.students.find_one({"student_id": sid}, {"_id": 0})
    if row is None:
        raise RosterError("Không thấy học sinh.")
    name, gender, born = _check_student_fields(
        row["full_name"] if full_name is None else full_name,
        row.get("gioi_tinh") or "" if gioi_tinh is None else gioi_tinh,
        row.get("ngay_sinh") or "" if ngay_sinh is None else ngay_sinh,
    )
    new_class = row["class_name"] if class_name is None else _check_class_id(class_name)
    if not db.classes.find_one({"class_id": new_class}):
        raise RosterError(f"Chưa có lớp {new_class}.")
    db.students.update_one(
        {"student_id": sid},
        {"$set": {"full_name": name, "class_name": new_class, "gioi_tinh": gender, "ngay_sinh": born}},
    )
    photos = get_db().photos.count_documents({"student_id": sid})
    return _public({
        "student_id": sid, "full_name": name, "class_name": new_class,
        "gioi_tinh": gender, "ngay_sinh": born,
    }, photos, _gallery_ids())


def add_photos(student_id: str, photos: list[tuple[str, bytes]]) -> dict:
    ensure_ready()
    if not photos:
        raise RosterError("Chưa chọn ảnh.")
    sid = (student_id or "").strip()
    db = get_db()
    row = db.students.find_one({"student_id": sid}, {"_id": 0})
    if row is None:
        raise RosterError("Không thấy học sinh.")
    _insert_photos(sid, _decode_photos(photos))
    _drop_thumb(sid)
    count = db.photos.count_documents({"student_id": sid})
    return _public({
        "student_id": sid,
        "full_name": row.get("full_name") or "",
        "class_name": row.get("class_name") or "",
        "gioi_tinh": row.get("gioi_tinh") or "",
        "ngay_sinh": row.get("ngay_sinh") or "",
    }, count, _gallery_ids())


def delete_student(student_id: str) -> None:
    ensure_ready()
    sid = (student_id or "").strip()
    db = get_db()
    result = db.students.delete_one({"student_id": sid})
    if result.deleted_count == 0:
        raise RosterError("Không thấy học sinh.")
    db.photos.delete_many({"student_id": sid})
    _drop_thumb(sid)


def _drop_thumb(student_id: str) -> None:
    with _thumb_lock:
        _thumb_cache.pop((student_id or "").strip(), None)


def first_photo_jpeg(student_id: str) -> bytes | None:
    ensure_ready()
    sid = (student_id or "").strip()
    with _thumb_lock:
        cached = _thumb_cache.get(sid)
    if cached is not None:
        return cached
    doc = get_db().photos.find_one({"student_id": sid}, sort=[("seq", 1)])
    if not doc or not doc.get("image"):
        return None
    raw = bytes(doc["image"])
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        with _thumb_lock:
            _thumb_cache[sid] = raw
        return raw
    h, w = img.shape[:2]
    scale = 160 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpeg = buf.tobytes() if ok else raw
    with _thumb_lock:
        _thumb_cache[sid] = jpeg
    return jpeg


def photo_files(student_id: str) -> list[bytes]:
    """Ảnh gốc để dựng gallery. Gọi khi đã ensure_ready."""
    docs = get_db().photos.find({"student_id": student_id}).sort("seq", 1)
    return [bytes(doc["image"]) for doc in docs if doc.get("image")]


def save_report_bytes(name: str, data: bytes) -> None:
    ensure_ready()
    get_db().reports.replace_one(
        {"name": name},
        {"name": name, "content": Binary(data), "modified": datetime.now().strftime("%Y-%m-%d %H:%M")},
        upsert=True,
    )


def get_report_bytes(name: str) -> bytes | None:
    ensure_ready()
    doc = get_db().reports.find_one({"name": name}, {"content": 1})
    if not doc or not doc.get("content"):
        return None
    return bytes(doc["content"])


def preview_report(name: str) -> dict:
    """Đọc workbook trên Atlas thành bảng để xem trên trang quản lý."""
    import openpyxl

    raw_file = get_report_bytes(name)
    if not raw_file:
        raise RosterError("Không thấy báo cáo")
    with _xlsx_lock:
        return _preview_report_locked(name, raw_file, openpyxl)


# Preview admin chỉ hiện điểm danh + kết luận (báo cáo cũ vẫn còn sheet kỹ thuật).
_PREVIEW_SHEETS = (
    "Điểm danh theo HS",
    "Kết luận tập trung",
    "KET_LUAN_TAP_TRUNG",
)
_ATTENDANCE_COLS = (
    "STT", "Mã HS", "Họ và tên", "Lớp", "Trạng thái",
    "Kết luận tập trung", "Điểm tập trung",
    "Ngày (như video)", "Giờ điểm danh (lần đầu nhận diện)",
    "Lần cuối thấy", "Thời gian có mặt (phút)",
    "Mức tin cậy", "Ảnh khuôn mặt (trong lớp)", "Ghi chú (giáo viên)",
)
_FOCUS_COLS = ("STT", "Mã HS", "Họ và tên", "Kết luận", "Điểm")
_SUMMARY_HIDE = (
    "Tín hiệu", "Không xác định", "Track chưa", "Quy tắc điểm danh", "B1/", "B4/", "pose",
)


def _preview_report_locked(name: str, raw_file: bytes, openpyxl) -> dict:
    wb = openpyxl.load_workbook(io.BytesIO(raw_file), read_only=True, data_only=True)
    sheets = []
    try:
        for ws in wb.worksheets:
            if ws.title not in _PREVIEW_SHEETS:
                continue
            raw_rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i > 120:
                    break
                if not row or not any(cell is not None and str(cell).strip() for cell in row):
                    continue
                raw_rows.append(["" if cell is None else str(cell).strip() for cell in row])
            if not raw_rows:
                continue
            header_idx = 0
            for i, row in enumerate(raw_rows):
                if "Mã HS" in row or "student_id" in row:
                    header_idx = i
                    break
            headers = list(raw_rows[header_idx])
            while headers and headers[-1] == "":
                headers.pop()
            summary = []
            for row in raw_rows[:header_idx]:
                label = row[0] if row else ""
                value = row[3] if len(row) > 3 and row[3] else (row[1] if len(row) > 1 else "")
                if label and not label.startswith("BÁO CÁO"):
                    if any(k in label for k in _SUMMARY_HIDE):
                        continue
                    summary.append({"label": label, "value": value})
            keep = None
            if ws.title == "Điểm danh theo HS":
                keep = _ATTENDANCE_COLS
            elif ws.title in ("Kết luận tập trung", "KET_LUAN_TAP_TRUNG"):
                keep = _FOCUS_COLS
            if keep:
                idxs = [i for i, h in enumerate(headers) if h in keep]
                if idxs:
                    headers = [headers[i] for i in idxs]
                    raw_rows = [
                        *raw_rows[: header_idx + 1],
                        *[
                            [row[j] if j < len(row) else "" for j in idxs]
                            for row in raw_rows[header_idx + 1:]
                        ],
                    ]
            width = len(headers) or 1
            body = [
                row[:width] + [""] * max(0, width - len(row))
                for row in raw_rows[header_idx + 1:]
            ]
            display_name = (
                "Kết luận tập trung" if ws.title == "KET_LUAN_TAP_TRUNG" else ws.title
            )
            sheets.append({
                "name": display_name,
                "summary": summary[:20],
                "headers": headers,
                "rows": body[:80],
            })
    finally:
        wb.close()
    doc = get_db().reports.find_one({"name": name}, {"modified": 1})
    return {
        "name": name,
        "modified": (doc or {}).get("modified") or "",
        "download_url": f"/api/admin/reports/{name}",
        "sheets": sheets,
    }


def list_reports(page: int = 1, page_size: int = 8) -> dict:
    """Một trang báo cáo. Chỉ đọc Excel của trang đang xem."""
    ensure_ready()
    page = max(1, int(page or 1))
    page_size = min(40, max(1, int(page_size or 8)))
    db = get_db()
    total = db.reports.count_documents({})
    pages = max(1, (total + page_size - 1) // page_size) if total else 1
    page = min(page, pages)
    cursor = (
        db.reports.find({}, {"content": 1, "name": 1, "modified": 1})
        .sort([("modified", -1), ("name", -1)])
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    items = []
    for doc in cursor:
        raw = bytes(doc.get("content") or b"")
        stats = _attendance_counts(data=raw) if raw else {}
        name = doc.get("name") or ""
        items.append({
            "name": name,
            "modified": doc.get("modified") or "",
            "download_url": f"/api/admin/reports/{name}",
            **stats,
        })
    return {"items": items, "page": page, "page_size": page_size, "total": total, "pages": pages}


def delete_reports(names: list[str]) -> int:
    ensure_ready()
    if not names:
        return 0
    result = get_db().reports.delete_many({"name": {"$in": list(names)}})
    return int(result.deleted_count)


def _import_local(db) -> None:
    rows = _roster_rows()
    classes = []
    seen: set[str] = set()
    for row in _read_csv(classes_csv()):
        code = (row.get("class_id") or "").strip()
        if code and code not in seen:
            seen.add(code)
            classes.append({"class_id": code, "note": (row.get("note") or "").strip()})
    for row in rows:
        code = row["class_name"]
        if code and code not in seen:
            seen.add(code)
            classes.append({"class_id": code, "note": ""})
    if classes:
        db.classes.insert_many(classes)
    if rows:
        db.students.insert_many(rows)
    photo_docs = []
    for row in rows:
        folder = _enroll_dir(row["class_name"], row["student_id"])
        if not folder.is_dir():
            continue
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for seq, path in enumerate(images, start=1):
            photo_docs.append({
                "student_id": row["student_id"],
                "seq": seq,
                "image": Binary(path.read_bytes()),
            })
    if photo_docs:
        db.photos.insert_many(photo_docs)


def _import_reports(db) -> None:
    if not REPORTS_DIR.is_dir():
        return
    docs = []
    for path in sorted(REPORTS_DIR.glob("bao_cao_hanh_vi_*.xlsx")):
        docs.append({
            "name": path.name,
            "content": Binary(path.read_bytes()),
            "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    if docs:
        db.reports.insert_many(docs)
