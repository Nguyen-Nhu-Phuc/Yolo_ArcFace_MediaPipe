"""Danh sách lớp và học sinh dùng cho điểm danh / gallery ArcFace.

Nguồn dữ liệu (không tạo kho riêng):
  danh_sach_hoc_sinh.csv  — họ tên, lớp, giới tính, ngày sinh
  students.csv            — cùng mã, để xuất gallery
  faces/<lớp>/<mã>/enroll — ảnh khuôn mặt
  classes.csv             — lớp đã tạo, kể cả lớp chưa có học sinh
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.face_gallery import (
    DATASET_DIR,
    GALLERY_NPZ,
    LEGACY_NPZ,
    ROSTER_CSV,
    STUDENTS_CSV,
    _npz_ids,
)

_lock = threading.Lock()

CLASS_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
STUDENT_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_PHOTO_BYTES = 8 * 1024 * 1024
MAX_PHOTOS = 8
GENDERS = {"", "Nam", "Nữ", "Khác"}

ROSTER_FIELDS = ["student_id", "full_name", "class_name", "gioi_tinh", "ngay_sinh"]
STUDENT_FIELDS = ["student_id", "full_name", "class_name"]
CLASS_FIELDS = ["class_id", "note"]


class RosterError(ValueError):
    """Dữ liệu lớp/học sinh không hợp lệ hoặc không xóa được."""


def classes_csv() -> Path:
    return DATASET_DIR / "classes.csv"


def faces_root() -> Path:
    return DATASET_DIR / "faces"


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    tmp.replace(path)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]


def _roster_rows() -> list[dict]:
    rows = _read_csv(ROSTER_CSV if ROSTER_CSV.exists() else STUDENTS_CSV)
    out = []
    for row in rows:
        sid = (row.get("student_id") or "").strip()
        if not sid:
            continue
        out.append({
            "student_id": sid,
            "full_name": (row.get("full_name") or sid).strip(),
            "class_name": (row.get("class_name") or "").strip(),
            "gioi_tinh": (row.get("gioi_tinh") or "").strip(),
            "ngay_sinh": (row.get("ngay_sinh") or "").strip(),
        })
    return out


def _sync_students_csv(rows: list[dict]) -> None:
    _write_csv(STUDENTS_CSV, STUDENT_FIELDS, rows)
    _write_csv(ROSTER_CSV, ROSTER_FIELDS, rows)


def _known_class_ids(rows: list[dict] | None = None) -> list[str]:
    rows = _roster_rows() if rows is None else rows
    found: list[str] = []
    seen: set[str] = set()

    def add(code: str) -> None:
        code = (code or "").strip()
        if code and code not in seen and CLASS_ID_RE.match(code):
            seen.add(code)
            found.append(code)

    for row in _read_csv(classes_csv()):
        add(row.get("class_id") or "")
    for row in rows:
        add(row.get("class_name") or "")
    root = faces_root()
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.is_dir() and not path.name.startswith("_"):
                add(path.name)
    return found


def _ensure_classes_file(rows: list[dict]) -> list[dict]:
    """Giữ classes.csv. Chỉ thêm mã lớp mới thấy trong danh sách hoặc thư mục ảnh."""
    path = classes_csv()
    merged = []
    seen: set[str] = set()
    for row in _read_csv(path):
        code = (row.get("class_id") or "").strip()
        if code and CLASS_ID_RE.match(code) and code not in seen:
            seen.add(code)
            merged.append({"class_id": code, "note": (row.get("note") or "").strip()})
    added = False
    for code in _known_class_ids(rows):
        if code not in seen:
            seen.add(code)
            merged.append({"class_id": code, "note": ""})
            added = True
    if not path.exists() or added:
        _write_csv(path, CLASS_FIELDS, merged)
    return merged


def _enroll_dir(class_name: str, student_id: str) -> Path:
    return faces_root() / class_name / student_id / "enroll"


def _count_enroll(class_name: str, student_id: str) -> int:
    folder = _enroll_dir(class_name, student_id)
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _gallery_ids() -> list[str]:
    ids = _npz_ids(GALLERY_NPZ)
    if ids:
        return ids
    return _npz_ids(LEGACY_NPZ) or []


def _next_student_id(class_name: str, rows: list[dict]) -> str:
    in_class = [r["student_id"] for r in rows if r["class_name"] == class_name]
    parsed = []
    for sid in in_class:
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*?)_(\d+)", sid)
        if match:
            parsed.append((match.group(1), int(match.group(2))))
    if parsed:
        prefix = Counter(p for p, _n in parsed).most_common(1)[0][0]
        number = max(n for p, n in parsed if p == prefix) + 1
        width = max(2, max(len(str(n)) for p, n in parsed if p == prefix))
        return f"{prefix}_{number:0{width}d}"
    taken = {r["student_id"] for r in rows}
    n = 1
    while True:
        sid = f"HS_{class_name}_{n:02d}"
        if sid not in taken:
            return sid
        n += 1


def _check_class_id(class_id: str) -> str:
    code = (class_id or "").strip()
    if not CLASS_ID_RE.match(code):
        raise RosterError("Mã lớp chỉ gồm chữ, số, gạch dưới, bắt đầu bằng chữ (vd. LOP08).")
    return code


def _check_student_fields(full_name: str, gioi_tinh: str, ngay_sinh: str) -> tuple[str, str, str]:
    name = " ".join((full_name or "").split())
    if not name or len(name) > 80:
        raise RosterError("Cần họ tên học sinh (tối đa 80 ký tự).")
    gender = (gioi_tinh or "").strip()
    if gender not in GENDERS:
        raise RosterError("Giới tính: Nam, Nữ, Khác, hoặc để trống.")
    born = (ngay_sinh or "").strip()
    if len(born) > 20:
        raise RosterError("Ngày sinh quá dài.")
    return name, gender, born


def _decode_photos(blobs: list[tuple[str, bytes]]) -> list[np.ndarray]:
    if len(blobs) > MAX_PHOTOS:
        raise RosterError(f"Tối đa {MAX_PHOTOS} ảnh mỗi lần.")
    images = []
    for _name, raw in blobs:
        if len(raw) > MAX_PHOTO_BYTES:
            raise RosterError("Mỗi ảnh tối đa 8 MB.")
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise RosterError("File không phải ảnh đọc được.")
        h, w = img.shape[:2]
        scale = 1280 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        images.append(img)
    return images


def _save_photo_bytes(dest_dir: Path, blobs: list[tuple[str, bytes]]) -> int:
    images = _decode_photos(blobs)
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in dest_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    start = len(existing) + 1
    for offset, img in enumerate(images):
        path = dest_dir / f"enroll_{start + offset:02d}.jpg"
        if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RosterError("Không lưu được ảnh khuôn mặt.")
    return len(images)


def _student_public(row: dict, gallery: set[str]) -> dict:
    sid = row["student_id"]
    photos = _count_enroll(row["class_name"], sid)
    return {
        **row,
        "enroll_count": photos,
        "in_gallery": sid in gallery,
        "photo_url": f"/api/admin/students/{sid}/photo" if photos else "",
    }


def list_classes() -> list[dict]:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import list_classes as listed
        return listed()
    with _lock:
        rows = _roster_rows()
        classes = _ensure_classes_file(rows)
        gallery = set(_gallery_ids())
        out = []
        for item in classes:
            members = [r for r in rows if r["class_name"] == item["class_id"]]
            with_photo = sum(1 for r in members if _count_enroll(r["class_name"], r["student_id"]) > 0)
            out.append({
                "class_id": item["class_id"],
                "note": item["note"],
                "students": len(members),
                "with_photo": with_photo,
                "missing_photo": len(members) - with_photo,
                "in_gallery": sum(1 for r in members if r["student_id"] in gallery),
            })
        return out


def list_students(class_id: str = "") -> list[dict]:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import list_students as listed
        return listed(class_id)
    with _lock:
        rows = _roster_rows()
        gallery = set(_gallery_ids())
        code = (class_id or "").strip()
        if code:
            rows = [r for r in rows if r["class_name"] == code]
        return [_student_public(r, gallery) for r in rows]


def overview(reports_dir: Path | None = None) -> dict:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import overview as listed
        return listed()
    classes = list_classes()
    students = list_students()
    gallery_ids = _gallery_ids()
    roster_ids = [s["student_id"] for s in students]
    missing_gallery = [s["student_id"] for s in students if s["enroll_count"] and not s["in_gallery"]]
    extra_gallery = [sid for sid in gallery_ids if sid not in set(roster_ids)]
    return {
        "dataset": str(DATASET_DIR),
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
    from core.db import enabled
    if enabled():
        from core.mongo_roster import add_class as impl
        return impl(class_id, note)
    code = _check_class_id(class_id)
    note = " ".join((note or "").split())
    if len(note) > 80:
        raise RosterError("Ghi chú lớp tối đa 80 ký tự.")
    with _lock:
        rows = _roster_rows()
        classes = _ensure_classes_file(rows)
        if any(c["class_id"].lower() == code.lower() for c in classes):
            raise RosterError(f"Lớp {code} đã có.")
        classes.append({"class_id": code, "note": note})
        _write_csv(classes_csv(), CLASS_FIELDS, classes)
        (faces_root() / code).mkdir(parents=True, exist_ok=True)
    return {"class_id": code, "note": note}


def delete_class(class_id: str) -> None:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import delete_class as impl
        return impl(class_id)
    code = _check_class_id(class_id)
    with _lock:
        rows = _roster_rows()
        if any(r["class_name"] == code for r in rows):
            raise RosterError("Lớp còn học sinh. Chuyển hoặc xóa học sinh trước.")
        classes = _ensure_classes_file(rows)
        if not any(c["class_id"] == code for c in classes):
            raise RosterError("Không thấy lớp.")
        _write_csv(classes_csv(), CLASS_FIELDS, [c for c in classes if c["class_id"] != code])
        folder = faces_root() / code
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def add_student(
    *,
    class_name: str,
    full_name: str,
    gioi_tinh: str = "",
    ngay_sinh: str = "",
    student_id: str = "",
    photos: list[tuple[str, bytes]] | None = None,
) -> dict:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import add_student as impl
        return impl(
            class_name=class_name, full_name=full_name, gioi_tinh=gioi_tinh,
            ngay_sinh=ngay_sinh, student_id=student_id, photos=photos,
        )
    code = _check_class_id(class_name)
    name, gender, born = _check_student_fields(full_name, gioi_tinh, ngay_sinh)
    photos = photos or []
    with _lock:
        rows = _roster_rows()
        classes = _ensure_classes_file(rows)
        if not any(c["class_id"] == code for c in classes):
            raise RosterError(f"Chưa có lớp {code}. Hãy tạo lớp trước.")
        sid = (student_id or "").strip() or _next_student_id(code, rows)
        if not STUDENT_ID_RE.match(sid):
            raise RosterError("Mã học sinh chỉ gồm chữ, số, gạch dưới.")
        if any(r["student_id"].lower() == sid.lower() for r in rows):
            raise RosterError(f"Mã {sid} đã tồn tại.")
        if photos:
            _save_photo_bytes(_enroll_dir(code, sid), photos)
        else:
            _enroll_dir(code, sid).mkdir(parents=True, exist_ok=True)
        rows.append({
            "student_id": sid,
            "full_name": name,
            "class_name": code,
            "gioi_tinh": gender,
            "ngay_sinh": born,
        })
        _sync_students_csv(rows)
        gallery = set(_gallery_ids())
        return _student_public(rows[-1], gallery)


def update_student(
    student_id: str,
    *,
    full_name: str | None = None,
    class_name: str | None = None,
    gioi_tinh: str | None = None,
    ngay_sinh: str | None = None,
) -> dict:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import update_student as impl
        return impl(
            student_id, full_name=full_name, class_name=class_name,
            gioi_tinh=gioi_tinh, ngay_sinh=ngay_sinh,
        )
    sid = (student_id or "").strip()
    with _lock:
        rows = _roster_rows()
        row = next((r for r in rows if r["student_id"] == sid), None)
        if row is None:
            raise RosterError("Không thấy học sinh.")
        name = row["full_name"] if full_name is None else full_name
        gender = row["gioi_tinh"] if gioi_tinh is None else gioi_tinh
        born = row["ngay_sinh"] if ngay_sinh is None else ngay_sinh
        name, gender, born = _check_student_fields(name, gender, born)
        new_class = row["class_name"] if class_name is None else _check_class_id(class_name)
        classes = _ensure_classes_file(rows)
        if not any(c["class_id"] == new_class for c in classes):
            raise RosterError(f"Chưa có lớp {new_class}.")
        old_class = row["class_name"]
        if new_class != old_class:
            src = faces_root() / old_class / sid
            dst = faces_root() / new_class / sid
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    raise RosterError("Thư mục ảnh ở lớp mới đã tồn tại.")
                shutil.move(str(src), str(dst))
        row["full_name"] = name
        row["class_name"] = new_class
        row["gioi_tinh"] = gender
        row["ngay_sinh"] = born
        _sync_students_csv(rows)
        return _student_public(row, set(_gallery_ids()))


def add_photos(student_id: str, photos: list[tuple[str, bytes]]) -> dict:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import add_photos as impl
        return impl(student_id, photos)
    if not photos:
        raise RosterError("Chưa chọn ảnh.")
    sid = (student_id or "").strip()
    with _lock:
        rows = _roster_rows()
        row = next((r for r in rows if r["student_id"] == sid), None)
        if row is None:
            raise RosterError("Không thấy học sinh.")
        _save_photo_bytes(_enroll_dir(row["class_name"], sid), photos)
        return _student_public(row, set(_gallery_ids()))


def delete_student(student_id: str) -> None:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import delete_student as impl
        return impl(student_id)
    sid = (student_id or "").strip()
    with _lock:
        rows = _roster_rows()
        row = next((r for r in rows if r["student_id"] == sid), None)
        if row is None:
            raise RosterError("Không thấy học sinh.")
        folder = faces_root() / row["class_name"] / sid
        if folder.exists():
            shutil.rmtree(folder)
        _sync_students_csv([r for r in rows if r["student_id"] != sid])


def first_photo_jpeg(student_id: str) -> bytes | None:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import first_photo_jpeg as impl
        return impl(student_id)
    sid = (student_id or "").strip()
    with _lock:
        rows = _roster_rows()
        row = next((r for r in rows if r["student_id"] == sid), None)
        if row is None:
            return None
        folder = _enroll_dir(row["class_name"], sid)
        if not folder.is_dir():
            return None
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            return None
        img = cv2.imread(str(images[0]))
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = 160 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None


def list_reports(page: int = 1, page_size: int = 8, reports_dir: Path | None = None) -> dict:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import list_reports as listed
        return listed(page, page_size)
    page = max(1, int(page or 1))
    page_size = min(40, max(1, int(page_size or 8)))
    files = []
    if reports_dir and reports_dir.is_dir():
        files = sorted(
            reports_dir.glob("bao_cao_hanh_vi_*.xlsx"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    total = len(files)
    pages = max(1, (total + page_size - 1) // page_size) if total else 1
    page = min(page, pages)
    start = (page - 1) * page_size
    items = []
    for path in files[start:start + page_size]:
        stats = _attendance_counts(path)
        items.append({
            "name": path.name,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "download_url": f"/api/admin/reports/{path.name}",
            **stats,
        })
    return {"items": items, "page": page, "page_size": page_size, "total": total, "pages": pages}


_REPORT_NAME = re.compile(r"bao_cao_hanh_vi_[0-9_]+_[0-9a-f]+\.xlsx$")


def delete_reports(names: list[str], reports_dir: Path | None = None) -> int:
    from core.db import enabled
    clean: list[str] = []
    for name in names or []:
        text = (name or "").strip()
        if not _REPORT_NAME.fullmatch(text):
            raise RosterError("Tên báo cáo không hợp lệ.")
        if text not in clean:
            clean.append(text)
    if not clean:
        raise RosterError("Chưa chọn báo cáo.")
    if len(clean) > 100:
        raise RosterError("Chỉ xóa tối đa 100 báo cáo mỗi lần.")
    if enabled():
        from core.mongo_roster import delete_reports as drop
        return drop(clean)
    if reports_dir is None:
        return 0
    root = reports_dir.resolve()
    deleted = 0
    for name in clean:
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            continue
        path.unlink()
        deleted += 1
    return deleted


def _recent_reports(reports_dir: Path, limit: int = 8) -> list[dict]:
    if not reports_dir.is_dir():
        return []
    files = sorted(reports_dir.glob("bao_cao_hanh_vi_*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for path in files[:limit]:
        stats = _attendance_counts(path)
        out.append({
            "name": path.name,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "download_url": f"/api/admin/reports/{path.name}",
            **stats,
        })
    return out


_xlsx_lock = threading.Lock()


def _attendance_counts(path: Path | None = None, data: bytes | None = None) -> dict:
    empty = {"present": None, "absent": None, "focus": None, "unfocus": None, "no_data": None, "rows": None}
    with _xlsx_lock:
        return _attendance_counts_locked(path, data, empty)


def _attendance_counts_locked(path: Path | None, data: bytes | None, empty: dict) -> dict:
    try:
        import openpyxl
        if data is not None:
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        else:
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception:
        return empty
    try:
        ws = wb["Điểm danh theo HS"] if "Điểm danh theo HS" in wb.sheetnames else wb.worksheets[0]
        rows = ws.iter_rows(values_only=True)
        header = None
        for row in rows:
            labels = {str(cell).strip() for cell in row if cell}
            if "Trạng thái" in labels and "Mã HS" in labels:
                header = row
                break
        if not header:
            return empty
        index = {str(cell).strip(): i for i, cell in enumerate(header) if cell}
        status_i = index.get("Trạng thái")
        verdict_i = index.get("Kết luận tập trung")
        present = absent = focus = unfocus = nodata = total = 0
        for row in rows:
            if not row or not any(row):
                continue
            total += 1
            status = str(row[status_i]).strip() if status_i is not None and status_i < len(row) and row[status_i] else ""
            verdict = str(row[verdict_i]).strip() if verdict_i is not None and verdict_i < len(row) and row[verdict_i] else ""
            if status == "Có mặt":
                present += 1
            elif status == "Vắng":
                absent += 1
            if verdict == "Tập trung":
                focus += 1
            elif verdict == "Không tập trung":
                unfocus += 1
            elif verdict == "Chưa đủ dữ liệu":
                nodata += 1
        return {
            "present": present, "absent": absent, "focus": focus,
            "unfocus": unfocus, "no_data": nodata, "rows": total,
        }
    except Exception:
        return empty
    finally:
        wb.close()
