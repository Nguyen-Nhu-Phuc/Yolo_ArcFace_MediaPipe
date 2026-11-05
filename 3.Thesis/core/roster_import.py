"""File Excel mẫu và nhập danh sách học sinh theo từng lớp."""

from __future__ import annotations

import io
import re
import unicodedata
from datetime import date, datetime

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from core.roster_store import (
    STUDENT_ID_RE,
    RosterError,
    _check_student_fields,
    _next_student_id,
    _xlsx_lock,
    add_student,
    list_classes,
    list_students,
    update_student,
)

_MAX_ROWS = 400
_MAX_BYTES = 2 * 1024 * 1024
_HEADERS = ("Mã học sinh", "Họ và tên", "Giới tính", "Ngày sinh")
_NAME_KEYS = {"ho va ten", "ho ten", "full name", "full_name", "ten hoc sinh"}
_ID_KEYS = {"ma hoc sinh", "ma hs", "student id", "student_id"}
_GENDER_KEYS = {"gioi tinh", "gender", "gioi_tinh"}
_BIRTH_KEYS = {"ngay sinh", "date of birth", "ngay_sinh"}
_GENDER_MAP = {
    "": "",
    "nam": "Nam",
    "male": "Nam",
    "m": "Nam",
    "nu": "Nữ",
    "female": "Nữ",
    "f": "Nữ",
    "khac": "Khác",
    "other": "Khác",
}


def _norm(value) -> str:
    text = str(value or "").strip().lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text)


def _cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%d/%m/%Y")
    if isinstance(value, date):
        return value.strftime("%d/%m/%Y")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _require_class(class_id: str) -> str:
    code = (class_id or "").strip()
    known = {row["class_id"] for row in list_classes()}
    if code not in known:
        raise RosterError(f"Chưa có lớp {code}. Hãy tạo lớp trước.")
    return code


def student_template_bytes(class_id: str = "") -> tuple[bytes, str]:
    """File .xlsx mẫu. Có mã lớp thì kèm học sinh hiện tại của lớp đó."""
    code = (class_id or "").strip()
    rows = []
    if code:
        code = _require_class(code)
        rows = [s for s in list_students(code) if s["class_name"] == code]
        rows.sort(key=lambda s: s["student_id"])
    book = Workbook()
    sheet = book.active
    sheet.title = "Danh_sach"
    sheet.append(["Lớp", code])
    sheet.append(list(_HEADERS))
    for student in rows:
        sheet.append([
            student["student_id"],
            student["full_name"],
            student.get("gioi_tinh") or "",
            student.get("ngay_sinh") or "",
        ])
    header_fill = PatternFill("solid", fgColor="1D4ED8")
    header_font = Font(bold=True, color="FFFFFF")
    for cell in sheet[2]:
        cell.fill = header_fill
        cell.font = header_font
    sheet.column_dimensions["A"].width = 18
    sheet.column_dimensions["B"].width = 32
    sheet.column_dimensions["C"].width = 14
    sheet.column_dimensions["D"].width = 16
    sheet.freeze_panes = "A3"
    sheet.auto_filter.ref = f"A2:D{max(2, 2 + len(rows))}"
    gender = DataValidation(type="list", formula1='"Nam,Nữ,Khác"', allow_blank=True)
    gender.error = "Giới tính: Nam, Nữ, Khác, hoặc để trống."
    gender.errorTitle = "Giới tính"
    sheet.add_data_validation(gender)
    gender.add("C3:C402")

    guide = book.create_sheet("Huong_dan")
    guide.column_dimensions["A"].width = 96
    for line in (
        "Điền học sinh vào sheet Danh_sach, ngay dưới dòng tiêu đề.",
        "Họ và tên là bắt buộc. Mã học sinh để trống thì hệ thống tự đánh số tiếp.",
        "Giới tính: Nam, Nữ, Khác, hoặc để trống. Ngày sinh dạng 15/03/2013.",
        "Giữ nguyên ô Lớp ở dòng đầu. Khi nhập lại, mã đã có trong lớp được cập nhật.",
        "Ảnh khuôn mặt không nằm trong file này. Thêm ảnh trên trang Quản lý lớp.",
    ):
        guide.append([line])

    buf = io.BytesIO()
    with _xlsx_lock:
        book.save(buf)
    filename = f"mau_danh_sach_{code}.xlsx" if code else "mau_danh_sach_hoc_sinh.xlsx"
    return buf.getvalue(), filename


def _parse_sheet(content: bytes, class_id: str) -> list[dict]:
    if not content:
        raise RosterError("Chưa chọn file.")
    if len(content) > _MAX_BYTES:
        raise RosterError("File quá lớn (tối đa 2 MB).")
    if not content.startswith(b"PK"):
        raise RosterError("Chỉ nhận file Excel .xlsx.")
    try:
        with _xlsx_lock:
            book = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    except RosterError:
        raise
    except Exception as exc:
        raise RosterError("Không đọc được file Excel.") from exc
    try:
        sheet = book["Danh_sach"] if "Danh_sach" in book.sheetnames else book[book.sheetnames[0]]
        grid = [tuple(row) for row in sheet.iter_rows(max_row=_MAX_ROWS + 5, max_col=8, values_only=True)]
    finally:
        book.close()

    file_class = ""
    header_at = None
    columns: dict[str, int] = {}
    for index, row in enumerate(grid[:20]):
        labels = [_norm(cell) for cell in row]
        if labels and labels[0] in {"lop", "class", "ma lop"} and len(row) > 1 and _cell_text(row[1]):
            file_class = _cell_text(row[1])
            continue
        if any(label in _NAME_KEYS for label in labels):
            header_at = index
            for col, label in enumerate(labels):
                if label in _ID_KEYS:
                    columns["student_id"] = col
                elif label in _NAME_KEYS:
                    columns["full_name"] = col
                elif label in _GENDER_KEYS:
                    columns["gioi_tinh"] = col
                elif label in _BIRTH_KEYS:
                    columns["ngay_sinh"] = col
            break
    if header_at is None or "full_name" not in columns:
        raise RosterError("Không thấy cột Họ và tên. Hãy dùng file mẫu.")
    if file_class and file_class != class_id:
        raise RosterError(f"File này dành cho lớp {file_class}, không phải {class_id}.")

    parsed = []
    for offset, row in enumerate(grid[header_at + 1:], start=header_at + 2):
        if not row or not any(_cell_text(cell) for cell in row):
            continue

        def take(key: str) -> str:
            col = columns.get(key)
            if col is None or col >= len(row):
                return ""
            return _cell_text(row[col])

        parsed.append({
            "row": offset,
            "student_id": take("student_id"),
            "full_name": take("full_name"),
            "gioi_tinh": take("gioi_tinh"),
            "ngay_sinh": take("ngay_sinh"),
        })
        if len(parsed) > _MAX_ROWS:
            raise RosterError(f"Tối đa {_MAX_ROWS} học sinh mỗi lần nhập.")
    if not parsed:
        raise RosterError("File không có dòng học sinh nào.")
    return parsed


def import_student_rows(class_id: str, content: bytes) -> dict:
    """Thêm học sinh mới và cập nhật mã đã có trong đúng lớp. Không ghi nếu còn dòng lỗi."""
    code = _require_class(class_id)
    parsed = _parse_sheet(content, code)
    existing = {row["student_id"].lower(): row for row in list_students()}
    planned_rows = [
        {"student_id": row["student_id"], "class_name": row["class_name"]}
        for row in existing.values()
    ]
    seen: set[str] = set()
    actions = []
    errors = []
    for item in parsed:
        where = f"Dòng {item['row']}"
        gender_key = _norm(item["gioi_tinh"])
        if gender_key not in _GENDER_MAP:
            errors.append(f"{where}: giới tính chỉ nhận Nam, Nữ, Khác hoặc để trống.")
            continue
        try:
            name, gender, born = _check_student_fields(
                item["full_name"], _GENDER_MAP[gender_key], item["ngay_sinh"],
            )
        except RosterError as exc:
            errors.append(f"{where}: {exc}")
            continue
        sid = item["student_id"]
        if sid:
            if not STUDENT_ID_RE.match(sid):
                errors.append(f"{where}: mã học sinh chỉ gồm chữ, số, gạch dưới, bắt đầu bằng chữ.")
                continue
            key = sid.lower()
            if key in seen:
                errors.append(f"{where}: mã {sid} bị lặp trong file.")
                continue
            seen.add(key)
            current = existing.get(key)
            if current and current["class_name"] != code:
                errors.append(f"{where}: mã {sid} đang thuộc lớp {current['class_name']}.")
                continue
            if current is None:
                actions.append(("create", sid, name, gender, born))
                planned_rows.append({"student_id": sid, "class_name": code})
            elif (
                current["full_name"] == name
                and (current.get("gioi_tinh") or "") == gender
                and (current.get("ngay_sinh") or "") == born
            ):
                actions.append(("same", sid, name, gender, born))
            else:
                actions.append(("update", current["student_id"], name, gender, born))
        else:
            sid = _next_student_id(code, planned_rows)
            planned_rows.append({"student_id": sid, "class_name": code})
            seen.add(sid.lower())
            actions.append(("create", sid, name, gender, born))

    if errors:
        shown = "; ".join(errors[:8])
        extra = f" (+{len(errors) - 8} lỗi nữa)" if len(errors) > 8 else ""
        raise RosterError(shown + extra)

    created = updated = unchanged = 0
    for kind, sid, name, gender, born in actions:
        if kind == "create":
            add_student(
                class_name=code, full_name=name, gioi_tinh=gender,
                ngay_sinh=born, student_id=sid,
            )
            created += 1
        elif kind == "update":
            update_student(sid, full_name=name, gioi_tinh=gender, ngay_sinh=born)
            updated += 1
        else:
            unchanged += 1
    return {
        "class_id": code,
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
    }
