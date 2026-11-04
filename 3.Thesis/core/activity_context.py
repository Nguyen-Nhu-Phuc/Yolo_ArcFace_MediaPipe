"""Nạp ngữ cảnh hoạt động A1–A5 từ Excel/JSON; mặc định A1 khi thiếu dữ liệu."""

from __future__ import annotations

import json
from pathlib import Path

from core.ernu import UNCONFIRMED_ACTIVITY, VALID_ACTIVITIES, normalize_activity

DEFAULT_ACTIVITY = "A1"


def resolve_activity(t_sec: float, segments: list[dict] | None, default: str = DEFAULT_ACTIVITY) -> str:
    """Trả về mã A1–A5 tại thời điểm t_sec (giây video)."""
    default = normalize_activity(default, DEFAULT_ACTIVITY)
    if not segments:
        return default
    for seg in segments:
        try:
            s = float(seg["start_sec"])
            e = float(seg["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        code = normalize_activity(seg.get("activity_code"), default)
        if s <= t_sec < e or (t_sec == e and abs(t_sec - e) < 1e-9):
            return code
    return default


def _clean_segments(rows: list[dict], video_id: str | None = None) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        if video_id and str(row.get("video_id") or "").strip() not in ("", video_id):
            continue
        try:
            start = float(row["start_sec"])
            end = float(row["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        raw = str(row.get("activity_code") or "").strip().upper()
        if raw in VALID_ACTIVITIES:
            code = raw
        elif raw in (UNCONFIRMED_ACTIVITY, "UNKNOWN", "UNCONFIRMED", "?"):
            # Giữ đoạn chưa xác nhận nhiệm vụ — diễn giải không gán E*
            code = UNCONFIRMED_ACTIVITY
        elif not raw:
            continue
        else:
            continue
        task = row.get("teacher_task") or row.get("task_note") or ""
        out.append({
            "video_id": str(row.get("video_id") or "").strip(),
            "start_sec": start,
            "end_sec": end,
            "activity_code": code,
            "teacher_task": str(task),
            "task_note": str(task),
            "annotator_id": str(row.get("annotator_id") or ""),
            "notes": str(row.get("notes") or ""),
        })
    out.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
    return out


def load_activity_context(
    path: str | Path | None,
    video_id: str | None = None,
) -> list[dict]:
    """Đọc ACTIVITY_CONTEXT từ .json hoặc sheet Excel. Path rỗng/không tồn tại → []."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"Cảnh báo ERNU: không thấy ACTIVITY_CONTEXT tại {p}")
        return []
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            rows = data.get("ACTIVITY_CONTEXT") or data.get("segments") or data.get("rows") or []
        else:
            rows = data
        return _clean_segments(list(rows), video_id)
    if suffix in (".xlsx", ".xlsm"):
        return _load_excel(p, video_id)
    print(f"Cảnh báo ERNU: định dạng ACTIVITY_CONTEXT không hỗ trợ: {p.suffix}")
    return []


def _load_excel(path: Path, video_id: str | None) -> list[dict]:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    sheet_name = "ACTIVITY_CONTEXT"
    if sheet_name not in wb.sheetnames:
        # sheet đầu nếu chỉ có một bảng ngữ cảnh
        sheet_name = wb.sheetnames[0]
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(values_only=True)
    try:
        header = next(rows_iter)
    except StopIteration:
        return []
    keys = [str(h or "").strip() for h in header]
    rows = []
    for vals in rows_iter:
        row = {keys[i]: vals[i] for i in range(min(len(keys), len(vals)))}
        rows.append(row)
    return _clean_segments(rows, video_id)
