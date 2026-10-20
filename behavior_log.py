"""Ghi hành vi từng học sinh theo thời gian và xuất file Excel.

Lưu gọn để chạy video dài nhiều giờ:
  - đợt hành vi: (track, hành vi) liên tiếp được gộp thành 1 dòng bắt đầu → kết thúc
  - thống kê theo track: số lần quan sát, điểm, điểm theo phút
Thời gian lưu bằng chỉ số khung, đổi ra ngày giờ lúc xuất qua VideoClock
(mốc OSD có thể đọc được sau khi đã ghi).
"""

from __future__ import annotations

import re
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from behavior import POSITIVE_BEHAVIORS as POSITIVE
from behavior import REASON_SHORT
from video_clock import VideoClock

BEHAVIORS = tuple(name for name, _short in REASON_SHORT)
SHORT_OF = dict(REASON_SHORT)
# Không thấy lại hành vi sau N giây video → kết thúc đợt
EPISODE_GAP_SEC = 3.0

_SECONDS_RE = re.compile(r"(\d+(?:\.\d+)?)s\)")


def _normalize(reason: str) -> tuple[str | None, float]:
    """'Cúi đầu lâu (3.2s)' → ('Cúi đầu lâu', 3.2)."""
    for name in BEHAVIORS:
        if reason.startswith(name):
            m = _SECONDS_RE.search(reason)
            return name, float(m.group(1)) if m else 0.0
    return None, 0.0


@dataclass
class _Episode:
    tid: int
    behavior: str
    start_frame: int
    last_frame: int
    start_wall: float
    last_wall: float
    min_score: int
    detail: str = ""


@dataclass
class _TrackStats:
    n_obs: int = 0
    sum_score: int = 0
    min_score: int = 100
    unfocused_obs: int = 0
    first_frame: int = 0
    last_frame: int = 0
    first_wall: float = 0.0
    last_wall: float = 0.0
    minute_sum: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    minute_n: dict[int, int] = field(default_factory=lambda: defaultdict(int))


class BehaviorRecorder:
    def __init__(self, fps: float, clock: VideoClock, source_label: str,
                 student_names: dict[str, str] | None = None,
                 legend_fn: Callable[[], list[dict]] | None = None):
        self.fps = fps
        self.clock = clock
        self.source_label = source_label
        self.legend_fn = legend_fn or (lambda: [])
        self.report_name = "bao_cao_hanh_vi.xlsx"
        self.student_names = student_names or {}
        self._lock = threading.Lock()
        self._open: dict[tuple[int, str], _Episode] = {}
        self._closed: list[_Episode] = []
        self._stats: dict[int, _TrackStats] = {}
        self._sid_votes: dict[int, Counter] = defaultdict(Counter)
        self._names: dict[str, str] = {}
        self.first_frame: int | None = None
        self.last_frame = 0
        self.first_wall = 0.0
        self.last_wall = 0.0

    def record(self, frame_idx: int, detections: list[dict], wall: float) -> None:
        gap_frames = EPISODE_GAP_SEC * self.fps
        with self._lock:
            if self.first_frame is None:
                self.first_frame, self.first_wall = frame_idx, wall
            self.last_frame, self.last_wall = frame_idx, wall
            minute = int((frame_idx - self.first_frame) / self.fps // 60)
            seen: set[tuple[int, str]] = set()
            for d in detections:
                tid = int(d["track_id"])
                if d.get("known") and d.get("student_id"):
                    self._sid_votes[tid][d["student_id"]] += 1
                    if d.get("name"):
                        self._names[d["student_id"]] = d["name"]
                score = d.get("score")
                if score is None:
                    continue
                st = self._stats.get(tid)
                if st is None:
                    st = self._stats[tid] = _TrackStats(
                        first_frame=frame_idx, first_wall=wall)
                st.n_obs += 1
                st.sum_score += score
                st.min_score = min(st.min_score, score)
                st.unfocused_obs += 0 if d.get("is_focused") else 1
                st.last_frame, st.last_wall = frame_idx, wall
                st.minute_sum[minute] += score
                st.minute_n[minute] += 1

                for reason in d.get("reasons") or []:
                    name, secs = _normalize(reason)
                    if name is None:
                        continue
                    key = (tid, name)
                    seen.add(key)
                    ep = self._open.get(key)
                    if ep is None:
                        back = int(round(secs * self.fps))
                        self._open[key] = _Episode(
                            tid, name, max(0, frame_idx - back), frame_idx,
                            wall - secs, wall, score, reason)
                    else:
                        ep.last_frame, ep.last_wall = frame_idx, wall
                        ep.min_score = min(ep.min_score, score)
                        ep.detail = reason
            for key in list(self._open):
                ep = self._open[key]
                if key not in seen and frame_idx - ep.last_frame > gap_frames:
                    self._closed.append(self._open.pop(key))

    # ---------------------------------------------------------------- xuất

    def _student_of(self, tid: int) -> str | None:
        votes = self._sid_votes.get(tid)
        return votes.most_common(1)[0][0] if votes else None

    def _who(self, tid: int) -> tuple[str, str]:
        sid = self._student_of(tid)
        if sid is None:
            return f"ID{tid} (chưa nhận diện)", ""
        name = self._names.get(sid) or self.student_names.get(sid) or ""
        return sid, "" if name == sid else name

    def _time_cells(self, frame_idx: int, wall: float) -> tuple[str, str, str]:
        """(ngày như video, giờ, vị trí trong video)."""
        dt = self.clock.at(frame_idx, wall)
        offset = str(timedelta(seconds=int(max(0, frame_idx) / self.fps)))
        if dt is None:
            return "", offset, offset
        return self.clock.format_date(dt), dt.strftime("%H:%M:%S"), offset

    def export_excel(self, path: Path) -> Path:
        with self._lock:
            wb = self._build_workbook()
        path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(path)
        return path

    def _build_workbook(self):
        """Gọi khi đang giữ self._lock (luồng phân tích vẫn ghi song song)."""
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        episodes = sorted(
            self._closed + list(self._open.values()), key=lambda e: e.start_frame
        )
        stats = self._stats
        first_frame = self.first_frame or 0
        last_frame, first_wall, last_wall = self.last_frame, self.first_wall, self.last_wall

        wb = Workbook()
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill("solid", fgColor="1F4E78")
        bad_fill = PatternFill("solid", fgColor="FDE2E1")
        good_fill = PatternFill("solid", fgColor="E2F5E6")

        def write_sheet(ws, headers, rows, widths):
            ws.append(headers)
            for c in ws[1]:
                c.font = header_font
                c.fill = header_fill
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for r in rows:
                ws.append(r)
            for i, w in enumerate(widths, start=1):
                ws.column_dimensions[get_column_letter(i)].width = w
            ws.freeze_panes = "A2"
            if rows:
                ws.auto_filter.ref = ws.dimensions

        # --- Chi tiết hành vi
        ws = wb.active
        ws.title = "Chi tiết hành vi"
        rows = []
        for i, ep in enumerate(episodes, start=1):
            sid, name = self._who(ep.tid)
            date_s, start_s, off_s = self._time_cells(ep.start_frame, ep.start_wall)
            _d, end_s, _o = self._time_cells(ep.last_frame, ep.last_wall)
            dur = round(max(0.0, (ep.last_frame - ep.start_frame) / self.fps), 1)
            rows.append([
                i, sid, name, ep.behavior, SHORT_OF.get(ep.behavior, ""),
                "Tích cực" if ep.behavior in POSITIVE else "Mất tập trung",
                date_s, start_s, end_s, dur, ep.min_score, off_s, ep.tid,
            ])
        write_sheet(
            ws,
            ["STT", "Mã HS", "Họ tên", "Hành vi", "Viết tắt (trên video)", "Loại",
             "Ngày (như video)", "Bắt đầu", "Kết thúc", "Thời lượng (giây)",
             "Điểm thấp nhất", "Vị trí trong video", "Track ID"],
            rows,
            [6, 22, 22, 20, 12, 15, 16, 11, 11, 12, 11, 14, 9],
        )
        for row in ws.iter_rows(min_row=2):
            row[5].fill = good_fill if row[5].value == "Tích cực" else bad_fill

        # --- Tổng hợp theo học sinh (gộp các track cùng 1 HS)
        per_student: dict[str, dict] = {}
        for tid, st in stats.items():
            sid, name = self._who(tid)
            agg = per_student.setdefault(sid, {
                "name": name, "n": 0, "sum": 0, "min": 100, "unf": 0,
                "first": (st.first_frame, st.first_wall), "last": (st.last_frame, st.last_wall),
                "count": Counter(), "secs": Counter(),
            })
            agg["n"] += st.n_obs
            agg["sum"] += st.sum_score
            agg["min"] = min(agg["min"], st.min_score)
            agg["unf"] += st.unfocused_obs
            if st.first_frame < agg["first"][0]:
                agg["first"] = (st.first_frame, st.first_wall)
            if st.last_frame > agg["last"][0]:
                agg["last"] = (st.last_frame, st.last_wall)
        for ep in episodes:
            sid, _name = self._who(ep.tid)
            agg = per_student.get(sid)
            if agg is None:
                continue
            agg["count"][ep.behavior] += 1
            agg["secs"][ep.behavior] += max(0.0, (ep.last_frame - ep.start_frame) / self.fps)

        ws2 = wb.create_sheet("Tổng hợp theo HS")
        headers = ["Mã HS", "Họ tên", "Ngày (như video)", "Xuất hiện từ", "Đến",
                   "Điểm TB", "Điểm thấp nhất", "% thời gian mất tập trung"]
        for b in BEHAVIORS:
            headers += [f"{b} (lần)", f"{b} (giây)"]
        rows2 = []
        for sid in sorted(per_student, key=lambda s: (s.startswith("ID"), s)):
            agg = per_student[sid]
            date_s, from_s, _o = self._time_cells(*agg["first"])
            _d, to_s, _o2 = self._time_cells(*agg["last"])
            row = [
                sid, agg["name"], date_s, from_s, to_s,
                round(agg["sum"] / agg["n"], 1) if agg["n"] else None,
                agg["min"],
                round(100.0 * agg["unf"] / agg["n"], 1) if agg["n"] else None,
            ]
            for b in BEHAVIORS:
                row += [agg["count"][b], round(agg["secs"][b], 1)]
            rows2.append(row)
        write_sheet(ws2, headers, rows2, [22, 22, 16, 12, 12, 9, 11, 14] + [11] * (2 * len(BEHAVIORS)))

        # --- Điểm theo phút
        ws3 = wb.create_sheet("Điểm theo phút")
        minute_scores: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        all_minutes: set[int] = set()
        for tid, st in stats.items():
            sid, _name = self._who(tid)
            for m, s in st.minute_sum.items():
                cell = minute_scores[sid][m]
                cell[0] += s
                cell[1] += st.minute_n[m]
                all_minutes.add(m)
        minutes = sorted(all_minutes)
        headers3 = ["Mã HS"]
        for m in minutes:
            f = first_frame + int(m * 60 * self.fps)
            _d, t_s, _o = self._time_cells(f, first_wall + m * 60)
            headers3.append(t_s)
        rows3 = []
        for sid in sorted(minute_scores, key=lambda s: (s.startswith("ID"), s)):
            row = [sid]
            for m in minutes:
                tot, n = minute_scores[sid].get(m, (0, 0))
                row.append(round(tot / n) if n else None)
            rows3.append(row)
        write_sheet(ws3, headers3, rows3, [22] + [10] * len(minutes))

        # --- Thông tin
        ws4 = wb.create_sheet("Thông tin")
        src = {
            "osd": "Đọc từ ngày giờ in trên video (OSD)",
            "system": "Đồng hồ máy (camera trực tiếp)",
            "video": "Không đọc được ngày giờ trên video — dùng vị trí trong video",
        }[self.clock.source]
        d1, t1, o1 = self._time_cells(first_frame, first_wall)
        d2, t2, o2 = self._time_cells(last_frame, last_wall)
        info = [
            ("Nguồn video", self.source_label),
            ("Nguồn thời gian", src),
            ("Bắt đầu ghi", f"{d1} {t1}".strip()),
            ("Kết thúc ghi", f"{d2} {t2}".strip()),
            ("Đoạn video đã phân tích", f"{o1} → {o2}"),
            ("Số đợt hành vi", len(episodes)),
            ("Số học sinh / track", len(per_student)),
            ("Ghi chú", "Điểm tập trung 0–100; dưới 50 = mất tập trung. "
                        "Một đợt kết thúc khi không thấy lại hành vi sau "
                        f"{EPISODE_GAP_SEC:.0f} giây video."),
        ]
        for k, v in info:
            ws4.append([k, v])
        for c in ws4["A"]:
            c.font = Font(bold=True)
        ws4.column_dimensions["A"].width = 26
        ws4.column_dimensions["B"].width = 90

        # --- Chú thích nhãn viết tắt
        ws5 = wb.create_sheet("Chú thích")
        rows5 = [
            [lg["short"], lg["name"], "Tích cực" if lg["positive"] else "Mất tập trung",
             lg["condition"], -lg["penalty"] if lg["penalty"] else 0,
             "Bật" if lg.get("enabled", True) else "Tắt"]
            for lg in self.legend_fn()
        ]
        write_sheet(
            ws5,
            ["Viết tắt (trên video)", "Hành vi đầy đủ", "Loại", "Điều kiện nhận biết", "Điểm trừ",
             "Trạng thái (lúc xuất)"],
            rows5,
            [16, 22, 15, 70, 10, 14],
        )
        off_fill = PatternFill("solid", fgColor="E7E6E6")
        for row in ws5.iter_rows(min_row=2):
            row[2].fill = good_fill if row[2].value == "Tích cực" else bad_fill
            if row[5].value == "Tắt":
                row[5].fill = off_fill
        ws5.append([])
        box_notes = [
            ("Khung xanh lá", "Tập trung (điểm ≥ 50)", "E2F5E6"),
            ("Khung đỏ", "Mất tập trung (điểm < 50); đỏ dày = cúi đầu lâu", "FDE2E1"),
            ("Khung xanh dương nhạt", "Chưa lấy được tư thế — chưa chấm điểm", "DDEBF7"),
            ("HS12 85 · Cúi", "Mã học sinh · điểm tập trung (0–100) · hành vi chính", None),
            ("ID7", "Chưa nhận diện được học sinh (số theo dõi tạm)", None),
        ]
        for label, meaning, color in box_notes:
            ws5.append([label, meaning])
            if color:
                ws5.cell(row=ws5.max_row, column=1).fill = PatternFill("solid", fgColor=color)
            ws5.cell(row=ws5.max_row, column=1).font = Font(bold=True)
        return wb
