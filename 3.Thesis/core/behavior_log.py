"""Ghi hành vi từng học sinh theo thời gian và xuất file Excel.

Lưu gọn để chạy video dài nhiều giờ:
  - đợt hành vi: (track, hành vi) liên tiếp được gộp thành 1 dòng bắt đầu → kết thúc
  - thống kê theo track: hoạt động quan sát được, tư thế cần xem, khung không xác định
Thời gian lưu bằng chỉ số khung, đổi ra ngày giờ lúc xuất qua VideoClock
(mốc OSD có thể đọc được sau khi đã ghi).
"""

from __future__ import annotations

import io
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from core.activity_context import load_activity_context
from core.behavior import (
    ACTIVITY_BEHAVIORS,
    NEUTRAL_BEHAVIORS,
    POSTURE_BEHAVIORS,
    REVIEW_BEHAVIORS,
    UNKNOWN_BEHAVIOR,
    REASON_SHORT,
    behavior_kind,
    observation_group,
    reason_name,
)
from core.config import ErnuConfig
from core.ernu import (
    BehaviorInterval,
    ObservationInterval,
    analyze_session,
)
from core.face_gallery import display_code
from core.video_clock import VideoClock

BEHAVIORS = tuple(name for name, _short in REASON_SHORT)
SHORT_OF = dict(REASON_SHORT)
# Không thấy lại hành vi sau N giây video → kết thúc đợt.
# Khớp dung sai mất tín hiệu: khoảng trống dài hơn mức này không được gộp vào đợt.
EPISODE_GAP_SEC = 1.0
# Điểm danh — phân biệt hai đại lượng (CHINHCODE §7.1):
#   identity_observations: số khung track đang mang mã HS (có thể khóa từ 1 lần ArcFace trước)
#   face_match_events: số lần ArcFace khớp hợp lệ thực sự
# Mặc định giữ phương án gốc; đặt attendance_mode="face_match_events" để thử thận trọng hơn.
PRESENT_MIN_FRAMES = 3
PRESENT_MIN_FACE_MATCHES = 3
ATTENDANCE_MODE = "identity_observations"  # hoặc "face_match_events"
DASHBOARD_MAX_MINUTES = 90
# Mức tin cậy điểm danh (theo điểm tương đồng quy đổi (%) cao nhất và số lần quan sát)
CONF_HIGH_PCT = 45.0
CONF_HIGH_VOTES = 10
CONF_MID_PCT = 35.0
FACE_THUMB_H = 96  # px, ảnh mặt nhúng vào Excel
FACE_CROP_FRAC = 0.45  # phần trên box người chứa mặt
# Không nội suy ngược start_frame từ bộ đếm (thời gian quan sát hợp lệ).
# Mốc bắt đầu/kết thúc = thời gian video khi nhãn xuất hiện / biến mất trên khung.
USE_TIMER_BACKDATE = False

_SECONDS_RE = re.compile(r"(\d+(?:\.\d+)?)s\)")


def _fmt_unknown(per_student: dict[str, dict]) -> str:
    total = sum(a["secs"].get(UNKNOWN_BEHAVIOR, 0.0) for a in per_student.values())
    return f"{total:.0f} giây" if total else "—"


def _fmt_class(per_student: dict[str, dict], names: frozenset[str]) -> str:
    secs = Counter()
    for agg in per_student.values():
        for name in names:
            secs[name] += agg["secs"][name]
    parts = [f"{name} {secs[name]:.0f}s" for name in names if secs[name] > 0]
    return ", ".join(parts) if parts else "—"


def _normalize(reason: str) -> tuple[str | None, float]:
    """'Cúi đầu lâu (5.2s)' → ('Cúi đầu lâu', 5.2)."""
    name = reason_name(reason)
    if name is None:
        return None, 0.0
    m = _SECONDS_RE.search(reason)
    return name, float(m.group(1)) if m else 0.0


def _secs_of(agg: dict, names: frozenset[str] | set[str]) -> float:
    return sum(s for b, s in agg["secs"].items() if b in names)


def _fmt_behaviors(agg: dict, names: frozenset[str] | set[str]) -> str:
    parts = [f"{b} {agg['secs'][b]:.0f}s" for b in names if agg["secs"][b] > 0]
    return ", ".join(parts) if parts else "—"


@dataclass
class _Episode:
    tid: int
    behavior: str
    start_frame: int
    last_frame: int
    start_wall: float
    last_wall: float
    detail: str = ""
    # Danh tính tại thời điểm sự kiện (không dùng đa số phiếu của cả track)
    student_id: str | None = None


@dataclass
class _TrackStats:
    n_obs: int = 0
    review_obs: int = 0
    unknown_obs: int = 0
    first_frame: int = 0
    last_frame: int = 0
    first_wall: float = 0.0
    last_wall: float = 0.0
    minute_review: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    minute_n: dict[int, int] = field(default_factory=lambda: defaultdict(int))


class BehaviorRecorder:
    def __init__(self, fps: float, clock: VideoClock, source_label: str,
                 student_names: dict[str, str] | None = None,
                 legend_fn: Callable[[], list[dict]] | None = None,
                 student_classes: dict[str, str] | None = None,
                 default_class: str = "",
                 attendance_mode: str | None = None,
                 ernu_cfg: ErnuConfig | None = None,
                 activity_segments: list[dict] | None = None,
                 video_id: str | None = None):
        self.fps = fps
        self.clock = clock
        self.source_label = source_label
        self.legend_fn = legend_fn or (lambda: [])
        self.report_name = "bao_cao_hanh_vi.xlsx"
        self.student_names = student_names or {}
        self.student_classes = student_classes or {}
        self.default_class = default_class
        mode = (attendance_mode or ATTENDANCE_MODE).strip().lower()
        if mode not in ("identity_observations", "face_match_events"):
            mode = "identity_observations"
        self.attendance_mode = mode
        self.ernu_cfg = ernu_cfg or ErnuConfig()
        self.video_id = (video_id or Path(source_label).stem or "session").strip() or "session"
        if activity_segments is not None:
            self.activity_segments = list(activity_segments)
        else:
            self.activity_segments = load_activity_context(
                self.ernu_cfg.activity_context_path or None, self.video_id
            )
        self._lock = threading.Lock()
        # sid → [tổng %, số lần, % cao nhất] điểm tương đồng quy đổi ArcFace
        self._sid_pct: dict[str, list[float]] = {}
        # sid → (% tương đồng, JPEG mặt) ảnh rõ nhất trong lớp
        self._face_best: dict[str, tuple[float, bytes]] = {}
        self._open: dict[tuple[int, str], _Episode] = {}
        self._closed: list[_Episode] = []
        self._stats: dict[int, _TrackStats] = {}
        self._sid_votes: dict[int, Counter] = defaultdict(Counter)
        # Số lần ArcFace khớp hợp lệ theo track (bằng chứng nhận diện độc lập)
        self._face_match_events: dict[int, int] = defaultdict(int)
        self._names: dict[str, str] = {}
        # Khoảng quan sát hợp lệ theo student_id (posed + known) — dùng cho coverage ERNU
        self._obs_open: dict[str, list[float]] = {}  # sid → [start_sec, last_sec]
        self._obs_closed: list[ObservationInterval] = []
        self.first_frame: int | None = None
        self.last_frame = 0
        self.first_wall = 0.0
        self.last_wall = 0.0

    @staticmethod
    def _face_jpeg(frame, d: dict) -> bytes | None:
        import cv2

        x1, y1, x2, y2 = (int(d[k]) for k in ("x1", "y1", "x2", "y2"))
        crop = frame[max(0, y1):max(0, y1) + int((y2 - y1) * FACE_CROP_FRAC), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        crop = cv2.resize(crop, (max(1, int(w * FACE_THUMB_H / h)), FACE_THUMB_H),
                          interpolation=cv2.INTER_AREA if h > FACE_THUMB_H else cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else None

    def _frame_sec(self, frame_idx: int) -> float:
        base = self.first_frame if self.first_frame is not None else frame_idx
        return max(0.0, (frame_idx - base) / self.fps) if self.fps else 0.0

    def _touch_observation(self, sid: str, t_sec: float) -> None:
        """Gộp khoảng quan sát liên tục theo student_id (posed + known)."""
        gap = EPISODE_GAP_SEC
        open_span = self._obs_open.get(sid)
        if open_span is None:
            self._obs_open[sid] = [t_sec, t_sec]
            return
        if t_sec - open_span[1] > gap:
            self._obs_closed.append(ObservationInterval(sid, open_span[0], open_span[1]))
            self._obs_open[sid] = [t_sec, t_sec]
        else:
            open_span[1] = max(open_span[1], t_sec)

    def record(self, frame_idx: int, detections: list[dict], wall: float, frame=None) -> None:
        gap_frames = EPISODE_GAP_SEC * self.fps
        with self._lock:
            if self.first_frame is None:
                self.first_frame, self.first_wall = frame_idx, wall
            self.last_frame, self.last_wall = frame_idx, wall
            minute = int((frame_idx - self.first_frame) / self.fps // 60)
            t_sec = self._frame_sec(frame_idx)
            seen: set[tuple[int, str]] = set()
            for d in detections:
                tid = int(d["track_id"])
                face_events = int(d.get("face_match_events") or 0)
                if face_events > self._face_match_events[tid]:
                    self._face_match_events[tid] = face_events
                if d.get("known") and d.get("student_id"):
                    sid = d["student_id"]
                    self._sid_votes[tid][sid] += 1
                    if d.get("name"):
                        self._names[sid] = d["name"]
                    pct = d.get("match_pct")
                    if pct is not None:
                        acc = self._sid_pct.setdefault(sid, [0.0, 0, 0.0])
                        acc[0] += pct
                        acc[1] += 1
                        acc[2] = max(acc[2], pct)
                        if frame is not None and pct > self._face_best.get(sid, (-1.0, b""))[0]:
                            jpg = self._face_jpeg(frame, d)
                            if jpg:
                                self._face_best[sid] = (pct, jpg)
                if not d.get("posed"):
                    continue
                st = self._stats.get(tid)
                if st is None:
                    st = self._stats[tid] = _TrackStats(
                        first_frame=frame_idx, first_wall=wall)
                st.n_obs += 1
                group = d.get("group") or observation_group(d.get("reasons") or [])
                if group == "review":
                    st.review_obs += 1
                    st.minute_review[minute] += 1
                elif group == "unknown":
                    st.unknown_obs += 1
                st.last_frame, st.last_wall = frame_idx, wall
                st.minute_n[minute] += 1

                event_sid = d["student_id"] if d.get("known") and d.get("student_id") else None
                if event_sid is not None:
                    self._touch_observation(event_sid, t_sec)
                for reason in d.get("reasons") or []:
                    name, secs = _normalize(reason)
                    if name is None:
                        continue
                    key = (tid, name)
                    seen.add(key)
                    ep = self._open.get(key)
                    if ep is not None and frame_idx - ep.last_frame > gap_frames:
                        # Mất dấu lâu hơn dung sai → đóng đợt; không cộng khoảng thiếu bằng chứng
                        self._closed.append(self._open.pop(key))
                        ep = None
                    if ep is None:
                        if USE_TIMER_BACKDATE and secs > 0:
                            back = int(round(secs * self.fps))
                            start_f = max(0, frame_idx - back)
                            start_w = wall - secs
                        else:
                            # Thời gian video: mốc khi nhãn lần đầu xuất hiện trên khung phân tích
                            start_f, start_w = frame_idx, wall
                        self._open[key] = _Episode(
                            tid, name, start_f, frame_idx,
                            start_w, wall, reason, student_id=event_sid)
                    else:
                        ep.last_frame, ep.last_wall = frame_idx, wall
                        ep.detail = reason
                        # Chỉ gắn/ cập nhật mã HS khi có xác minh tại khung hiện tại
                        if event_sid is not None:
                            if ep.student_id is None:
                                ep.student_id = event_sid
                            elif ep.student_id != event_sid:
                                # Đổi danh tính giữa chừng → đóng đợt cũ, mở đợt mới
                                self._closed.append(self._open.pop(key))
                                self._open[key] = _Episode(
                                    tid, name, frame_idx, frame_idx,
                                    wall, wall, reason, student_id=event_sid)
            for key in list(self._open):
                ep = self._open[key]
                if key not in seen and frame_idx - ep.last_frame > gap_frames:
                    self._closed.append(self._open.pop(key))

    # ---------------------------------------------------------------- xuất

    def _student_of(self, tid: int) -> str | None:
        votes = self._sid_votes.get(tid)
        return votes.most_common(1)[0][0] if votes else None

    def _episode_sid(self, ep: _Episode) -> str | None:
        """Mã HS của sự kiện: ưu tiên danh tính tại thời điểm ghi, không gộp theo cả track."""
        if ep.student_id:
            return ep.student_id
        return None

    def _display_name(self, sid: str) -> str:
        """Họ tên trong students.csv được ưu tiên; tên lúc nhận diện chỉ dùng khi danh sách chưa có."""
        official = (self.student_names.get(sid) or "").strip()
        seen = (self._names.get(sid) or "").strip()
        name = official or seen
        return "" if not name or name == sid else name

    def _who(self, tid: int) -> tuple[str, str]:
        sid = self._student_of(tid)
        if sid is None:
            return f"ID{tid} (chưa nhận diện)", ""
        return sid, self._display_name(sid)

    def _who_episode(self, ep: _Episode) -> tuple[str, str]:
        sid = self._episode_sid(ep)
        if sid is None:
            return f"ID{ep.tid} (chưa nhận diện)", ""
        return sid, self._display_name(sid)

    def _time_cells(self, frame_idx: int, wall: float) -> tuple[str, str, str]:
        """(ngày như video, giờ, vị trí trong video)."""
        dt = self.clock.at(frame_idx, wall)
        offset = str(timedelta(seconds=int(max(0, frame_idx) / self.fps)))
        if dt is None:
            return "", offset, offset
        return self.clock.format_date(dt), dt.strftime("%H:%M:%S"), offset

    def _per_student(self, episodes: list[_Episode]) -> dict[str, dict]:
        """Gộp quan sát theo HS. Sự kiện chỉ cộng cho mã HS đã xác minh tại thời điểm đó."""
        per_student: dict[str, dict] = {}

        def _ensure(sid: str, name: str = "") -> dict:
            if not name and not sid.startswith("ID"):
                name = self._display_name(sid)
            return per_student.setdefault(sid, {
                "name": name,
                "n": 0, "review_n": 0, "unknown_n": 0,
                "first": None, "last": None,
                "count": Counter(), "secs": Counter(),
            })

        for tid, st in self._stats.items():
            sid, name = self._who(tid)
            agg = _ensure(sid, name)
            if not agg["name"] and name:
                agg["name"] = name
            agg["n"] += st.n_obs
            agg["review_n"] += st.review_obs
            agg["unknown_n"] += st.unknown_obs
            first = (st.first_frame, st.first_wall)
            last = (st.last_frame, st.last_wall)
            if agg["first"] is None or st.first_frame < agg["first"][0]:
                agg["first"] = first
            if agg["last"] is None or st.last_frame > agg["last"][0]:
                agg["last"] = last
        for ep in episodes:
            sid, name = self._who_episode(ep)
            # Chỉ gộp hành vi khi có danh tính tại thời điểm sự kiện
            if self._episode_sid(ep) is None:
                continue
            agg = _ensure(sid, name)
            if not agg["name"] and name:
                agg["name"] = name
            agg["count"][ep.behavior] += 1
            agg["secs"][ep.behavior] += max(0.0, (ep.last_frame - ep.start_frame) / self.fps)
            if agg["first"] is None or ep.start_frame < agg["first"][0]:
                agg["first"] = (ep.start_frame, ep.start_wall)
            if agg["last"] is None or ep.last_frame > agg["last"][0]:
                agg["last"] = (ep.last_frame, ep.last_wall)
        for agg in per_student.values():
            if agg["first"] is None:
                agg["first"] = (0, 0.0)
            if agg["last"] is None:
                agg["last"] = agg["first"]
        return per_student

    def _sid_vote_totals(self) -> Counter:
        """identity_observations: số khung track đang mang mã HS. Gọi khi đang giữ self._lock."""
        votes: Counter = Counter()
        for tid, c in self._sid_votes.items():
            sid = self._student_of(tid)
            if sid is not None:
                votes[sid] += c[sid]
        return votes

    def _sid_face_match_totals(self) -> Counter:
        """face_match_events: tổng lần ArcFace khớp hợp lệ theo mã HS (qua track)."""
        totals: Counter = Counter()
        for tid, n in self._face_match_events.items():
            sid = self._student_of(tid)
            if sid is not None and n > 0:
                totals[sid] += int(n)
        return totals

    def _present_ids(self, votes: Counter | None = None,
                     face_totals: Counter | None = None) -> set[str]:
        votes = self._sid_vote_totals() if votes is None else votes
        face_totals = self._sid_face_match_totals() if face_totals is None else face_totals
        if self.attendance_mode == "face_match_events":
            return {sid for sid, n in face_totals.items() if n >= PRESENT_MIN_FACE_MATCHES}
        return {sid for sid, n in votes.items() if n >= PRESENT_MIN_FRAMES}

    def _behavior_items(self, agg: dict, names: frozenset[str]) -> list[dict]:
        items = [
            {"name": b, "short": SHORT_OF.get(b, b), "count": agg["count"][b], "secs": round(s, 1)}
            for b, s in agg["secs"].items() if b in names and s > 0
        ]
        items.sort(key=lambda x: -x["secs"])
        return items

    def _flush_observations(self) -> list[ObservationInterval]:
        """Đóng các khoảng quan sát đang mở. Gọi khi đang giữ self._lock."""
        out = list(self._obs_closed)
        for sid, span in self._obs_open.items():
            if span[1] >= span[0]:
                out.append(ObservationInterval(sid, span[0], span[1]))
        return out

    def _behavior_intervals(self, episodes: list[_Episode]) -> list[BehaviorInterval]:
        """Chỉ episode có student_id xác minh → khoảng giây video từ first_frame."""
        out: list[BehaviorInterval] = []
        base = self.first_frame or 0
        fps = self.fps or 1.0
        for ep in episodes:
            sid = self._episode_sid(ep)
            if sid is None:
                continue
            start = max(0.0, (ep.start_frame - base) / fps)
            end = max(start, (ep.last_frame - base) / fps)
            out.append(BehaviorInterval(sid, ep.behavior, start, end))
        return out

    def _session_duration_sec(self) -> float:
        if self.first_frame is None:
            return 0.0
        return max(0.0, (self.last_frame - self.first_frame) / (self.fps or 1.0))

    def build_ernu(self) -> tuple[list, list, dict]:
        """Cửa sổ PREDICTED + metrics từng HS + tổng hợp lớp. Thread-safe."""
        with self._lock:
            episodes = self._closed + list(self._open.values())
            behaviors = self._behavior_intervals(episodes)
            observations = self._flush_observations()
            duration = self._session_duration_sec()
            votes = self._sid_vote_totals()
            face_totals = self._sid_face_match_totals()
            present = self._present_ids(votes, face_totals)
            student_ids = sorted(set(self.student_names) | present | {
                b.student_id for b in behaviors
            } | {o.student_id for o in observations})
            # Chỉ phân tích HS có danh tính thật (không ID track giả)
            student_ids = [s for s in student_ids if not str(s).startswith("ID")]
            segs = list(self.activity_segments)
            vid = self.video_id
            cfg = self.ernu_cfg
        return analyze_session(
            video_id=vid,
            duration_sec=duration,
            student_ids=student_ids,
            behaviors=behaviors,
            observations=observations,
            activity_segments=segs,
            cfg=cfg,
        )

    def summary(self) -> dict:
        """Sĩ số, hành vi thô B1–B5 và chỉ số ERNU E/R/N/U."""
        with self._lock:
            episodes = self._closed + list(self._open.values())
            per_student = self._per_student(episodes)
            votes = self._sid_vote_totals()
            face_totals = self._sid_face_match_totals()
            unknown_tracks = sum(1 for tid in self._stats if self._student_of(tid) is None)
            minute_review: dict[int, int] = defaultdict(int)
            minute_n: dict[int, int] = defaultdict(int)
            for st in self._stats.values():
                for m, n in st.minute_n.items():
                    minute_n[m] += n
                    minute_review[m] += st.minute_review[m]
            present_ids = self._present_ids(votes, face_totals)
            roster = list(self.student_names)
            roster += sorted(present_ids - set(roster))
            name_of = {sid: self._display_name(sid) for sid in roster}
            first_frame, first_wall = self.first_frame or 0, self.first_wall
            attendance_mode = self.attendance_mode

        windows, ernu_metrics, class_ernu = self.build_ernu()
        ernu_by_sid = {m.student_id: m for m in ernu_metrics}

        students = []
        for sid in sorted(present_ids):
            agg = per_student.get(sid)
            em = ernu_by_sid.get(sid)
            if agg is None:
                review_secs = activity_secs = unknown_secs = 0.0
                review = activity = posture = neutral = []
                first_seen = ""
            else:
                review_secs = round(_secs_of(agg, REVIEW_BEHAVIORS), 1)
                activity_secs = round(_secs_of(agg, ACTIVITY_BEHAVIORS), 1)
                unknown_secs = round(agg["secs"].get(UNKNOWN_BEHAVIOR, 0.0), 1)
                review = self._behavior_items(agg, REVIEW_BEHAVIORS)
                activity = self._behavior_items(agg, ACTIVITY_BEHAVIORS)
                posture = self._behavior_items(agg, POSTURE_BEHAVIORS)
                neutral = self._behavior_items(agg, NEUTRAL_BEHAVIORS)
                first_seen = self._time_cells(*agg["first"])[1]
            students.append({
                "student_id": sid,
                "code": display_code(sid),
                "name": name_of.get(sid, ""),
                "first_seen": first_seen,
                "identity_observations": int(votes.get(sid, 0)),
                "face_match_events": int(face_totals.get(sid, 0)),
                "review_secs": review_secs,
                "activity_secs": activity_secs,
                "unknown_secs": unknown_secs,
                "review": review,
                "activity": activity,
                "posture": posture,
                "neutral": neutral,
                "ernu": None if em is None else {
                    "T": em.T, "T_E": em.T_E, "T_R": em.T_R, "T_N": em.T_N, "T_U": em.T_U,
                    "T_valid": em.T_valid,
                    "C": em.C, "P_E": em.P_E, "P_R": em.P_R, "P_N": em.P_N, "P_U": em.P_U,
                    "data_status": em.data_status,
                    "publishable": em.publishable,
                    "BEEI": em.BEEI,
                    "BEEI_wN0": em.BEEI_wN0,
                    "BEEI_wN1": em.BEEI_wN1,
                    "interpretation": em.interpretation,
                },
            })

        def class_secs(pool) -> float:
            return round(sum(_secs_of(a, pool) for a in per_student.values()), 1)

        total_n = sum(a["n"] for a in per_student.values())
        total_review = sum(a["review_n"] for a in per_student.values())
        timeline = []
        for m in sorted(minute_n)[-DASHBOARD_MAX_MINUTES:]:
            f = first_frame + int(m * 60 * self.fps)
            n = minute_n[m]
            timeline.append({
                "label": self._time_cells(f, first_wall + m * 60)[1],
                "review_pct": round(100.0 * minute_review[m] / n, 1) if n else None,
            })

        return {
            "roster_size": len(roster),
            "attendance_mode": attendance_mode,
            "present": students,
            "absent": [
                {"student_id": sid, "name": name_of.get(sid, "")}
                for sid in roster if sid not in present_ids
            ],
            "unknown_tracks": unknown_tracks,
            "class_activity_secs": class_secs(ACTIVITY_BEHAVIORS),
            "class_review_secs": class_secs(REVIEW_BEHAVIORS),
            "class_unknown_secs": round(
                sum(a["secs"].get(UNKNOWN_BEHAVIOR, 0.0) for a in per_student.values()), 1),
            "class_review_pct": round(100.0 * total_review / total_n, 1) if total_n else None,
            "timeline": timeline,
            "ernu": {
                **class_ernu,
                "rules_version": self.ernu_cfg.rules_version,
                "window_sec": self.ernu_cfg.window_sec,
                "c_min": self.ernu_cfg.c_min,
                "publish_c_min": self.ernu_cfg.publish_c_min,
                "default_activity": self.ernu_cfg.default_activity,
                "activity_segments": len(self.activity_segments),
                "n_windows": len(windows),
                "disclaimer": (
                    "E/R/N/U là nhãn cửa sổ theo hành vi quan sát được — "
                    "không phải điểm tập trung nhận thức hay xếp hạng học sinh. "
                    "BEEI chỉ là chỉ số phụ; khi coverage thấp không dùng đơn độc."
                ),
            },
        }

    def export_excel(self, path: Path) -> Path:
        with self._lock:
            wb = self._build_workbook()
        path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(path)
        return path

    @staticmethod
    def _confidence_level(votes: int, max_pct: float | None) -> str:
        if max_pct is None:
            return ""
        if max_pct >= CONF_HIGH_PCT and votes >= CONF_HIGH_VOTES:
            return "Cao"
        if max_pct >= CONF_MID_PCT:
            return "Trung bình"
        return "Thấp — cần kiểm tra"

    def _write_attendance_sheet(self, ws, per_student: dict[str, dict]) -> None:
        """Sheet chính: mỗi học sinh trong danh sách lớp là 1 hàng. Gọi khi đang giữ self._lock."""
        from openpyxl.drawing.image import Image as XLImage
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter

        ws.title = "Điểm danh theo HS"
        votes = self._sid_vote_totals()
        face_totals = self._sid_face_match_totals()
        present_ids = self._present_ids(votes, face_totals)
        roster = list(self.student_names)
        roster += sorted(present_ids - set(roster))

        headers = [
            ("STT", 5), ("Mã HS", 10), ("Họ và tên", 22), ("Lớp", 8), ("Trạng thái", 11),
            ("Ngày (như video)", 12), ("Giờ điểm danh (lần đầu nhận diện)", 13),
            ("Lần cuối thấy", 11), ("Thời gian có mặt (phút)", 11),
            ("Quan sát có danh tính", 11), ("Lần ArcFace khớp", 11),
            ("Điểm tương đồng TB (%)", 10), ("Điểm tương đồng cao nhất (%)", 10),
            ("Mức tin cậy", 13), ("Ảnh khuôn mặt (trong lớp)", 13),
            ("Hoạt động quan sát được", 28), ("Tư thế cần xem lại", 28),
            ("Không xác định (giây)", 14),
        ]
        for b in BEHAVIORS:
            headers += [(f"{b} (lần)", 9), (f"{b} (giây)", 9)]
        headers.append(("Ghi chú (giáo viên)", 24))
        ncol = len(headers)
        col_img = [h for h, _w in headers].index("Ảnh khuôn mặt (trong lớp)") + 1
        last_col = get_column_letter(ncol)

        rows, images = [], []
        n_present = 0
        for i, sid in enumerate(roster, start=1):
            agg = per_student.get(sid)
            present = sid in present_ids
            n_present += present
            name = self._display_name(sid)
            row = [i, sid, name, self.student_classes.get(sid) or self.default_class,
                   "Có mặt" if present else "Vắng"]
            if present and agg:
                date_s, from_s, _o = self._time_cells(*agg["first"])
                _d, to_s, _o2 = self._time_cells(*agg["last"])
                minutes = round(max(0, agg["last"][0] - agg["first"][0]) / self.fps / 60, 1)
            else:
                date_s = from_s = to_s = ""
                minutes = None
            pct = self._sid_pct.get(sid)
            avg_pct = round(pct[0] / pct[1], 1) if present and pct and pct[1] else None
            max_pct = round(pct[2], 1) if present and pct else None
            id_obs = int(votes.get(sid, 0)) if present else 0
            face_n = int(face_totals.get(sid, 0)) if present else 0
            row += [
                date_s, from_s, to_s, minutes, id_obs, face_n,
                avg_pct, max_pct,
                self._confidence_level(id_obs, max_pct) if present else "",
                "",
            ]
            n = agg["n"] if present and agg else 0
            if n:
                row += [
                    _fmt_behaviors(agg, ACTIVITY_BEHAVIORS),
                    _fmt_behaviors(agg, REVIEW_BEHAVIORS),
                    round(agg["secs"].get(UNKNOWN_BEHAVIOR, 0.0), 1),
                ]
                for b in BEHAVIORS:
                    row += [agg["count"][b], round(agg["secs"][b], 1)]
            else:
                row += [None] * (3 + 2 * len(BEHAVIORS))
            row.append("")
            rows.append(row)
            face = self._face_best.get(sid) if present else None
            images.append(face[1] if face else None)

        # --- phần đầu: thông tin buổi học
        title_font = Font(bold=True, size=14, color="1F4E78")
        label_font = Font(bold=True)
        ws.merge_cells(f"A1:{last_col}1")
        ws["A1"] = "BÁO CÁO ĐIỂM DANH VÀ HÀNH VI QUAN SÁT"
        ws["A1"].font = title_font
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 24

        d1, t1, o1 = self._time_cells(self.first_frame or 0, self.first_wall)
        _d2, t2, o2 = self._time_cells(self.last_frame, self.last_wall)
        classes = sorted({r[3] for r in rows if r[3]})
        total = len(roster)
        info = [
            ("Lớp", ", ".join(classes)),
            ("Nguồn video", self.source_label),
            ("Ngày (như video)", d1),
            ("Thời gian phân tích", f"{t1} → {t2}  (video {o1} → {o2})"),
            ("Sĩ số", total),
            ("Có mặt", n_present),
            ("Vắng / chưa nhận diện", total - n_present),
            ("Tỉ lệ chuyên cần", f"{100.0 * n_present / total:.1f}%" if total else ""),
            ("Tín hiệu hoạt động thô (B1/B2)", _fmt_class(per_student, ACTIVITY_BEHAVIORS)),
            ("Tín hiệu cần xem thô (B4/B5)", _fmt_class(per_student, REVIEW_BEHAVIORS)),
            ("Không xác định (thiếu pose)", _fmt_unknown(per_student)),
            ("Track chưa nhận diện được mặt", sum(1 for s in per_student if s.startswith("ID"))),
            ("Chỉ số ERNU", "Xem sheet TONG_HOP_ERNU và PREDICTED_WINDOWS (E/R/N/U theo cửa sổ 10s)."),
            ("Quy tắc điểm danh", (
                f"Chế độ đang dùng: {self.attendance_mode}. "
                f"identity_observations ≥ {PRESENT_MIN_FRAMES} "
                f"hoặc face_match_events ≥ {PRESENT_MIN_FACE_MATCHES} tùy chế độ. "
                "Ba quan sát có danh tính không đồng nghĩa ba lần ArcFace độc lập. "
                f"Mức tin cậy Cao: điểm tương đồng quy đổi ≥ {CONF_HIGH_PCT:.0f}% và ≥ {CONF_HIGH_VOTES} lần; "
                f"Trung bình: ≥ {CONF_MID_PCT:.0f}%. "
                "Hành vi chỉ gộp theo mã HS đã xác minh tại thời điểm sự kiện. "
                "Thời lượng sự kiện = (last_frame − start_frame) / FPS video (không cộng khoảng thiếu bằng chứng)."
            )),
        ]
        r0 = 3
        for k, (label, value) in enumerate(info):
            r = r0 + k
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=3)
            ws.merge_cells(start_row=r, start_column=4, end_row=r, end_column=min(ncol, 13))
            ws.cell(row=r, column=1, value=label).font = label_font
            ws.cell(row=r, column=4, value=value).alignment = Alignment(horizontal="left")

        # --- bảng học sinh
        hr = r0 + len(info) + 1
        thin = Side(style="thin", color="BFBFBF")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)
        header_fill = PatternFill("solid", fgColor="1F4E78")
        for c, (h, w) in enumerate(headers, start=1):
            cell = ws.cell(row=hr, column=c, value=h)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border
            ws.column_dimensions[get_column_letter(c)].width = w
        ws.row_dimensions[hr].height = 48

        fills = {
            "Có mặt": PatternFill("solid", fgColor="E2F5E6"),
            "Vắng": PatternFill("solid", fgColor="FDE2E1"),
            "Cao": PatternFill("solid", fgColor="E2F5E6"),
            "Trung bình": PatternFill("solid", fgColor="FFF2CC"),
            "Thấp — cần kiểm tra": PatternFill("solid", fgColor="FCE4D6"),
        }
        img_max_w = FACE_THUMB_H * 3 // 2
        has_images = any(images)
        widest = 0
        for k, (row, jpg) in enumerate(zip(rows, images)):
            r = hr + 1 + k
            for c, v in enumerate(row, start=1):
                cell = ws.cell(row=r, column=c, value=v)
                cell.border = border
                cell.alignment = Alignment(horizontal="left" if c in (2, 3, 16, 17, ncol) else "center",
                                           vertical="center", wrap_text=c in (3, 16, 17, ncol))
            ws.cell(row=r, column=5).fill = fills[row[4]]
            # Cột "Mức tin cậy" (index 13 sau khi thêm cột ArcFace khớp)
            if len(row) > 13 and row[13] in fills:
                ws.cell(row=r, column=14).fill = fills[row[13]]
            if has_images:
                ws.row_dimensions[r].height = FACE_THUMB_H * 0.75 + 4
            if jpg:
                img = XLImage(io.BytesIO(jpg))
                s = min(FACE_THUMB_H / img.height, img_max_w / img.width)
                img.width, img.height = int(img.width * s), int(img.height * s)
                widest = max(widest, img.width)
                ws.add_image(img, f"{get_column_letter(col_img)}{r}")
        if has_images:
            ws.column_dimensions[get_column_letter(col_img)].width = max(13, widest / 7 + 1)
        ws.freeze_panes = ws.cell(row=hr + 1, column=4)
        if rows:
            ws.auto_filter.ref = f"A{hr}:{last_col}{hr + len(rows)}"
        ws.sheet_view.zoomScale = 90
        ws.page_setup.orientation = "landscape"
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.print_title_rows = f"{hr}:{hr}"

    def _build_workbook(self):
        """Gọi khi đang giữ self._lock (luồng phân tích vẫn ghi song song)."""
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        episodes = sorted(
            self._closed + list(self._open.values()), key=lambda e: e.start_frame
        )
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

        per_student = self._per_student(episodes)
        self._write_attendance_sheet(wb.active, per_student)

        # --- Chi tiết hành vi
        ws = wb.create_sheet("Chi tiết hành vi")
        rows = []
        for i, ep in enumerate(episodes, start=1):
            sid, name = self._who_episode(ep)
            date_s, start_s, off_s = self._time_cells(ep.start_frame, ep.start_wall)
            _d, end_s, _o = self._time_cells(ep.last_frame, ep.last_wall)
            dur = round(max(0.0, (ep.last_frame - ep.start_frame) / self.fps), 1)
            rows.append([
                i, sid, name, ep.behavior, SHORT_OF.get(ep.behavior, ""),
                behavior_kind(ep.behavior),
                date_s, start_s, end_s, dur, off_s, ep.tid,
                "Đã xác minh" if ep.student_id else "Chưa xác định",
            ])
        write_sheet(
            ws,
            ["STT", "Mã HS", "Họ tên", "Hành vi", "Viết tắt (trên video)", "Nhóm",
             "Ngày (như video)", "Bắt đầu", "Kết thúc", "Thời lượng (giây)",
             "Vị trí trong video", "Track ID", "Danh tính sự kiện"],
            rows,
            [6, 22, 22, 20, 12, 26, 16, 11, 11, 12, 14, 9, 14],
        )
        kind_fill = {
            "Hoạt động quan sát được": good_fill,
            "Cần giáo viên xem": bad_fill,
            "Không xác định": PatternFill("solid", fgColor="DDEBF7"),
            "Tư thế ghi nhận": PatternFill("solid", fgColor="FFF2CC"),
            "Tín hiệu trung tính (ERNU→N)": PatternFill("solid", fgColor="E2EFDA"),
        }
        for row in ws.iter_rows(min_row=2):
            row[5].fill = kind_fill.get(row[5].value, PatternFill("solid", fgColor="E7E6E6"))

        # --- Track chưa nhận diện được mặt (không thuộc danh sách lớp)
        ws2 = wb.create_sheet("Track chưa nhận diện")
        headers = ["Track", "Ngày (như video)", "Xuất hiện từ", "Đến",
                   "Cần xem (giây)", "Không xác định (giây)"]
        for b in BEHAVIORS:
            headers += [f"{b} (lần)", f"{b} (giây)"]
        rows2 = []
        unknown = [s for s in per_student if s.startswith("ID")]
        for sid in sorted(unknown, key=lambda s: int(re.sub(r"\D", "", s) or 0)):
            agg = per_student[sid]
            date_s, from_s, _o = self._time_cells(*agg["first"])
            _d, to_s, _o2 = self._time_cells(*agg["last"])
            row = [
                sid.split()[0], date_s, from_s, to_s,
                round(_secs_of(agg, REVIEW_BEHAVIORS), 1),
                round(agg["secs"].get(UNKNOWN_BEHAVIOR, 0.0), 1),
            ]
            for b in BEHAVIORS:
                row += [agg["count"][b], round(agg["secs"][b], 1)]
            rows2.append(row)
        write_sheet(ws2, headers, rows2, [10, 16, 12, 12, 16, 18] + [11] * (2 * len(BEHAVIORS)))

        # --- ERNU: cửa sổ dự đoán + tổng hợp
        ernu_windows, ernu_metrics, class_ernu = analyze_session(
            video_id=self.video_id,
            duration_sec=self._session_duration_sec(),
            student_ids=sorted({
                s for s in (
                    list(self.student_names)
                    + [b.student_id for b in self._behavior_intervals(episodes)]
                    + [o.student_id for o in self._flush_observations()]
                ) if s and not str(s).startswith("ID")
            }),
            behaviors=self._behavior_intervals(episodes),
            observations=self._flush_observations(),
            activity_segments=self.activity_segments,
            cfg=self.ernu_cfg,
        )

        ws_pred = wb.create_sheet("PREDICTED_WINDOWS")
        pred_rows = [
            [
                w.video_id, w.student_id, w.start_sec, w.end_sec, w.activity_code,
                ",".join(w.behavior_events), w.valid_coverage, w.pred_label,
                w.reason, "1" if w.mixed else "0", w.rules_version,
            ]
            for w in ernu_windows
        ]
        write_sheet(
            ws_pred,
            ["video_id", "student_id", "start_sec", "end_sec", "activity_code",
             "behavior_events", "valid_coverage", "pred_label", "reason", "mixed",
             "rules_version"],
            pred_rows,
            [14, 12, 10, 10, 12, 22, 12, 10, 40, 8, 12],
        )
        label_fill = {
            "E": good_fill,
            "R": bad_fill,
            "N": PatternFill("solid", fgColor="FFF2CC"),
            "U": PatternFill("solid", fgColor="DDEBF7"),
        }
        for row in ws_pred.iter_rows(min_row=2, min_col=8, max_col=8):
            cell = row[0]
            if cell.value in label_fill:
                cell.fill = label_fill[cell.value]

        ws_ernu = wb.create_sheet("TONG_HOP_ERNU")
        write_sheet(
            ws_ernu,
            ["student_id", "name", "T", "T_E", "T_R", "T_N", "T_U",
             "C_%", "P_E_%", "P_R_%", "P_N_%", "P_U_%",
             "data_status", "publishable", "BEEI", "BEEI_wN0", "BEEI_wN1"],
            [
                [
                    m.student_id, self._display_name(m.student_id),
                    m.T, m.T_E, m.T_R, m.T_N, m.T_U,
                    m.C, m.P_E, m.P_R, m.P_N, m.P_U,
                    m.data_status, "1" if m.publishable else "0",
                    m.BEEI, m.BEEI_wN0, m.BEEI_wN1,
                ]
                for m in ernu_metrics
            ] + [[
                "(LỚP)", "",
                class_ernu["T"], class_ernu["T_E"], class_ernu["T_R"], class_ernu["T_N"], class_ernu["T_U"],
                class_ernu["C"], class_ernu["P_E"], class_ernu["P_R"], class_ernu["P_N"], class_ernu["P_U"],
                class_ernu["data_status"], "1" if class_ernu.get("publishable") else "0",
                class_ernu["BEEI"], class_ernu["BEEI_wN0"], class_ernu["BEEI_wN1"],
            ]],
            [12, 18, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 16, 12, 8, 10, 10],
        )

        # --- Diễn giải (mục 6 bộ tiêu chí)
        ws_interp = wb.create_sheet("DIEN_GIAI_ERNU")
        interp_rows = [
            [m.student_id, self._display_name(m.student_id), m.data_status,
             "1" if m.publishable else "0", m.interpretation]
            for m in ernu_metrics
        ]
        interp_rows.append([
            "(LỚP)", "", class_ernu["data_status"],
            "1" if class_ernu.get("publishable") else "0",
            class_ernu.get("interpretation") or "",
        ])
        write_sheet(
            ws_interp,
            ["student_id", "name", "data_status", "publishable", "interpretation"],
            interp_rows,
            [12, 18, 16, 12, 100],
        )

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
            ("video_id (ERNU)", self.video_id),
            ("Nguồn thời gian", src),
            ("Bắt đầu ghi", f"{d1} {t1}".strip()),
            ("Kết thúc ghi", f"{d2} {t2}".strip()),
            ("Đoạn video đã phân tích", f"{o1} → {o2}"),
            ("Số đợt hành vi (B thô)", len(episodes)),
            ("Số học sinh / track", len(per_student)),
            ("ERNU rules_version", self.ernu_cfg.rules_version),
            ("Cửa sổ ERNU (giây)", self.ernu_cfg.window_sec),
            ("C_min", self.ernu_cfg.c_min),
            ("publish_c_min", self.ernu_cfg.publish_c_min),
            ("min_duration B1/B3/B4/B5 (s)",
             f"{self.ernu_cfg.min_duration_b1}/{self.ernu_cfg.min_duration_b3}/"
             f"{self.ernu_cfg.min_duration_b4}/{self.ernu_cfg.min_duration_b5}"),
            ("BEEI weights (w_E,w_R,w_N)",
             f"{self.ernu_cfg.beei_w_e},{self.ernu_cfg.beei_w_r},{self.ernu_cfg.beei_w_n}"),
            ("Hoạt động mặc định", self.ernu_cfg.default_activity),
            ("Số đoạn ACTIVITY_CONTEXT", len(self.activity_segments)),
            ("Số cửa sổ PREDICTED", len(ernu_windows)),
            ("Ghi chú", "Hành vi thô B1–B5 tách khỏi nhãn cửa sổ E/R/N/U. "
                        "E = bằng chứng thực hiện nhiệm vụ; R = cần giáo viên xem lại; "
                        "N = quan sát được nhưng chưa đủ căn cứ; U = thiếu dữ liệu quan sát. "
                        "Không đồng nhất tư thế với mức tập trung nhận thức. "
                        "BEEI chỉ là chỉ số phụ (dashboard), không phải điểm tập trung; "
                        "khi coverage thấp không xếp mức / không dùng đơn độc. "
                        "Không dùng E/R/N/U hệ thống làm ground truth của chính hệ thống. "
                        "Quay đầu (B3) là tín hiệu trung tính (→N), không xếp hoạt động bằng chứng. "
                        f"Một đợt B kết thúc sau {EPISODE_GAP_SEC:.0f}s không thấy lại."),
        ]
        for k, v in info:
            ws4.append([k, v])
        for c in ws4["A"]:
            c.font = Font(bold=True)
        ws4.column_dimensions["A"].width = 28
        ws4.column_dimensions["B"].width = 96

        # --- Chú thích nhãn viết tắt
        ws5 = wb.create_sheet("Chú thích")
        rows5 = [
            [lg["short"], lg["name"], behavior_kind(lg["name"]),
             lg["condition"],
             "Bật" if lg.get("enabled", True) else "Tắt"]
            for lg in self.legend_fn()
        ]
        write_sheet(
            ws5,
            ["Viết tắt (trên video)", "Hành vi đầy đủ", "Nhóm", "Điều kiện nhận biết",
             "Trạng thái (lúc xuất)"],
            rows5,
            [16, 22, 28, 70, 14],
        )
        off_fill = PatternFill("solid", fgColor="E7E6E6")
        kind_fill = {
            "Hoạt động quan sát được": good_fill,
            "Cần giáo viên xem": bad_fill,
            "Không xác định": PatternFill("solid", fgColor="DDEBF7"),
            "Tư thế ghi nhận": PatternFill("solid", fgColor="FFF2CC"),
            "Tín hiệu trung tính (ERNU→N)": PatternFill("solid", fgColor="E2EFDA"),
        }
        for row in ws5.iter_rows(min_row=2, max_row=1 + len(rows5)):
            row[2].fill = kind_fill.get(row[2].value, PatternFill("solid", fgColor="E7E6E6"))
            if row[4].value == "Tắt":
                row[4].fill = off_fill
        ws5.append([])
        box_notes = [
            ("Khung đỏ", "Tín hiệu cần xem (B4/B5): cúi đầu lâu hoặc gục xuống bàn — diễn giải R tùy A", "FDE2E1"),
            ("Khung xanh dương", "Tín hiệu hoạt động (B1/B2): giơ tay, viết bài — diễn giải E tùy A", "DDEBF7"),
            ("Khung xám", "Có tư thế, không gắn cảnh báo (ngồi bình thường → N theo ERNU)", "E7E6E6"),
            ("Khung vàng", "Nghiêng người: chỉ ghi tư thế", "FFF2CC"),
            ("Khung xanh nhạt", "Chưa đủ tư thế hoặc không xác định (→ U khi coverage thấp)", "D6EAF8"),
            ("E / R / N / U", "Nhãn cửa sổ 10s — xem sheet PREDICTED_WINDOWS và TONG_HOP_ERNU", None),
            ("BEEI", "Chỉ số bằng chứng hành vi (phụ); không phải điểm tập trung hay xếp hạng học sinh", None),
            ("ID7", "Chưa nhận diện được học sinh (số theo dõi tạm) — không gộp vào E/R/N/U", None),
        ]
        for label, meaning, color in box_notes:
            ws5.append([label, meaning])
            if color:
                ws5.cell(row=ws5.max_row, column=1).fill = PatternFill("solid", fgColor=color)
            ws5.cell(row=ws5.max_row, column=1).font = Font(bold=True)

        return wb
