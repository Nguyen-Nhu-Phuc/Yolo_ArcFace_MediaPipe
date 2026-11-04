"""Lớp diễn giải ERNU: B1–B5 + ngữ cảnh A1–A5 → nhãn cửa sổ E/R/N/U và chỉ số phiên.

Tách khỏi detector hành vi thô. Không đồng nhất tư thế quan sát với mức tập trung nhận thức.
Chỉ số BEEI (nếu dùng) chỉ là chỉ báo phụ cho dashboard/nghiên cứu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from core.config import ErnuConfig

BEHAVIOR_TO_B = {
    "Giơ tay": "B1",
    "Viết bài": "B2",
    "Quay đầu": "B3",
    "Cúi đầu lâu": "B4",
    "Gục xuống bàn": "B5",
}
B_TO_NAME = {v: k for k, v in BEHAVIOR_TO_B.items()}
UNKNOWN_SIGNAL = "MISSING"  # thiếu quan sát / không xác định

# Ma trận diễn giải: behavior_code × activity → E | R | N | U
# B1→E*; B2→E chỉ A2 còn lại N*; B3→N; B4→R* ở A1/A4 còn lại N; B5→R*; ngồi→N; thiếu→U
_MATRIX: dict[str, dict[str, str]] = {
    "B1": {a: "E" for a in ("A1", "A2", "A3", "A4", "A5")},
    "B2": {"A1": "N", "A2": "E", "A3": "N", "A4": "N", "A5": "N"},
    "B3": {a: "N" for a in ("A1", "A2", "A3", "A4", "A5")},
    "B4": {"A1": "R", "A2": "N", "A3": "N", "A4": "R", "A5": "N"},
    "B5": {a: "R" for a in ("A1", "A2", "A3", "A4", "A5")},
    "NORMAL": {a: "N" for a in ("A1", "A2", "A3", "A4", "A5")},
    UNKNOWN_SIGNAL: {a: "U" for a in ("A1", "A2", "A3", "A4", "A5")},
}

VALID_ACTIVITIES = frozenset({"A1", "A2", "A3", "A4", "A5"})
# Nhiệm vụ chưa xác nhận: không gán E* (N an toàn); vẫn cho phép R rõ (B5) và U
UNCONFIRMED_ACTIVITY = "CHUA_XAC_MINH"
LABELS = ("E", "R", "N", "U")


@dataclass
class BehaviorInterval:
    """Khoảng thời gian hành vi thô theo student_id (giây video)."""

    student_id: str
    behavior: str  # tên VN hoặc B-code hoặc MISSING
    start_sec: float
    end_sec: float

    @property
    def b_code(self) -> str:
        if self.behavior in _MATRIX:
            return self.behavior
        return BEHAVIOR_TO_B.get(self.behavior, self.behavior)


@dataclass
class ObservationInterval:
    student_id: str
    start_sec: float
    end_sec: float


@dataclass
class WindowLabel:
    video_id: str
    student_id: str
    start_sec: float
    end_sec: float
    activity_code: str
    behavior_events: list[str]
    valid_coverage: float
    pred_label: str  # E | R | N | U
    reason: str
    mixed: bool
    rules_version: str


@dataclass
class StudentMetrics:
    student_id: str
    T: float = 0.0
    T_E: float = 0.0
    T_R: float = 0.0
    T_N: float = 0.0
    T_U: float = 0.0
    C: float | None = None
    P_E: float | None = None
    P_R: float | None = None
    P_N: float | None = None
    P_U: float | None = None
    data_status: str = "Chưa đủ dữ liệu"
    publishable: bool = False  # True khi đủ T_valid và C ≥ publish_c_min
    BEEI: float | None = None
    BEEI_wN0: float | None = None
    BEEI_wN1: float | None = None
    interpretation: str = ""
    windows: list[WindowLabel] = field(default_factory=list)

    @property
    def T_valid(self) -> float:
        return self.T_E + self.T_R + self.T_N


def normalize_activity(code: str | None, default: str = "A1") -> str:
    c = (code or "").strip().upper()
    if c in VALID_ACTIVITIES:
        return c
    if c in (UNCONFIRMED_ACTIVITY, "UNKNOWN", "UNCONFIRMED", "?"):
        return UNCONFIRMED_ACTIVITY
    return default if default in VALID_ACTIVITIES else "A1"


def matrix_label(behavior_code: str, activity_code: str) -> str:
    """Tra cứu một ô ma trận; hành vi lạ → N.

    Khi activity = CHUA_XAC_MINH: không gán E (thiếu xác nhận nhiệm vụ);
    B5 vẫn R*; còn lại N (nhãn mặc định an toàn).
    """
    act = normalize_activity(activity_code)
    if act == UNCONFIRMED_ACTIVITY:
        if behavior_code == "B5":
            return "R"
        if behavior_code == UNKNOWN_SIGNAL:
            return "U"
        return "N"
    row = _MATRIX.get(behavior_code)
    if row is None:
        return "N"
    return row.get(act, "N")


def overlap_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def merge_intervals(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []
    spans = sorted((min(a, b), max(a, b)) for a, b in spans if b > a)
    if not spans:
        return []
    out = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def covered_sec(spans: list[tuple[float, float]], start: float, end: float) -> float:
    total = 0.0
    for s, e in merge_intervals(spans):
        total += overlap_sec(s, e, start, end)
    return total


def build_window_bounds(
    duration_sec: float,
    window_sec: float,
    activity_segments: list[dict] | None = None,
    default_activity: str = "A1",
) -> list[tuple[float, float, str]]:
    """Cửa sổ không chồng; tách tại mốc đổi A. Trả về (start, end, activity_code)."""
    if duration_sec <= 0 or window_sec <= 0:
        return []

    # Thu thập các mốc cắt: 0, duration, biên hoạt động, và lưới window_sec
    cuts = {0.0, float(duration_sec)}
    segs = []
    for seg in activity_segments or []:
        try:
            s = float(seg["start_sec"])
            e = float(seg["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        raw = str(seg.get("activity_code") or "").strip().upper()
        if raw in VALID_ACTIVITIES or raw in (UNCONFIRMED_ACTIVITY, "UNKNOWN", "UNCONFIRMED", "?"):
            code = normalize_activity(raw, default_activity)
        else:
            code = normalize_activity(default_activity)
        segs.append((s, e, code))
        if 0 < s < duration_sec:
            cuts.add(s)
        if 0 < e < duration_sec:
            cuts.add(e)

    t = 0.0
    while t < duration_sec:
        cuts.add(t)
        t += window_sec
    ordered = sorted(cuts)

    def activity_at(t0: float, t1: float) -> str:
        mid = (t0 + t1) / 2.0
        for s, e, code in segs:
            if s <= mid < e or (mid == e and e == duration_sec):
                return normalize_activity(code, default_activity)
        return normalize_activity(default_activity)

    windows: list[tuple[float, float, str]] = []
    for i in range(len(ordered) - 1):
        a, b = ordered[i], ordered[i + 1]
        if b - a <= 1e-9:
            continue
        # Lưới 10s: nếu đoạn cắt dài hơn window_sec (không có mốc A), chia tiếp
        cursor = a
        while cursor < b - 1e-9:
            end = min(cursor + window_sec, b)
            if end - cursor > 1e-9:
                windows.append((cursor, end, activity_at(cursor, end)))
            cursor = end
    return windows


def _event_meets_duration(ev: BehaviorInterval, code: str, cfg: ErnuConfig) -> bool:
    """Sự kiện thô phải đủ thời lượng tối thiểu (B1/B3/B4/B5) mới đưa vào diễn giải."""
    need = cfg.min_duration(code)
    if need <= 0:
        return True
    return (ev.end_sec - ev.start_sec) + 1e-9 >= need


def label_window(
    *,
    video_id: str,
    student_id: str,
    start_sec: float,
    end_sec: float,
    activity_code: str,
    behavior_events: list[BehaviorInterval],
    observation_spans: list[tuple[float, float]],
    cfg: ErnuConfig,
) -> WindowLabel:
    """Gán một nhãn E/R/N/U cho cửa sổ theo bộ quy tắc ERNU.

    Bước: coverage → U nếu < C_min; ma trận A×B; khử xung đột B2+B4 ở A2;
    E và R cùng lúc không phân định → N + mixed; còn lại N.
    """
    win_len = max(end_sec - start_sec, 1e-9)
    act = normalize_activity(activity_code, cfg.default_activity)
    obs = covered_sec(observation_spans, start_sec, end_sec)
    coverage = min(1.0, obs / win_len)

    # Sự kiện hành vi chồng cửa sổ (bỏ MISSING khỏi evidence E/R)
    codes: list[str] = []
    has_missing = False
    e_codes: list[str] = []
    r_codes: list[str] = []
    n_codes: list[str] = []
    writing_e = False  # B2→E trong A2

    for ev in behavior_events:
        ov = overlap_sec(ev.start_sec, ev.end_sec, start_sec, end_sec)
        if ov <= 0:
            continue
        code = ev.b_code
        if code == UNKNOWN_SIGNAL or ev.behavior in ("Không xác định", "MISSING"):
            has_missing = True
            if "MISSING" not in codes:
                codes.append("MISSING")
            continue
        if code not in _MATRIX and code not in BEHAVIOR_TO_B.values():
            continue
        if not _event_meets_duration(ev, code, cfg):
            continue
        if code not in codes:
            codes.append(code)
        lab = matrix_label(code, act)
        if code == "B2" and lab == "E":
            writing_e = True
        if lab == "E":
            if code not in e_codes:
                e_codes.append(code)
        elif lab == "R":
            if code not in r_codes:
                r_codes.append(code)
        else:
            if code not in n_codes:
                n_codes.append(code)

    # Khử xung đột: B2 hợp lệ (E) trong A2 không bị biến thành R chỉ do cúi đầu (B4)
    if writing_e:
        r_codes = [c for c in r_codes if c != "B4"]

    mixed = False
    reason_parts: list[str] = []

    if coverage < cfg.c_min:
        label = "U"
        reason_parts.append(f"coverage={coverage:.2f}<C_min={cfg.c_min:.2f}")
    elif e_codes and r_codes:
        # E và R cùng xuất hiện, không phân định → N + mixed
        label = "N"
        mixed = True
        reason_parts.append(f"mixed E{e_codes}+R{r_codes}")
    elif e_codes:
        label = "E"
        reason_parts.append(f"evidence {e_codes} in {act}")
    elif r_codes:
        label = "R"
        reason_parts.append(f"review {r_codes} in {act}")
    elif has_missing and not (e_codes or r_codes or n_codes) and coverage >= cfg.c_min:
        # Có quan sát một phần nhưng tín hiệu chính là thiếu — vẫn N nếu coverage đủ
        # (U đã xử lý khi coverage thấp). Ngồi/thiếu tín hiệu rõ → N.
        label = "N"
        reason_parts.append("observed but insufficient evidence")
    else:
        label = "N"
        if n_codes:
            reason_parts.append(f"signals {n_codes}→N in {act}")
        else:
            reason_parts.append(f"normal sitting / no E-R evidence in {act}")

    return WindowLabel(
        video_id=video_id,
        student_id=student_id,
        start_sec=round(start_sec, 3),
        end_sec=round(end_sec, 3),
        activity_code=act,
        behavior_events=codes,
        valid_coverage=round(coverage, 4),
        pred_label=label,
        reason="; ".join(reason_parts),
        mixed=mixed,
        rules_version=cfg.rules_version,
    )


def beei_score(T_E: float, T_R: float, T_N: float, w_e: float, w_r: float, w_n: float) -> float | None:
    t_valid = T_E + T_R + T_N
    if t_valid <= 0:
        return None
    return round(100.0 * (w_e * T_E + w_r * T_R + w_n * T_N) / t_valid, 1)


def interpret_metrics(
    *,
    T: float,
    T_E: float,
    T_R: float,
    T_N: float,
    T_U: float,
    C: float | None,
    P_E: float | None,
    P_R: float | None,
    P_N: float | None,
    P_U: float | None,
    data_status: str,
    publishable: bool,
) -> str:
    """Đoạn diễn giải mẫu theo mục 6 bộ tiêu chí — không khẳng định tập trung nhận thức."""
    if data_status == "Chưa đủ dữ liệu" or T_E + T_R + T_N <= 0:
        return (
            "Chưa đủ dữ liệu quan sát hợp lệ để tính P_E/P_R/P_N. "
            "Không gán các tỷ lệ này bằng 0%. Kết quả không phản ánh trực tiếp tỷ lệ tập trung nhận thức."
        )
    minutes = T / 60.0 if T > 0 else 0.0
    pub = (
        "Đủ điều kiện công bố thống kê theo ngưỡng thiết kế."
        if publishable
        else "Coverage thấp hoặc chưa đạt ngưỡng công bố — không dùng chỉ số đơn độc / không xếp mức."
    )
    return (
        f"Trong phiên khoảng {minutes:.1f} phút (T={T:.0f}s), dữ liệu quan sát hợp lệ chiếm "
        f"{C if C is not None else '—'}% thời lượng (C). Trong phần quan sát hợp lệ, "
        f"{P_E if P_E is not None else '—'}% thời gian có bằng chứng thực hiện hoạt động học tập phù hợp (P_E), "
        f"{P_R if P_R is not None else '—'}% có tín hiệu cần giáo viên xem lại (P_R) và "
        f"{P_N if P_N is not None else '—'}% chưa đủ bằng chứng kết luận (P_N). "
        f"Thiếu quan sát P_U={P_U if P_U is not None else '—'}% trên toàn phiên "
        f"(T_U={T_U:.0f}s). {pub} "
        "Kết quả không phản ánh trực tiếp tỷ lệ tập trung nhận thức."
    )


def metrics_from_windows(
    student_id: str,
    windows: list[WindowLabel],
    cfg: ErnuConfig,
) -> StudentMetrics:
    m = StudentMetrics(student_id=student_id, windows=windows)
    for w in windows:
        dur = max(0.0, w.end_sec - w.start_sec)
        m.T += dur
        if w.pred_label == "E":
            m.T_E += dur
        elif w.pred_label == "R":
            m.T_R += dur
        elif w.pred_label == "N":
            m.T_N += dur
        else:
            m.T_U += dur

    m.T = round(m.T, 3)
    m.T_E = round(m.T_E, 3)
    m.T_R = round(m.T_R, 3)
    m.T_N = round(m.T_N, 3)
    m.T_U = round(m.T_U, 3)

    if m.T <= 0 or m.T_valid <= 0:
        m.data_status = "Chưa đủ dữ liệu"
        m.publishable = False
        m.C = round(100.0 * m.T_valid / m.T, 1) if m.T > 0 else None
        m.P_U = round(100.0 * m.T_U / m.T, 1) if m.T > 0 else None
        m.interpretation = interpret_metrics(
            T=m.T, T_E=m.T_E, T_R=m.T_R, T_N=m.T_N, T_U=m.T_U,
            C=m.C, P_E=m.P_E, P_R=m.P_R, P_N=m.P_N, P_U=m.P_U,
            data_status=m.data_status, publishable=m.publishable,
        )
        return m

    m.data_status = "Đủ dữ liệu"
    m.C = round(100.0 * m.T_valid / m.T, 1)
    m.P_E = round(100.0 * m.T_E / m.T_valid, 1)
    m.P_R = round(100.0 * m.T_R / m.T_valid, 1)
    m.P_N = round(100.0 * m.T_N / m.T_valid, 1)
    m.P_U = round(100.0 * m.T_U / m.T, 1)
    m.publishable = (m.C is not None and m.C >= cfg.publish_c_min * 100.0)
    m.BEEI = beei_score(m.T_E, m.T_R, m.T_N, cfg.beei_w_e, cfg.beei_w_r, cfg.beei_w_n)
    m.BEEI_wN0 = beei_score(m.T_E, m.T_R, m.T_N, cfg.beei_w_e, cfg.beei_w_r, 0.0)
    m.BEEI_wN1 = beei_score(m.T_E, m.T_R, m.T_N, cfg.beei_w_e, cfg.beei_w_r, 1.0)
    m.interpretation = interpret_metrics(
        T=m.T, T_E=m.T_E, T_R=m.T_R, T_N=m.T_N, T_U=m.T_U,
        C=m.C, P_E=m.P_E, P_R=m.P_R, P_N=m.P_N, P_U=m.P_U,
        data_status=m.data_status, publishable=m.publishable,
    )
    return m


def aggregate_class_metrics(students: Iterable[StudentMetrics], cfg: ErnuConfig) -> dict:
    """Tổng hợp lớp theo giây (không theo số cửa sổ)."""
    T = T_E = T_R = T_N = T_U = 0.0
    for s in students:
        T += s.T
        T_E += s.T_E
        T_R += s.T_R
        T_N += s.T_N
        T_U += s.T_U
    T_valid = T_E + T_R + T_N
    out: dict = {
        "T": round(T, 3),
        "T_E": round(T_E, 3),
        "T_R": round(T_R, 3),
        "T_N": round(T_N, 3),
        "T_U": round(T_U, 3),
        "T_valid": round(T_valid, 3),
        "C": None,
        "P_E": None,
        "P_R": None,
        "P_N": None,
        "P_U": None,
        "data_status": "Chưa đủ dữ liệu",
        "publishable": False,
        "BEEI": None,
        "BEEI_wN0": None,
        "BEEI_wN1": None,
        "interpretation": "",
        "beei_weights": {
            "default": {"w_E": cfg.beei_w_e, "w_R": cfg.beei_w_r, "w_N": cfg.beei_w_n},
            "sensitivity_wN": [0.0, cfg.beei_w_n, 1.0],
        },
    }
    if T <= 0:
        out["interpretation"] = interpret_metrics(
            T=0, T_E=0, T_R=0, T_N=0, T_U=0,
            C=None, P_E=None, P_R=None, P_N=None, P_U=None,
            data_status=out["data_status"], publishable=False,
        )
        return out
    out["C"] = round(100.0 * T_valid / T, 1)
    out["P_U"] = round(100.0 * T_U / T, 1)
    if T_valid <= 0:
        out["interpretation"] = interpret_metrics(
            T=out["T"], T_E=0, T_R=0, T_N=0, T_U=out["T_U"],
            C=out["C"], P_E=None, P_R=None, P_N=None, P_U=out["P_U"],
            data_status=out["data_status"], publishable=False,
        )
        return out
    out["data_status"] = "Đủ dữ liệu"
    out["P_E"] = round(100.0 * T_E / T_valid, 1)
    out["P_R"] = round(100.0 * T_R / T_valid, 1)
    out["P_N"] = round(100.0 * T_N / T_valid, 1)
    out["publishable"] = out["C"] is not None and out["C"] >= cfg.publish_c_min * 100.0
    out["BEEI"] = beei_score(T_E, T_R, T_N, cfg.beei_w_e, cfg.beei_w_r, cfg.beei_w_n)
    out["BEEI_wN0"] = beei_score(T_E, T_R, T_N, cfg.beei_w_e, cfg.beei_w_r, 0.0)
    out["BEEI_wN1"] = beei_score(T_E, T_R, T_N, cfg.beei_w_e, cfg.beei_w_r, 1.0)
    out["interpretation"] = interpret_metrics(
        T=out["T"], T_E=out["T_E"], T_R=out["T_R"], T_N=out["T_N"], T_U=out["T_U"],
        C=out["C"], P_E=out["P_E"], P_R=out["P_R"], P_N=out["P_N"], P_U=out["P_U"],
        data_status=out["data_status"], publishable=out["publishable"],
    )
    return out


def analyze_session(
    *,
    video_id: str,
    duration_sec: float,
    student_ids: list[str],
    behaviors: list[BehaviorInterval],
    observations: list[ObservationInterval],
    activity_segments: list[dict] | None = None,
    cfg: ErnuConfig | None = None,
) -> tuple[list[WindowLabel], list[StudentMetrics], dict]:
    """Chạy toàn bộ pipeline cửa sổ → metrics từng HS → tổng hợp lớp."""
    cfg = cfg or ErnuConfig()
    bounds = build_window_bounds(
        duration_sec, cfg.window_sec, activity_segments, cfg.default_activity
    )
    by_beh: dict[str, list[BehaviorInterval]] = {}
    for b in behaviors:
        by_beh.setdefault(b.student_id, []).append(b)
    by_obs: dict[str, list[tuple[float, float]]] = {}
    for o in observations:
        by_obs.setdefault(o.student_id, []).append((o.start_sec, o.end_sec))

    all_windows: list[WindowLabel] = []
    metrics: list[StudentMetrics] = []
    for sid in student_ids:
        sw: list[WindowLabel] = []
        for start, end, act in bounds:
            w = label_window(
                video_id=video_id,
                student_id=sid,
                start_sec=start,
                end_sec=end,
                activity_code=act,
                behavior_events=by_beh.get(sid, []),
                observation_spans=by_obs.get(sid, []),
                cfg=cfg,
            )
            sw.append(w)
            all_windows.append(w)
        metrics.append(metrics_from_windows(sid, sw, cfg))
    return all_windows, metrics, aggregate_class_metrics(metrics, cfg)
