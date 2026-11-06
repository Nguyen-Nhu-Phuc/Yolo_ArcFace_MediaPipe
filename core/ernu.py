"""Lớp diễn giải ERNU: B1–B5 + ngữ cảnh A1–A5 → nhãn cửa sổ E/R/N/U và chỉ số phiên.

Chế độ tự động (mặc định): không cần nhập thủ công A1–A5 hay xác nhận OFF.
  - Proxy: E→ON, R→OFF, N→UNCERTAIN, U→NO_DATA
  - Hết phiên: mỗi HS có focus_verdict ∈ {Tập trung, Không tập trung, Chưa đủ dữ liệu}
  - Viết bài (B2) được tính E ở mọi ngữ cảnh để chạy không cần bảng ACTIVITY_CONTEXT
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

# Ma trận mặc định (chế độ tự động: B2→E mọi A; B4 cúi đầu lâu → N, không tính không tập trung)
DEFAULT_MATRIX: dict[str, dict[str, str]] = {
    "B1": {a: "E" for a in ("A1", "A2", "A3", "A4", "A5")},
    "B2": {a: "E" for a in ("A1", "A2", "A3", "A4", "A5")},
    "B3": {a: "N" for a in ("A1", "A2", "A3", "A4", "A5")},
    "B4": {a: "N" for a in ("A1", "A2", "A3", "A4", "A5")},
    "B5": {a: "R" for a in ("A1", "A2", "A3", "A4", "A5")},
    "NORMAL": {a: "N" for a in ("A1", "A2", "A3", "A4", "A5")},
    UNKNOWN_SIGNAL: {a: "U" for a in ("A1", "A2", "A3", "A4", "A5")},
}
# Giữ tên cũ cho tương thích import trong test/code cũ
_MATRIX = DEFAULT_MATRIX

# Tự động: R → OFF (không cần giáo viên xác nhận)
DEFAULT_PROXY_MAPPING: dict[str, str] = {
    "E": "ON",
    "R": "OFF",
    "N": "UNCERTAIN",
    "U": "NO_DATA",
}

VALID_ACTIVITIES = frozenset({"A1", "A2", "A3", "A4", "A5"})
# Nhiệm vụ chưa xác nhận: không gán E* (N an toàn); vẫn cho phép R rõ (B5) và U
UNCONFIRMED_ACTIVITY = "CHUA_XAC_MINH"
LABELS = ("E", "R", "N", "U")
PROXY_LABELS = ("ON", "OFF", "PENDING", "UNCERTAIN", "NO_DATA")


@dataclass
class BehaviorInterval:
    """Khoảng thời gian hành vi thô theo student_id (giây video)."""

    student_id: str
    behavior: str  # tên VN hoặc B-code hoặc MISSING
    start_sec: float
    end_sec: float

    @property
    def b_code(self) -> str:
        if self.behavior in DEFAULT_MATRIX:
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
    proxy_label: str = ""  # ON | OFF | PENDING | UNCERTAIN | NO_DATA
    confirmed_off: bool = False  # True khi giáo viên xác nhận R → OFF

    def __post_init__(self) -> None:
        if not self.proxy_label:
            self.proxy_label = proxy_from_label(self.pred_label, self.confirmed_off)


@dataclass
class StudentMetrics:
    student_id: str
    T: float = 0.0
    T_E: float = 0.0
    T_R: float = 0.0
    T_N: float = 0.0
    T_U: float = 0.0
    T_ON: float = 0.0
    T_OFF: float = 0.0
    T_PENDING: float = 0.0
    T_UNCERTAIN: float = 0.0
    T_NO_DATA: float = 0.0
    C: float | None = None
    P_E: float | None = None
    P_R: float | None = None
    P_N: float | None = None
    P_U: float | None = None
    P_ON: float | None = None
    P_OFF: float | None = None
    P_PENDING: float | None = None
    P_UNCERTAIN: float | None = None
    P_NO_DATA: float | None = None
    data_status: str = "Chưa đủ dữ liệu"
    publishable: bool = False  # True khi đủ T_valid và C ≥ publish_c_min
    BEEI: float | None = None
    BEEI_wN0: float | None = None
    BEEI_wN1: float | None = None
    interpretation: str = ""
    focus_verdict: str = "Chưa đủ dữ liệu"  # Tập trung | Không tập trung | Chưa đủ dữ liệu
    focus_score: float | None = None  # 0–100, dựa trên BEEI / tỷ lệ ON–OFF
    windows: list[WindowLabel] = field(default_factory=list)

    @property
    def T_valid(self) -> float:
        return self.T_E + self.T_R + self.T_N


def proxy_from_label(
    pred_label: str,
    confirmed_off: bool = False,
    mapping: dict[str, str] | None = None,
) -> str:
    """Ánh xạ E/R/N/U → proxy. Mặc định R→OFF (tự động)."""
    lab = (pred_label or "").strip().upper()
    m = mapping or DEFAULT_PROXY_MAPPING
    # Tương thích API cũ: confirmed_off=False + mapping R=PENDING → giữ PENDING
    if lab == "R" and confirmed_off:
        return "OFF"
    if lab == "R" and confirmed_off is False and m.get("R") == "PENDING":
        return "PENDING"
    return m.get(lab, "UNCERTAIN")


def apply_review_confirmation(window: WindowLabel, confirmed_off: bool) -> WindowLabel:
    """Tùy chọn: ghi đè proxy cửa sổ R (OFF hoặc PENDING). Không đổi pred_label."""
    if window.pred_label != "R":
        return window
    window.confirmed_off = bool(confirmed_off)
    window.proxy_label = "OFF" if confirmed_off else "PENDING"
    return window


def compute_focus_verdict(
    *,
    publishable: bool,
    T_valid: float,
    T_ON: float,
    T_OFF: float,
    BEEI: float | None,
    focus_threshold: float = 50.0,
) -> tuple[str, float | None]:
    """Kết luận hết phiên: Tập trung / Không tập trung / Chưa đủ dữ liệu.

    Điểm = BEEI nếu có; không thì 100 * T_ON / (T_ON + T_OFF) (bỏ UNCERTAIN).
    >= focus_threshold → Tập trung; ngược lại → Không tập trung.
    """
    if not publishable or T_valid <= 0:
        return "Chưa đủ dữ liệu", None
    if BEEI is not None:
        score = float(BEEI)
    else:
        denom = T_ON + T_OFF
        score = round(100.0 * T_ON / denom, 1) if denom > 0 else 50.0
    if score >= focus_threshold:
        return "Tập trung", score
    return "Không tập trung", score


def get_matrix(cfg: ErnuConfig | None = None) -> dict[str, dict[str, str]]:
    if cfg is not None and getattr(cfg, "matrix", None):
        return cfg.matrix
    return DEFAULT_MATRIX


def get_proxy_mapping(cfg: ErnuConfig | None = None) -> dict[str, str]:
    if cfg is not None and getattr(cfg, "proxy_mapping", None):
        return cfg.proxy_mapping
    return dict(DEFAULT_PROXY_MAPPING)


def normalize_activity(code: str | None, default: str = "A1") -> str:
    c = (code or "").strip().upper()
    if c in VALID_ACTIVITIES:
        return c
    if c in (UNCONFIRMED_ACTIVITY, "UNKNOWN", "UNCONFIRMED", "?"):
        return UNCONFIRMED_ACTIVITY
    return default if default in VALID_ACTIVITIES else "A1"


def matrix_label(
    behavior_code: str,
    activity_code: str,
    matrix: dict[str, dict[str, str]] | None = None,
) -> str:
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
    mat = matrix or DEFAULT_MATRIX
    row = mat.get(behavior_code)
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
    mat = get_matrix(cfg)
    proxy_map = get_proxy_mapping(cfg)

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
        if code not in mat and code not in BEHAVIOR_TO_B.values():
            continue
        if not _event_meets_duration(ev, code, cfg):
            continue
        if code not in codes:
            codes.append(code)
        lab = matrix_label(code, act, mat)
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
        proxy_label=proxy_from_label(label, False, proxy_map),
        confirmed_off=False,
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
        "Kết quả không phản ánh trực tiếp tỷ lệ tập trung nhận thức. "
        "Proxy ON/PENDING/UNCERTAIN/NO_DATA chỉ diễn giải hành vi theo nhiệm vụ; "
        "OFF chỉ khi được xác nhận xem lại."
    )


def metrics_from_windows(
    student_id: str,
    windows: list[WindowLabel],
    cfg: ErnuConfig,
) -> StudentMetrics:
    m = StudentMetrics(student_id=student_id, windows=windows)
    T_seen = 0.0  # chỉ các cửa sổ có nhìn thấy HS (coverage > 0)
    for w in windows:
        dur = max(0.0, w.end_sec - w.start_sec)
        m.T += dur
        if w.valid_coverage > 0:
            T_seen += dur
        if w.pred_label == "E":
            m.T_E += dur
        elif w.pred_label == "R":
            m.T_R += dur
        elif w.pred_label == "N":
            m.T_N += dur
        else:
            m.T_U += dur
        px = w.proxy_label or proxy_from_label(w.pred_label, w.confirmed_off, get_proxy_mapping(cfg))
        if px == "ON":
            m.T_ON += dur
        elif px == "OFF":
            m.T_OFF += dur
        elif px == "PENDING":
            m.T_PENDING += dur
        elif px == "NO_DATA":
            m.T_NO_DATA += dur
        else:
            m.T_UNCERTAIN += dur

    m.T = round(m.T, 3)
    m.T_E = round(m.T_E, 3)
    m.T_R = round(m.T_R, 3)
    m.T_N = round(m.T_N, 3)
    m.T_U = round(m.T_U, 3)
    m.T_ON = round(m.T_ON, 3)
    m.T_OFF = round(m.T_OFF, 3)
    m.T_PENDING = round(m.T_PENDING, 3)
    m.T_UNCERTAIN = round(m.T_UNCERTAIN, 3)
    m.T_NO_DATA = round(m.T_NO_DATA, 3)

    if m.T <= 0 or m.T_valid <= 0:
        m.data_status = "Chưa đủ dữ liệu"
        m.publishable = False
        m.C = round(100.0 * m.T_valid / m.T, 1) if m.T > 0 else None
        m.P_U = round(100.0 * m.T_U / m.T, 1) if m.T > 0 else None
        m.P_NO_DATA = round(100.0 * m.T_NO_DATA / m.T, 1) if m.T > 0 else None
        m.interpretation = interpret_metrics(
            T=m.T, T_E=m.T_E, T_R=m.T_R, T_N=m.T_N, T_U=m.T_U,
            C=m.C, P_E=m.P_E, P_R=m.P_R, P_N=m.P_N, P_U=m.P_U,
            data_status=m.data_status, publishable=m.publishable,
        )
        m.focus_verdict, m.focus_score = compute_focus_verdict(
            publishable=False, T_valid=0, T_ON=m.T_ON, T_OFF=m.T_OFF, BEEI=None,
            focus_threshold=getattr(cfg, "focus_threshold", 50.0),
        )
        return m

    m.data_status = "Đủ dữ liệu"
    m.C = round(100.0 * m.T_valid / m.T, 1)
    m.P_E = round(100.0 * m.T_E / m.T_valid, 1)
    m.P_R = round(100.0 * m.T_R / m.T_valid, 1)
    m.P_N = round(100.0 * m.T_N / m.T_valid, 1)
    m.P_U = round(100.0 * m.T_U / m.T, 1)
    # Proxy ratios: ON/OFF/PENDING/UNCERTAIN trên T_valid; NO_DATA trên T
    m.P_ON = round(100.0 * m.T_ON / m.T_valid, 1)
    m.P_OFF = round(100.0 * m.T_OFF / m.T_valid, 1)
    m.P_PENDING = round(100.0 * m.T_PENDING / m.T_valid, 1)
    m.P_UNCERTAIN = round(100.0 * m.T_UNCERTAIN / m.T_valid, 1)
    m.P_NO_DATA = round(100.0 * m.T_NO_DATA / m.T, 1)
    m.publishable = (m.C is not None and m.C >= cfg.publish_c_min * 100.0)
    m.BEEI = beei_score(m.T_E, m.T_R, m.T_N, cfg.beei_w_e, cfg.beei_w_r, cfg.beei_w_n)
    m.BEEI_wN0 = beei_score(m.T_E, m.T_R, m.T_N, cfg.beei_w_e, cfg.beei_w_r, 0.0)
    m.BEEI_wN1 = beei_score(m.T_E, m.T_R, m.T_N, cfg.beei_w_e, cfg.beei_w_r, 1.0)
    m.interpretation = interpret_metrics(
        T=m.T, T_E=m.T_E, T_R=m.T_R, T_N=m.T_N, T_U=m.T_U,
        C=m.C, P_E=m.P_E, P_R=m.P_R, P_N=m.P_N, P_U=m.P_U,
        data_status=m.data_status, publishable=m.publishable,
    )
    # Kết luận tập trung: theo thời gian ĐÃ THẤY học sinh, không phạt đoạn video chưa xuất hiện.
    # UI nhận diện ≠ coverage cả phiên; C_present = T_valid / T_seen.
    C_present = round(100.0 * m.T_valid / T_seen, 1) if T_seen > 0 else None
    focus_min = float(getattr(cfg, "focus_min_valid_sec", 20.0))
    focus_ok = (
        m.T_valid + 1e-9 >= focus_min
        and C_present is not None
        and C_present >= cfg.publish_c_min * 100.0
    ) or bool(m.publishable)
    m.focus_verdict, m.focus_score = compute_focus_verdict(
        publishable=focus_ok,
        T_valid=m.T_valid,
        T_ON=m.T_ON,
        T_OFF=m.T_OFF,
        BEEI=m.BEEI,
        focus_threshold=getattr(cfg, "focus_threshold", 50.0),
    )
    return m


def _proxy_fields_from_students(students: Iterable[StudentMetrics]) -> dict:
    T_ON = T_OFF = T_PENDING = T_UNCERTAIN = T_NO_DATA = 0.0
    T = T_valid = 0.0
    for s in students:
        T += s.T
        T_valid += s.T_valid
        T_ON += s.T_ON
        T_OFF += s.T_OFF
        T_PENDING += s.T_PENDING
        T_UNCERTAIN += s.T_UNCERTAIN
        T_NO_DATA += s.T_NO_DATA
    out = {
        "T_ON": round(T_ON, 3),
        "T_OFF": round(T_OFF, 3),
        "T_PENDING": round(T_PENDING, 3),
        "T_UNCERTAIN": round(T_UNCERTAIN, 3),
        "T_NO_DATA": round(T_NO_DATA, 3),
        "P_ON": None,
        "P_OFF": None,
        "P_PENDING": None,
        "P_UNCERTAIN": None,
        "P_NO_DATA": round(100.0 * T_NO_DATA / T, 1) if T > 0 else None,
    }
    if T_valid > 0:
        out["P_ON"] = round(100.0 * T_ON / T_valid, 1)
        out["P_OFF"] = round(100.0 * T_OFF / T_valid, 1)
        out["P_PENDING"] = round(100.0 * T_PENDING / T_valid, 1)
        out["P_UNCERTAIN"] = round(100.0 * T_UNCERTAIN / T_valid, 1)
    return out


def aggregate_class_metrics(students: Iterable[StudentMetrics], cfg: ErnuConfig) -> dict:
    """Tổng hợp lớp theo giây (không theo số cửa sổ)."""
    students = list(students)
    T = T_E = T_R = T_N = T_U = 0.0
    for s in students:
        T += s.T
        T_E += s.T_E
        T_R += s.T_R
        T_N += s.T_N
        T_U += s.T_U
    T_valid = T_E + T_R + T_N
    proxy = _proxy_fields_from_students(students)
    out: dict = {
        "T": round(T, 3),
        "T_E": round(T_E, 3),
        "T_R": round(T_R, 3),
        "T_N": round(T_N, 3),
        "T_U": round(T_U, 3),
        "T_valid": round(T_valid, 3),
        **proxy,
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
        "proxy_mapping": get_proxy_mapping(cfg),
        "proxy_note": (
            "Tự động: E→ON (tập trung), R→OFF (không tập trung), "
            "N→UNCERTAIN, U→NO_DATA. Hết phiên: focus_verdict theo điểm ≥ ngưỡng."
        ),
        "focus_threshold": getattr(cfg, "focus_threshold", 50.0),
        "n_tap_trung": sum(1 for s in students if s.focus_verdict == "Tập trung"),
        "n_khong_tap_trung": sum(1 for s in students if s.focus_verdict == "Không tập trung"),
        "n_chua_du_du_lieu": sum(1 for s in students if s.focus_verdict == "Chưa đủ dữ liệu"),
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
    review_confirmations: list[dict] | None = None,
) -> tuple[list[WindowLabel], list[StudentMetrics], dict]:
    """Chạy toàn bộ pipeline cửa sổ → metrics từng HS → tổng hợp lớp.

    review_confirmations: [{student_id, start_sec, end_sec, confirmed_off}, ...]
    khớp cửa sổ R theo chồng thời gian lớn nhất → apply_review_confirmation.
    """
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

    if review_confirmations:
        apply_confirmations_to_session(all_windows, metrics, review_confirmations, cfg)

    return all_windows, metrics, aggregate_class_metrics(metrics, cfg)


def apply_confirmations_to_session(
    windows: list[WindowLabel],
    metrics: list[StudentMetrics],
    confirmations: list[dict],
    cfg: ErnuConfig,
) -> None:
    """Áp xác nhận OFF lên cửa sổ R khớp nhất; tính lại metrics theo student."""
    by_sid: dict[str, list[WindowLabel]] = {}
    for w in windows:
        by_sid.setdefault(w.student_id, []).append(w)

    for conf in confirmations:
        sid = str(conf.get("student_id") or "")
        s = conf.get("start_sec")
        e = conf.get("end_sec")
        if not sid or s is None or e is None:
            continue
        try:
            s_f, e_f = float(s), float(e)
        except (TypeError, ValueError):
            continue
        confirmed_off = bool(conf.get("confirmed_off", True))
        best = None
        best_ov = 0.0
        for w in by_sid.get(sid, []):
            if w.pred_label != "R":
                continue
            ov = overlap_sec(s_f, e_f, w.start_sec, w.end_sec)
            if ov > best_ov:
                best_ov = ov
                best = w
        if best is not None and best_ov > 0:
            apply_review_confirmation(best, confirmed_off)

    # Tính lại metrics sau xác nhận
    metrics.clear()
    for sid, sw in by_sid.items():
        metrics.append(metrics_from_windows(sid, sw, cfg))
