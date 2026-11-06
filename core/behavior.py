"""Nhận biết hành vi từ keypoint COCO-17 (YOLO-pose).

Ngưỡng trong BehaviorConfig là giá trị khởi đầu: giữ cố định khi đo kết quả cuối;
chỉ hiệu chỉnh trên tập phát triển (eval_behavior.py), không tinh chỉnh theo từng video kiểm thử cuối.


Hậu xử lý, không đổi mô hình pose:
  - bỏ điểm khớp có độ tin cậy < 0,5; làm mượt tọa độ bằng EMA (0,6 cho giá trị mới)
  - chuẩn hóa theo bề rộng vai; tư thế nền chỉ lấy từ khung ổn định, đủ tin cậy
  - thiếu điểm khớp quan trọng → Không xác định (không kết luận bình thường hay bất thường)
  - năm hành vi: giơ tay, viết bài, quay đầu, cúi đầu lâu, gục xuống bàn
  - nghiêng người chỉ ghi tư thế, không tính là cảnh báo
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from core.config import Config

# (tên đầy đủ, nhãn viết tắt trên video). Hành vi cần xem đứng trước để chọn nhãn chính.
REASON_SHORT = (
    ("Gục xuống bàn", "Gục bàn"),
    ("Cúi đầu lâu", "Cúi lâu"),
    ("Không xác định", "Chưa rõ"),
    ("Giơ tay", "Giơ tay"),
    ("Viết bài", "Viết"),
    ("Quay đầu", "Quay"),
    ("Nghiêng người", "Nghiêng"),
)
# Nhãn cũ trong file gán nhãn → tên hiện tại
LABEL_ALIASES = {"Ngủ gục": "Gục xuống bàn"}

# ERNU: B1/B2 là tín hiệu có thể thành E tùy ngữ cảnh A; B3 quay đầu → luôn N (không xếp "activity").
ACTIVITY_BEHAVIORS = frozenset({"Giơ tay", "Viết bài"})
REVIEW_BEHAVIORS = frozenset({"Cúi đầu lâu", "Gục xuống bàn"})
# Quay đầu: tín hiệu trung tính theo ma trận ERNU (→ N); giữ riêng để báo cáo thô.
NEUTRAL_BEHAVIORS = frozenset({"Quay đầu"})
POSTURE_BEHAVIORS = frozenset({"Nghiêng người"})
UNKNOWN_BEHAVIOR = "Không xác định"
ALWAYS_ON = frozenset({UNKNOWN_BEHAVIOR})

_TIMED = (
    "Cúi đầu lâu", "Giơ tay", "Quay đầu", "Gục xuống bàn", "Nghiêng người",
)
_HEAD = range(5)  # mũi, mắt, tai


def canonical_behavior(name: str) -> str:
    return LABEL_ALIASES.get(name, name)


def reason_name(reason: str) -> str | None:
    """'Cúi đầu lâu (5.2s)' → 'Cúi đầu lâu'."""
    for name, _short in REASON_SHORT:
        if reason.startswith(name):
            return name
    return None


def observation_group(reasons: list[str]) -> str:
    """activity | review | posture | unknown | clear | neutral.

    Nhóm legacy cho khung video; diễn giải E/R/N/U nằm ở core.ernu theo cửa sổ + A1–A5.
    """
    names = {reason_name(r) for r in reasons}
    names.discard(None)
    if names & REVIEW_BEHAVIORS:
        return "review"
    if UNKNOWN_BEHAVIOR in names:
        return "unknown"
    if names & ACTIVITY_BEHAVIORS:
        return "activity"
    if names & NEUTRAL_BEHAVIORS:
        return "neutral"
    if names & POSTURE_BEHAVIORS:
        return "posture"
    return "clear"


def behavior_kind(name: str) -> str:
    name = canonical_behavior(name)
    if name in REVIEW_BEHAVIORS:
        return "Cần giáo viên xem"
    if name == UNKNOWN_BEHAVIOR:
        return "Không xác định"
    if name in ACTIVITY_BEHAVIORS:
        return "Hoạt động quan sát được"
    if name in NEUTRAL_BEHAVIORS:
        return "Tín hiệu trung tính (ERNU→N)"
    if name in POSTURE_BEHAVIORS:
        return "Tư thế ghi nhận"
    return "Khác"


def _group_of(name: str) -> str:
    if name in ACTIVITY_BEHAVIORS:
        return "activity"
    if name in REVIEW_BEHAVIORS:
        return "review"
    if name in NEUTRAL_BEHAVIORS:
        return "neutral"
    if name in POSTURE_BEHAVIORS:
        return "posture"
    if name == UNKNOWN_BEHAVIOR:
        return "unknown"
    return "clear"


def behavior_legend(config: Config) -> list[dict]:
    """Chú thích nhãn viết tắt kèm điều kiện nhận biết."""
    bh = config.behavior
    info = {
        "Gục xuống bàn": (
            f"Đầu thấp hoặc lệch sang vai và thân nghiêng, giữ ≥ {bh.desk_sleep_duration:g} giây. "
            "Không kết luận học sinh đang ngủ. Không báo khi đang viết bài."
        ),
        "Cúi đầu lâu": (
            f"Cúi đầu ≥ {bh.head_down_duration:g} giây so với tư thế nền của chính em đó. "
            "Nếu vừa bắt được viết bài, chỉ tính những giây nhìn thấy tay không còn viết; "
            "mất tay hoặc mất tư thế chưa phải chứng minh."
        ),
        "Không xác định": (
            "Che khuất, quay lưng hoặc thiếu điểm khớp quan trọng — không kết luận hành vi."
        ),
        "Giơ tay": (
            f"Cổ tay và khuỷu tay cao hơn vai, đủ tin cậy, giữ ≥ {bh.hand_raise_duration:g} giây."
        ),
        "Viết bài": (
            "Cúi đầu, cổ tay ở khu vực bàn và tay đang chuyển động. "
            "Học sinh ngồi gần camera được nới vùng tay vì góc nhìn làm tay xòe rộng và thấp hơn vai. "
            "Đã bắt được thì giữ nhãn viết bài. Chỉ đổi sang cúi đầu lâu khi có đủ giây "
            "nhìn thấy đầu vẫn cúi và tay không còn viết."
        ),
        "Quay đầu": (
            f"Lệch > {bh.head_rotation_threshold:g}° so với hướng nhìn thường ngày của chính em đó, "
            f"giữ ≥ {bh.head_rotation_duration:g} giây."
        ),
        "Nghiêng người": (
            f"Vai lệch > {bh.shoulder_tilt_threshold:g}° trong ≥ {bh.shoulder_tilt_duration:g} giây. "
            "Chỉ ghi tư thế, không tính là cảnh báo."
        ),
    }
    return [
        {
            "short": short,
            "name": name,
            "group": _group_of(name),
            "positive": name in ACTIVITY_BEHAVIORS,
            "condition": info[name],
            "enabled": bh.is_on(name),
            "toggle": name not in ALWAYS_ON,
        }
        for name, short in REASON_SHORT
    ]


# ---------------------------------------------------------------- đo trên 1 khung


def _ok(point, conf: float) -> bool:
    return point is not None and len(point) >= 3 and float(point[2]) >= conf


def shoulder_width(keypoints, conf: float = 0.5) -> float | None:
    ls, rs = keypoints[5], keypoints[6]
    if not _ok(ls, conf) or not _ok(rs, conf):
        return None
    sw = float(np.hypot(rs[0] - ls[0], rs[1] - ls[1]))
    return sw if sw > 1e-3 else None


def head_height(keypoints, conf: float = 0.5) -> float | None:
    """Độ cao mũi trên đường vai, theo bề rộng vai (ngồi thẳng ~0.5, cúi ≤ 0)."""
    nose = keypoints[0]
    sw = shoulder_width(keypoints, conf)
    if sw is None or not _ok(nose, conf):
        return None
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
    return float(shoulder_y - nose[1]) / sw


def pose_usable(keypoints, conf: float) -> bool:
    """Đủ vai để chuẩn hóa và còn thấy đầu. Quay lưng / thiếu khớp → không dùng."""
    if keypoints is None or len(keypoints) < 17:
        return False
    if shoulder_width(keypoints, conf) is None:
        return False
    return any(_ok(keypoints[i], conf) for i in _HEAD)


def is_head_down(keypoints, bh) -> bool:
    """Cúi đầu theo ngưỡng tuyệt đối (không cần lịch sử). False nếu chưa đủ điểm."""
    h = head_height(keypoints, bh.kp_min_conf)
    return h is not None and h < bh.head_down_ratio


def head_yaw(keypoints, signals: dict | None = None, conf: float = 0.5) -> tuple[float, str] | None:
    """(góc quay đầu theo độ, nguồn đo), dương = quay sang phải ảnh. Ưu tiên góc từ InsightFace.

    Không suy "quay hẳn" từ việc chỉ thấy 1 tai: HS ngồi xa thường bị tóc che tai.
    """
    if signals and signals.get("yaw") is not None:
        return float(signals["yaw"]), "face"
    nose, left_ear, right_ear = keypoints[0], keypoints[3], keypoints[4]
    if not _ok(nose, conf) or not _ok(left_ear, conf) or not _ok(right_ear, conf):
        return None
    inter_ear = abs(right_ear[0] - left_ear[0])
    if inter_ear <= 1e-3:
        return None
    offset = nose[0] - (left_ear[0] + right_ear[0]) / 2
    return float(np.degrees(np.arcsin(max(-1.0, min(1.0, offset / inter_ear))))), "keypoint"


def _head_point(keypoints, conf: float):
    nose = keypoints[0]
    if _ok(nose, conf):
        return float(nose[0]), float(nose[1])
    pts = [keypoints[i] for i in _HEAD if _ok(keypoints[i], conf)]
    if not pts:
        return None
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def _near_factor(sw: float, bh) -> float:
    """0 = ngồi xa, 1 = hàng đầu sát camera. Suy từ bề rộng vai trên ảnh."""
    span = max(bh.writing_near_sw - bh.writing_far_sw, 1.0)
    return float(min(1.0, max(0.0, (sw - bh.writing_far_sw) / span)))


def _writing_limits(sw: float, bh) -> tuple[float, float, float, float]:
    """(độ tụt tối thiểu, tối đa, lệch ngang tối đa, tốc độ tối thiểu).

    Ngồi gần: tay phía ống kính bị phóng so với vai, và một nét bút là ít bề rộng vai hơn.
    """
    near = _near_factor(sw, bh)
    min_drop = bh.writing_min_drop * (1.0 - 0.75 * near)
    max_drop = bh.writing_max_drop + 3.0 * near
    max_offset = bh.writing_max_offset + 1.2 * near
    min_speed = bh.writing_min_speed * (1.0 - 0.45 * near)
    return min_drop, max_drop, max_offset, min_speed


def _writing_hands(keypoints, bh) -> str:
    """'zone' nếu có cổ tay ở bàn, 'away' nếu thấy tay nhưng không ở bàn, 'unseen' nếu không thấy tay."""
    sw = shoulder_width(keypoints, bh.kp_min_conf)
    if sw is None:
        return "unseen"
    indexes = [9, 10]
    if _near_factor(sw, bh) >= 0.45:
        for wrist_i, elbow_i in ((9, 7), (10, 8)):
            if not _ok(keypoints[wrist_i], bh.writing_kp_min_conf):
                indexes.append(elbow_i)
    seen = False
    for index in indexes:
        wrist = keypoints[index]
        if not _ok(wrist, bh.writing_kp_min_conf):
            continue
        seen = True
        if _wrist_in_writing_zone(wrist, keypoints, sw, bh):
            return "zone"
    return "away" if seen else "unseen"


def _wrist_in_writing_zone(wrist, keypoints, sw: float, bh) -> bool:
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
    center_x = (keypoints[5][0] + keypoints[6][0]) / 2
    drop = (wrist[1] - shoulder_y) / sw
    offset = abs(wrist[0] - center_x) / sw
    min_drop, max_drop, max_offset, _speed = _writing_limits(sw, bh)
    return min_drop <= drop <= max_drop and offset <= max_offset


def _shoulder_tilt(keypoints, conf: float) -> float | None:
    if shoulder_width(keypoints, conf) is None:
        return None
    ls, rs = keypoints[5], keypoints[6]
    return float(np.degrees(np.arctan2(abs(rs[1] - ls[1]), abs(rs[0] - ls[0]))))


def _hand_raised(keypoints, conf: float) -> bool:
    """Cổ tay và khuỷu tay đều cao hơn vai cùng bên. Thiếu một điểm thì không kết luận."""
    for shoulder, elbow, wrist in (
        (keypoints[5], keypoints[7], keypoints[9]),
        (keypoints[6], keypoints[8], keypoints[10]),
    ):
        if not (_ok(shoulder, conf) and _ok(elbow, conf) and _ok(wrist, conf)):
            continue
        if wrist[1] < shoulder[1] and elbow[1] < shoulder[1]:
            return True
    return False


def desk_evidence(keypoints, bh) -> bool | None:
    """Gục xuống bàn cần đồng thời vị trí đầu và độ nghiêng thân.

    True = đủ hai nhóm dấu hiệu, False = đủ điểm nhưng không phải, None = thiếu điểm.
    Không dùng riêng mũi thấp hơn vai.
    """
    conf = bh.kp_min_conf
    sw = shoulder_width(keypoints, conf)
    head = _head_point(keypoints, conf)
    if sw is None or head is None:
        return None
    ls, rs = keypoints[5], keypoints[6]
    hx, hy = head
    sx = (float(ls[0]) + float(rs[0])) / 2
    sy = (float(ls[1]) + float(rs[1])) / 2
    head_rel = (hy - sy) / sw
    head_side = abs(hx - sx) / sw
    shoulder_angle = float(np.degrees(np.arctan2(abs(rs[1] - ls[1]), abs(rs[0] - ls[0]) + 1e-6)))
    head_bad = head_rel >= bh.desk_sleep_head_drop or (
        head_side >= bh.desk_sleep_head_side and head_rel >= -0.2
    )
    torso_bad = shoulder_angle >= bh.desk_sleep_shoulder_angle
    hips = [p for p in (keypoints[11], keypoints[12]) if _ok(p, conf)]
    if hips:
        hipx = float(np.mean([p[0] for p in hips]))
        hipy = float(np.mean([p[1] for p in hips]))
        torso_angle = float(np.degrees(np.arctan2(abs(sx - hipx), abs(hipy - sy) + 1e-6)))
        if torso_angle >= bh.desk_sleep_torso_angle and head_rel >= -0.2:
            torso_bad = True
    return bool(head_bad and torso_bad)


# ---------------------------------------------------------------- trạng thái qua nhiều khung


class Debounce:
    """Thời gian điều kiện đang đúng. Khoảng mất tín hiệu không cộng thêm thời gian."""

    __slots__ = ("on", "miss")

    def __init__(self):
        self.on = 0.0
        self.miss = 0.0

    def step(self, active: bool, dt: float, tolerance: float) -> float:
        if active:
            self.on += dt
            self.miss = 0.0
        else:
            self.miss += dt
            if self.miss > tolerance:
                self.on = 0.0
                self.miss = 0.0
        return self.on


class StudentStateTracker:
    """Trạng thái hành vi theo từng học sinh qua nhiều khung."""

    def __init__(self):
        self._timers: dict[tuple, Debounce] = defaultdict(Debounce)
        self.clock: dict = defaultdict(float)
        self.head_track: dict = {}
        self.head_heights: dict = {}
        self.turn_side: dict = {}
        self.yaw_history: dict = {}
        self.last_turn: dict = {}
        self.last_speed: dict = {}
        self.wrist_hist: dict = {}
        self.writing_latched: dict = {}
        self.writing_gap: dict = {}
        self.writing_soft: dict = {}
        self._smooth: dict = {}

    def forget(self, student_id) -> None:
        for keyed in (self._timers, self.yaw_history, self.wrist_hist):
            for k in [k for k in keyed if k[0] == student_id]:
                del keyed[k]
        for d in (self.clock, self.head_track, self.head_heights, self.turn_side,
                  self.last_turn, self.last_speed, self.writing_latched, self.writing_gap,
                  self.writing_soft, self._smooth):
            d.pop(student_id, None)

    def _drop_writing(self, student_id, reset_bow: bool) -> None:
        """Thả nhãn viết bài. reset_bow khi chưa đủ chứng minh cúi đầu, để không nhảy nhãn."""
        self.writing_latched.pop(student_id, None)
        self.writing_gap.pop(student_id, None)
        self.writing_soft.pop(student_id, None)
        for key in [k for k in self.wrist_hist if k[0] == student_id]:
            del self.wrist_hist[key]
        if not reset_bow:
            return
        bow = self._timers.get((student_id, "Cúi đầu lâu"))
        if bow is not None:
            bow.on = 0.0
            bow.miss = 0.0

    def pause_writing(self, student_id, dt: float, bh) -> None:
        """Mất đầu hoặc mất tư thế: giữ nhãn viết trong dung sai, rồi xóa và không cộng cúi đầu."""
        if not self.writing_latched.get(student_id):
            return
        self.writing_gap[student_id] = self.writing_gap.get(student_id, 0.0) + dt
        if self.writing_gap[student_id] > bh.miss_tolerance_sec:
            self._drop_writing(student_id, reset_bow=True)

    def timer(self, student_id, name: str, active: bool, dt: float, bh) -> float:
        return self._timers[(student_id, name)].step(active, dt, bh.miss_tolerance_sec)

    def decay(self, student_id, dt: float, bh) -> None:
        """Mất tư thế: không kéo dài hành vi, quá dung sai thì xóa bộ đếm."""
        self.pause_writing(student_id, dt, bh)
        for name in _TIMED:
            if bh.is_on(name):
                self.timer(student_id, name, False, dt, bh)

    def smooth(self, student_id, keypoints, dt: float, bh) -> np.ndarray:
        """Bỏ điểm < ngưỡng. Điểm đủ tin cậy thì EMA; mất quá dung sai thì không dùng điểm cũ."""
        raw = np.asarray(keypoints, dtype=np.float32)
        out = raw.copy()
        state = self._smooth.get(student_id)
        if state is None:
            state = self._smooth[student_id] = {
                "xy": np.full((17, 2), np.nan, np.float32),
                "age": np.full(17, 1e9, np.float32),
            }
        xy, age = state["xy"], state["age"]
        age += float(dt)
        alpha = float(bh.ema_alpha)
        for i in range(min(17, len(raw))):
            if not _ok(raw[i], bh.kp_min_conf):
                out[i, 2] = 0.0
                continue
            if np.isnan(xy[i, 0]) or age[i] > bh.miss_tolerance_sec:
                xy[i, 0], xy[i, 1] = raw[i, 0], raw[i, 1]
            else:
                xy[i, 0] = alpha * raw[i, 0] + (1.0 - alpha) * xy[i, 0]
                xy[i, 1] = alpha * raw[i, 1] + (1.0 - alpha) * xy[i, 1]
            age[i] = 0.0
            out[i, 0], out[i, 1] = xy[i, 0], xy[i, 1]
        return out

    def head_speed(self, student_id, keypoints, dt: float, bh) -> float | None:
        """Tốc độ đầu trong cửa sổ gần nhất, theo bề rộng vai / giây."""
        self.clock[student_id] += dt
        t = self.clock[student_id]
        conf = bh.kp_min_conf
        head = _head_point(keypoints, conf)
        sw = shoulder_width(keypoints, conf)
        track = self.head_track.get(student_id)
        if track is None:
            track = self.head_track[student_id] = deque(maxlen=64)
        if head is not None and sw is not None:
            track.append((t, head[0], head[1], sw))
        while len(track) > 3 and t - track[1][0] >= bh.stillness_window:
            track.popleft()
        speed = None
        if len(track) >= 3:
            elapsed = track[-1][0] - track[0][0]
            if elapsed > 0:
                sw_mean = float(np.mean([p[3] for p in track]))
                path = sum(
                    float(np.hypot(b[1] - a[1], b[2] - a[2]))
                    for a, b in zip(track, list(track)[1:])
                )
                speed = path / sw_mean / elapsed
        self.last_speed[student_id] = speed
        return speed

    def _stable(self, student_id, bh) -> bool:
        speed = self.last_speed.get(student_id)
        return speed is not None and speed <= bh.baseline_max_speed

    def head_down(self, student_id, keypoints, bh, pitch: float | None) -> bool | None:
        """True/False khi đo được; None khi thiếu điểm. Tư thế nền không nhận khung đang cúi."""
        conf = bh.kp_min_conf
        h = head_height(keypoints, conf)
        down = None
        near = False
        if h is not None:
            hist = self.head_heights.get(student_id)
            if hist is None:
                hist = self.head_heights[student_id] = deque(maxlen=bh.head_baseline_window)
            if len(hist) >= bh.head_baseline_min_samples:
                base = float(np.percentile(hist, 75))
                down = h < base - bh.head_down_drop
                # Chỉ khung sát tư thế nền mới được cập nhật, tránh tư thế đang cúi kéo mức tham chiếu.
                near = abs(h - base) <= bh.head_down_drop * 0.5
            else:
                down = h < bh.head_down_ratio
                near = not down
        if pitch is not None and pitch >= bh.head_down_pitch:
            down = True
            near = False
        if h is not None and down is False and near and self._stable(student_id, bh):
            self.head_heights[student_id].append(h)
        return down

    def turn_angle(self, student_id, measured: tuple[float, str] | None, bh,
                   head_is_down: bool) -> float | None:
        """Độ quay so với hướng nhìn thường ngày. Chưa đủ tư thế nền thì không kết luận.

        Chỉ đưa vào tư thế nền những khung ổn định, không cúi, và không lệch xa mức hiện có.
        """
        if measured is None:
            self.last_turn.pop(student_id, None)
            return None
        yaw, source = measured
        hist = self.yaw_history.get((student_id, source))
        if hist is None:
            hist = self.yaw_history[(student_id, source)] = deque(maxlen=bh.head_baseline_window)
        turn = None
        if len(hist) >= bh.head_baseline_min_samples:
            turn = yaw - float(np.median(hist))
        stable = self._stable(student_id, bh) and not head_is_down
        if stable and (turn is None or abs(turn) <= bh.baseline_yaw_band):
            hist.append(yaw)
        if turn is None:
            self.last_turn.pop(student_id, None)
        else:
            self.last_turn[student_id] = turn
        return turn

    def writing(self, student_id, keypoints, bh, head_is_down: bool) -> bool:
        """Cúi đầu + tay ở bàn + tay đang chuyển động. Cúi đầu một mình không đủ.

        Vẫn ghi lịch sử cổ tay khi đầu chưa bị xếp là cúi, để một khung lệch ngưỡng
        không làm mất nhịp vừa viết. writing_soft: nhịp tay độ tin cậy thấp hơn,
        chỉ dùng để gia hạn nhãn khi đã bắt được viết bài.
        """
        sw = shoulder_width(keypoints, bh.kp_min_conf)
        soft = False
        strict = False
        if sw is not None:
            t = self.clock[student_id]
            wrist_conf = bh.writing_kp_min_conf
            _min_drop, _max_drop, _max_offset, min_speed = _writing_limits(sw, bh)
            # Cổ tay hàng đầu hay bị bàn che hoặc sát mép khung; khuỷu vẫn thấy.
            indexes = [9, 10]
            if _near_factor(sw, bh) >= 0.45:
                for wrist_i, elbow_i in ((9, 7), (10, 8)):
                    if not _ok(keypoints[wrist_i], wrist_conf):
                        indexes.append(elbow_i)
            for index in indexes:
                wrist = keypoints[index]
                key = (student_id, index)
                hist = self.wrist_hist.get(key)
                if hist is None:
                    hist = self.wrist_hist[key] = deque(maxlen=48)
                if not _ok(wrist, wrist_conf):
                    continue
                hist.append((t, float(wrist[0]), float(wrist[1]), sw))
                if not head_is_down or not _wrist_in_writing_zone(wrist, keypoints, sw, bh):
                    continue
                pts = [p for p in hist if t - p[0] <= bh.writing_move_window]
                if len(pts) < 2:
                    continue
                elapsed = pts[-1][0] - pts[0][0]
                if elapsed < 0.4:
                    continue
                path = sum(
                    float(np.hypot(b[1] - a[1], b[2] - a[2]))
                    for a, b in zip(pts, pts[1:])
                )
                sw_mean = float(np.mean([p[3] for p in pts]))
                if path / sw_mean / elapsed < min_speed:
                    continue
                soft = True
                if _ok(wrist, bh.kp_min_conf):
                    strict = True
        self.writing_soft[student_id] = soft
        return strict


def _emit(reasons: list[str], active: bool, held: float, need: float, text: str) -> None:
    """Chỉ gắn nhãn khi điều kiện đang đúng. Khoảng mất tín hiệu không kéo dài nhãn."""
    if active and held >= need:
        reasons.append(text)


def detect_behaviors(keypoints, config: Config,
                     state_tracker: StudentStateTracker,
                     student_id, fps: float,
                     signals: dict | None = None) -> list[str]:
    """Danh sách hành vi đang xảy ra.

    signals: yaw, pitch từ InsightFace (độ;
    yaw dương = quay sang phải ảnh, pitch dương = cúi xuống).
    """
    bh = config.behavior
    signals = signals or {}
    dt = 1.0 / max(float(fps), 1e-6)
    if keypoints is None or len(keypoints) < 17:
        state_tracker.decay(student_id, dt, bh)
        return [UNKNOWN_BEHAVIOR]

    kpts = state_tracker.smooth(student_id, keypoints, dt, bh)
    state_tracker.head_speed(student_id, kpts, dt, bh)
    if not pose_usable(kpts, bh.kp_min_conf):
        state_tracker.decay(student_id, dt, bh)
        return [UNKNOWN_BEHAVIOR]

    tol_timer = state_tracker.timer
    reasons: list[str] = []
    head_down = state_tracker.head_down(student_id, kpts, bh, signals.get("pitch"))
    # Nhịp tay đo trên điểm đã làm mượt. Cổ tay độ tin cậy thấp chỉ gia hạn khi đã bắt được.
    fresh_writing = bh.is_on("Viết bài") and state_tracker.writing(
        student_id, kpts, bh, head_down is True)
    soft_writing = bool(state_tracker.writing_soft.get(student_id))
    hands = _writing_hands(kpts, bh)
    raw_desk = desk_evidence(kpts, bh) is True if bh.is_on("Gục xuống bàn") else False
    latched = bool(state_tracker.writing_latched.get(student_id))
    # Đã bắt viết bài thì nhịp tay yếu hơn vẫn là đang viết, và xóa đồng hồ cúi đầu.
    caught = fresh_writing or (latched and soft_writing)
    if caught:
        state_tracker.writing_latched[student_id] = True
        state_tracker.writing_gap[student_id] = 0.0
        bow = state_tracker._timers[(student_id, "Cúi đầu lâu")]
        bow.on = 0.0
        bow.miss = 0.0
        latched = True
    elif latched and head_down is not True:
        state_tracker.pause_writing(student_id, dt, bh)
        latched = bool(state_tracker.writing_latched.get(student_id))
    elif latched:
        state_tracker.writing_gap[student_id] = 0.0
    # Chưa từng viết: cúi đầu vẫn tích như trước. Đã viết: chỉ tích khi nhìn thấy tay
    # không còn nhịp viết. Mất tay thì giữ nguyên đồng hồ, không coi là đã cúi đủ lâu.
    proving_bow = head_down is True and not caught and not raw_desk and (
        not latched or hands != "unseen"
    )
    if bh.is_on("Cúi đầu lâu") and not (latched and not proving_bow):
        held_bow = tol_timer(student_id, "Cúi đầu lâu", proving_bow, dt, bh)
    else:
        held_bow = state_tracker._timers[(student_id, "Cúi đầu lâu")].on if latched else 0.0
    writing = False
    if bh.is_on("Viết bài") and (caught or latched):
        if caught or held_bow < bh.head_down_duration:
            writing = True
        else:
            state_tracker._drop_writing(student_id, reset_bow=False)
            latched = False
    desk = raw_desk and not writing
    turn = state_tracker.turn_angle(
        student_id, head_yaw(kpts, signals, bh.kp_min_conf), bh, head_down is True)

    if writing:
        reasons.append("Viết bài")

    if bh.is_on("Cúi đầu lâu"):
        active = proving_bow and not writing
        _emit(reasons, active, held_bow, bh.head_down_duration, f"Cúi đầu lâu ({held_bow:.1f}s)")

    if bh.is_on("Gục xuống bàn"):
        held = tol_timer(student_id, "Gục xuống bàn", desk, dt, bh)
        _emit(reasons, desk, held, bh.desk_sleep_duration, f"Gục xuống bàn ({held:.1f}s)")

    if bh.is_on("Giơ tay"):
        raised = _hand_raised(kpts, bh.kp_min_conf)
        held = tol_timer(student_id, "Giơ tay", raised, dt, bh)
        _emit(reasons, raised, held, bh.hand_raise_duration, f"Giơ tay ({held:.1f}s)")

    if bh.is_on("Quay đầu"):
        turned = turn is not None and abs(turn) > bh.head_rotation_threshold
        if turned:
            state_tracker.turn_side[student_id] = "phải" if turn > 0 else "trái"
        held = tol_timer(student_id, "Quay đầu", turned, dt, bh)
        if turned and held >= bh.head_rotation_duration:
            side = state_tracker.turn_side.get(student_id, "")
            reasons.append(f"Quay đầu ({side} {held:.1f}s)")

    if bh.is_on("Nghiêng người"):
        tilt = _shoulder_tilt(kpts, bh.kp_min_conf)
        active = tilt is not None and tilt > bh.shoulder_tilt_threshold
        held = tol_timer(student_id, "Nghiêng người", active, dt, bh)
        if active and held >= bh.shoulder_tilt_duration:
            reasons.append(f"Nghiêng người ({int(tilt)}°)")

    return reasons
