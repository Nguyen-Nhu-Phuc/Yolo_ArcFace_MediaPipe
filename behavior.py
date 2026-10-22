"""Chấm điểm tập trung từ keypoint COCO-17 (YOLO-pose).

Mỗi keypoint là [x, y, confidence], x/y theo pixel. Mọi khoảng cách được chia cho
bề rộng vai để học sinh xa / gần camera dùng chung ngưỡng.

Chấm điểm chỉ phụ thuộc vào (keypoint, tín hiệu mặt, thời gian giữa 2 lần phân tích),
không cần ảnh → ghi các tín hiệu này ra file là chạy lại được y hệt (eval_behavior.py).
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from config import Config

# (tên đầy đủ trong lý do, nhãn viết tắt trên video / dashboard)
REASON_SHORT = (
    ("Cúi đầu lâu", "Cúi lâu"),
    ("Nhắm mắt", "Ngủ"),
    ("Ngủ gục", "Ngủ gục"),
    ("Thẫn thờ", "Thẫn"),
    ("Quay đầu", "Quay"),
    ("Cúi đầu thoáng qua", "Cúi"),
    ("Nghiêng người", "Nghiêng"),
    ("Tay bất thường", "Tay"),
    ("Nói chuyện", "Nói"),
    ("Viết bài", "Viết"),
    ("Giơ tay", "Giơ tay"),
)
POSITIVE_BEHAVIORS = {"Viết bài", "Giơ tay"}

KP_CONF = 0.5


def behavior_legend(config: Config) -> list[dict]:
    """Chú thích nhãn viết tắt kèm điều kiện + điểm trừ (lấy từ ngưỡng đang dùng)."""
    bh = config.behavior
    info = {
        "Cúi đầu lâu": (f"Cúi đầu liên tục ≥ {bh.head_down_duration:g} giây (so với tư thế ngồi "
                        "thẳng thường ngày của chính em đó)", bh.head_down_penalty),
        "Nhắm mắt": (f"Mắt nhắm liên tục ≥ {bh.eye_closed_duration:g} giây "
                     "(chỉ đo khi mặt đủ lớn và không cúi)", bh.eye_closed_penalty),
        "Ngủ gục": (f"Nằm nghiêng / nằm ngang lên bàn ≥ {bh.desk_sleep_duration:g} giây, hoặc úp mặt "
                    f"lên tay bất động ≥ {bh.desk_sleep_face_down_duration:g} giây "
                    "(thay cho cúi đầu lâu, thẫn thờ, nghiêng người)", bh.desk_sleep_penalty),
        "Thẫn thờ": (f"Đầu gần như bất động ≥ {bh.stillness_duration:g} giây",
                     bh.stillness_penalty),
        "Quay đầu": (f"Quay đầu sang ngang > {bh.head_rotation_threshold:g}° "
                     f"liên tục ≥ {bh.head_rotation_duration:g} giây", bh.head_rotation_penalty),
        "Cúi đầu thoáng qua": (f"Cúi đầu nhưng chưa đủ {bh.head_down_duration:g} giây",
                               bh.short_head_down_penalty),
        "Nghiêng người": (f"Vai lệch > {bh.shoulder_tilt_threshold:g}° "
                          f"liên tục ≥ {bh.shoulder_tilt_duration:g} giây", bh.shoulder_tilt_penalty),
        "Tay bất thường": (f"Cổ tay thấp hơn hông (tay để dưới gầm bàn) ≥ {bh.hand_low_duration:g} giây",
                           bh.hand_below_hip_penalty),
        "Nói chuyện": (f"Hai em ngồi cạnh quay mặt vào nhau ≥ {bh.chat_duration:g} giây "
                       "(thử nghiệm)", bh.chat_penalty),
        "Viết bài": ("Cúi đầu + cổ tay đặt trên bàn trước người — không trừ điểm cúi đầu/thẫn thờ", 0),
        "Giơ tay": ("Cổ tay cao hơn vai và mũi", 0),
    }
    return [
        {
            "short": short,
            "name": name,
            "positive": name in POSITIVE_BEHAVIORS,
            "condition": info[name][0],
            "penalty": int(info[name][1]),
            "enabled": bh.is_on(name),
        }
        for name, short in REASON_SHORT
    ]


# ---------------------------------------------------------------- đo trên 1 khung


def shoulder_width(keypoints) -> float | None:
    ls, rs = keypoints[5], keypoints[6]
    if ls[2] <= KP_CONF or rs[2] <= KP_CONF:
        return None
    sw = float(np.hypot(rs[0] - ls[0], rs[1] - ls[1]))
    return sw if sw > 1e-3 else None


def head_height(keypoints) -> float | None:
    """Độ cao mũi trên đường vai, theo bề rộng vai (ngồi thẳng ~0.5, cúi ≤ 0)."""
    nose = keypoints[0]
    sw = shoulder_width(keypoints)
    if sw is None or nose[2] <= KP_CONF:
        return None
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
    return float(shoulder_y - nose[1]) / sw


def is_head_down(keypoints, bh) -> bool:
    """Cúi đầu theo ngưỡng tuyệt đối (không cần lịch sử). False nếu keypoint không tin cậy."""
    h = head_height(keypoints)
    return h is not None and h < bh.head_down_ratio


def head_yaw(keypoints, signals: dict | None = None) -> tuple[float, str] | None:
    """(góc quay đầu theo độ, nguồn đo), dương = quay sang phải ảnh. Ưu tiên góc từ InsightFace.

    Không suy "quay hẳn" từ việc chỉ thấy 1 tai: HS ngồi xa thường bị tóc che tai.
    """
    if signals and signals.get("yaw") is not None:
        return float(signals["yaw"]), "face"
    nose, left_ear, right_ear = keypoints[0], keypoints[3], keypoints[4]
    if nose[2] <= KP_CONF or left_ear[2] <= KP_CONF or right_ear[2] <= KP_CONF:
        return None
    inter_ear = abs(right_ear[0] - left_ear[0])
    if inter_ear <= 1e-3:
        return None
    offset = nose[0] - (left_ear[0] + right_ear[0]) / 2
    return float(np.degrees(np.arcsin(max(-1.0, min(1.0, offset / inter_ear))))), "keypoint"


def _is_writing(keypoints, head_down: bool, bh) -> bool:
    """Cúi đầu + cổ tay đặt trên bàn trước người (đo theo bề rộng vai)."""
    sw = shoulder_width(keypoints)
    if not head_down or sw is None:
        return False
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    for wrist in (keypoints[9], keypoints[10]):
        if wrist[2] <= KP_CONF:
            continue
        drop = (wrist[1] - shoulder_y) / sw
        offset = abs(wrist[0] - center_x) / sw
        if (bh.writing_min_drop <= drop <= bh.writing_max_drop
                and offset <= bh.writing_max_offset):
            return True
    return False


def _hand_low(keypoints, bh) -> bool:
    """Cổ tay thấp hơn hông ≥ N bề rộng vai (hông bị bàn che thì pose đoán → đòi conf cao)."""
    sw = shoulder_width(keypoints)
    if sw is None:
        return False
    for wrist, hip in ((keypoints[9], keypoints[11]), (keypoints[10], keypoints[12])):
        if (wrist[2] > KP_CONF and hip[2] > bh.hand_hip_min_conf
                and (wrist[1] - hip[1]) / sw > bh.hand_below_hip_ratio):
            return True
    return False


def _shoulder_tilt(keypoints) -> float | None:
    if shoulder_width(keypoints) is None:
        return None
    ls, rs = keypoints[5], keypoints[6]
    return float(np.degrees(np.arctan2(abs(rs[1] - ls[1]), abs(rs[0] - ls[0]))))


def _hand_raised(keypoints) -> bool:
    nose = keypoints[0]
    if nose[2] <= KP_CONF:
        return False
    for hand, shoulder in ((keypoints[9], keypoints[5]), (keypoints[10], keypoints[6])):
        if (hand[2] > KP_CONF and shoulder[2] > KP_CONF
                and hand[1] < shoulder[1] and hand[1] < nose[1]):
            return True
    return False


def _head_point(keypoints, min_conf: float = 0.3):
    pts = [p for p in keypoints[:5] if p[2] > min_conf]
    if not pts:
        return None
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def desk_sleep_pose(keypoints, bh) -> str | None:
    """Tư thế nằm lên bàn (1 khung) → 'side' | 'flat' | 'face_down' | None.

    - side: nằm nghiêng — vai dựng đứng, đầu nằm bên cạnh vai (không cao hơn vai)
    - flat: thân nằm ngang — hông → vai gần phương ngang
    - face_down: úp mặt lên tay — đầu thấp hơn vai, cổ tay ngang tầm đầu
      (cúi viết sát bàn cũng ra dáng này → cần giữ lâu hơn mới tính ngủ)
    """
    ls, rs = keypoints[5], keypoints[6]
    sw = shoulder_width(keypoints)
    head = _head_point(keypoints)
    if sw is None or head is None:
        return None
    hx, hy = head
    sx, sy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
    head_rel = (hy - sy) / sw
    head_side = abs(hx - sx) / sw
    shoulder_angle = np.degrees(np.arctan2(abs(rs[1] - ls[1]), abs(rs[0] - ls[0]) + 1e-6))

    if (shoulder_angle >= bh.desk_sleep_shoulder_angle
            and head_side >= bh.desk_sleep_head_side and head_rel >= -0.2):
        return "side"

    hips = [p for p in (keypoints[11], keypoints[12]) if p[2] > 0.3]
    if hips:
        hipx = float(np.mean([p[0] for p in hips]))
        hipy = float(np.mean([p[1] for p in hips]))
        torso_angle = np.degrees(np.arctan2(abs(sx - hipx), abs(hipy - sy) + 1e-6))
        if torso_angle >= bh.desk_sleep_torso_angle and head_rel >= -0.2:
            return "flat"

    if head_rel >= bh.desk_sleep_head_drop:
        for w in (keypoints[9], keypoints[10]):
            if w[2] > 0.3 and (w[1] - hy) / sw <= bh.desk_sleep_wrist_dy and abs(w[0] - hx) / sw <= 1.5:
                return "face_down"
    return None


# ---------------------------------------------------------------- trạng thái qua nhiều khung


class Debounce:
    """Thời gian 1 điều kiện đúng liên tục; lần sai ngắn hơn `tolerance` giây không làm mất đợt."""

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
        return self.on


class StudentStateTracker:
    """Trạng thái hành vi theo từng học sinh qua nhiều khung."""

    def __init__(self):
        self._timers: dict[tuple, Debounce] = defaultdict(Debounce)
        self.clock: dict = defaultdict(float)  # giây đã phân tích theo từng HS
        self.head_track: dict = {}  # key → deque[(t, x, y, sw)]
        self.head_heights: dict = {}  # key → deque[head_height] (tư thế nền)
        self.score_ema: dict = {}
        self.turn_side: dict = {}
        self.yaw_history: dict = {}
        self.last_turn: dict = {}  # key → độ quay lần phân tích gần nhất (dùng cho nói chuyện)
        self.desk_sleep_timer = defaultdict(float)
        self.desk_sleep_miss = defaultdict(int)
        self.desk_sleep_head: dict = {}
        self.desk_sleep_strong: set = set()  # đợt hiện tại đã thấy nằm nghiêng / thân ngang

    def forget(self, student_id) -> None:
        for keyed in (self._timers, self.yaw_history):
            for k in [k for k in keyed if k[0] == student_id]:
                del keyed[k]
        for d in (self.clock, self.head_track, self.head_heights, self.score_ema, self.turn_side,
                  self.last_turn,
                  self.desk_sleep_timer, self.desk_sleep_miss, self.desk_sleep_head):
            d.pop(student_id, None)
        self.desk_sleep_strong.discard(student_id)

    def timer(self, student_id, name: str, active: bool, dt: float, bh) -> float:
        return self._timers[(student_id, name)].step(active, dt, bh.miss_tolerance_sec)

    def head_down(self, student_id, keypoints, bh, pitch: float | None) -> bool:
        """Cúi so với tư thế ngồi thẳng của chính HS (p75 độ cao đầu), chưa đủ lịch sử thì dùng ngưỡng chung."""
        h = head_height(keypoints)
        down = False
        if h is not None:
            hist = self.head_heights.get(student_id)
            if hist is None:
                hist = self.head_heights[student_id] = deque(maxlen=bh.head_baseline_window)
            if len(hist) >= bh.head_baseline_min_samples:
                down = h < float(np.percentile(hist, 75)) - bh.head_down_drop
            else:
                down = h < bh.head_down_ratio
            hist.append(h)
        if pitch is not None and pitch >= bh.head_down_pitch:
            down = True
        return down

    def turn_angle(self, student_id, measured: tuple[float, str] | None, bh) -> float | None:
        """Độ quay so với hướng nhìn thường ngày của HS (trung vị yaw, riêng từng nguồn đo).

        Camera đặt lệch → HS ngồi 2 bên nhìn bảng đã có yaw ±30° so với camera.
        """
        if measured is None:
            self.last_turn.pop(student_id, None)
            return None
        yaw, source = measured
        hist = self.yaw_history.get((student_id, source))
        if hist is None:
            hist = self.yaw_history[(student_id, source)] = deque(maxlen=bh.head_baseline_window)
        # Chưa biết hướng nhìn thường ngày → chưa kết luận (ưu tiên không báo nhầm)
        turn = yaw - float(np.median(hist)) if len(hist) >= bh.head_baseline_min_samples else None
        hist.append(yaw)
        if turn is None:
            self.last_turn.pop(student_id, None)
        else:
            self.last_turn[student_id] = turn
        return turn

    def head_speed(self, student_id, keypoints, dt: float, bh) -> float | None:
        """Tốc độ đầu trong ~stillness_window giây gần nhất, theo bề rộng vai / giây."""
        self.clock[student_id] += dt
        t = self.clock[student_id]
        head = _head_point(keypoints)
        sw = shoulder_width(keypoints)
        track = self.head_track.get(student_id)
        if track is None:
            track = self.head_track[student_id] = deque(maxlen=64)
        if head is not None and sw is not None:
            track.append((t, head[0], head[1], sw))
        while len(track) > 3 and t - track[1][0] >= bh.stillness_window:
            track.popleft()
        if len(track) < 3:
            return None
        elapsed = track[-1][0] - track[0][0]
        if elapsed <= 0:
            return None
        sw_mean = float(np.mean([p[3] for p in track]))
        path = sum(float(np.hypot(b[1] - a[1], b[2] - a[2])) for a, b in zip(track, list(track)[1:]))
        return path / sw_mean / elapsed

    def update_desk_sleep(self, student_id, keypoints, fps: float, bh) -> float | None:
        """Số giây đã nằm lên bàn liên tục (≥ ngưỡng thì trả về, chưa đủ trả None).

        Bỏ qua 1 lần lệch tư thế (pose chập chờn); đầu dịch mạnh → không phải ngủ, đếm lại.
        """
        head = _head_point(keypoints)
        sw = shoulder_width(keypoints) or 0.0
        prev = self.desk_sleep_head.get(student_id)
        moved = (head is not None and prev is not None and sw > 1e-3
                 and np.hypot(head[0] - prev[0], head[1] - prev[1]) / sw > bh.desk_sleep_max_move)
        if head is not None:
            self.desk_sleep_head[student_id] = head

        kind = desk_sleep_pose(keypoints, bh)
        if kind is not None and not moved:
            self.desk_sleep_timer[student_id] += 1 / fps
            self.desk_sleep_miss[student_id] = 0
            if kind != "face_down":
                self.desk_sleep_strong.add(student_id)
        else:
            self.desk_sleep_miss[student_id] += 1
            if moved or self.desk_sleep_miss[student_id] > 1:
                self.desk_sleep_timer[student_id] = 0
                self.desk_sleep_strong.discard(student_id)
        t = self.desk_sleep_timer[student_id]
        need = (bh.desk_sleep_duration if student_id in self.desk_sleep_strong
                else bh.desk_sleep_face_down_duration)
        return t if t >= need else None

    def smooth(self, student_id, score: float, bh) -> int:
        prev = self.score_ema.get(student_id)
        s = score if prev is None else bh.score_smoothing * prev + (1 - bh.score_smoothing) * score
        self.score_ema[student_id] = s
        return int(round(s))


def calculate_attention_score(keypoints, config: Config,
                              state_tracker: StudentStateTracker,
                              student_id, fps: float,
                              signals: dict | None = None) -> tuple[int, list[str]]:
    """Trả về (điểm 0–100, danh sách lý do).

    signals: tín hiệu mặt (có thể thiếu) — ear, mar (MediaPipe), yaw, pitch (InsightFace, độ;
    yaw dương = quay sang phải ảnh, pitch dương = cúi xuống).
    """
    if keypoints is None or len(keypoints) < 17:
        return 0, []

    bh = config.behavior
    signals = signals or {}
    dt = 1.0 / fps
    tol_timer = state_tracker.timer
    score = 100
    reasons: list[str] = []

    desk_sleep = (state_tracker.update_desk_sleep(student_id, keypoints, fps, bh)
                  if bh.is_on("Ngủ gục") else None)
    head_down = state_tracker.head_down(student_id, keypoints, bh, signals.get("pitch"))
    writing = bh.is_on("Viết bài") and _is_writing(keypoints, head_down, bh)
    speed = state_tracker.head_speed(student_id, keypoints, dt, bh)
    turn = state_tracker.turn_angle(student_id, head_yaw(keypoints, signals), bh)

    long_down = 0.0
    if bh.is_on("Cúi đầu lâu"):
        long_down = tol_timer(student_id, "Cúi đầu lâu", head_down and not writing, dt, bh)
        # Dung sai giữ đợt khi pose chập chờn, nhưng đang viết bài thì bỏ nhãn ngay
        if long_down >= bh.head_down_duration and not writing:
            score -= bh.head_down_penalty
            reasons.append(f"Cúi đầu lâu ({long_down:.1f}s)")

    if bh.is_on("Nhắm mắt"):
        ear = signals.get("ear")
        t = tol_timer(student_id, "Nhắm mắt", ear is not None and ear < bh.eye_closed_threshold, dt, bh)
        if t >= bh.eye_closed_duration:
            score -= bh.eye_closed_penalty
            reasons.append(f"Nhắm mắt ({t:.1f}s)")

    if bh.is_on("Thẫn thờ"):
        still = speed is not None and speed < bh.stillness_speed and not writing
        t = tol_timer(student_id, "Thẫn thờ", still, dt, bh)
        if t >= bh.stillness_duration and not writing:
            score -= bh.stillness_penalty
            reasons.append(f"Thẫn thờ ({t:.1f}s)")

    if bh.is_on("Quay đầu"):
        turned = turn is not None and abs(turn) > bh.head_rotation_threshold
        if turned:
            state_tracker.turn_side[student_id] = "phải" if turn > 0 else "trái"
        t = tol_timer(student_id, "Quay đầu", turned, dt, bh)
        if t >= bh.head_rotation_duration:
            side = state_tracker.turn_side.get(student_id, "")
            score -= bh.head_rotation_penalty
            reasons.append(f"Quay đầu ({side} {t:.1f}s)")

    if writing:
        reasons.append("Viết bài")

    if (bh.is_on("Cúi đầu thoáng qua") and head_down and not writing
            and long_down < bh.head_down_duration):
        score -= bh.short_head_down_penalty
        reasons.append("Cúi đầu thoáng qua")

    if bh.is_on("Nghiêng người"):
        tilt = _shoulder_tilt(keypoints)
        t = tol_timer(student_id, "Nghiêng người",
                      tilt is not None and tilt > bh.shoulder_tilt_threshold, dt, bh)
        if t >= bh.shoulder_tilt_duration:
            score -= bh.shoulder_tilt_penalty
            reasons.append(f"Nghiêng người ({int(tilt or bh.shoulder_tilt_threshold)}°)")

    if bh.is_on("Tay bất thường"):
        t = tol_timer(student_id, "Tay bất thường",
                      _hand_low(keypoints, bh) and not writing, dt, bh)
        if t >= bh.hand_low_duration and not writing:
            score -= bh.hand_below_hip_penalty
            reasons.append("Tay bất thường")

    if bh.is_on("Giơ tay") and _hand_raised(keypoints):
        reasons.append("Giơ tay")

    if desk_sleep is not None:
        # Cùng 1 tư thế nằm bàn sinh ra cúi đầu / thẫn thờ / nghiêng / quay → chỉ tính ngủ gục
        score, reasons = 100 - bh.desk_sleep_penalty, [f"Ngủ gục ({desk_sleep:.1f}s)"]

    return max(0, min(100, state_tracker.smooth(student_id, max(0, score), bh))), reasons
