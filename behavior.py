"""Chấm điểm tập trung từ keypoint COCO-17 (YOLO-pose).

Mỗi keypoint là [x, y, confidence], x/y theo pixel.
Luật và ngưỡng giống classroom_attention.
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


def behavior_legend(config: Config) -> list[dict]:
    """Chú thích nhãn viết tắt kèm điều kiện + điểm trừ (lấy từ ngưỡng đang dùng)."""
    bh = config.behavior
    info = {
        "Cúi đầu lâu": (f"Cúi đầu liên tục ≥ {bh.head_down_duration:g} giây", bh.head_down_penalty),
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
        "Nghiêng người": (f"Vai lệch > {bh.shoulder_tilt_threshold}°", bh.shoulder_tilt_penalty),
        "Tay bất thường": ("Cổ tay thấp hơn hông (tay để dưới gầm bàn)",
                           bh.hand_below_hip_penalty),
        "Nói chuyện": ("Hai em ngồi cạnh quay vào nhau (thử nghiệm, dễ báo nhầm)", bh.chat_penalty),
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


def is_head_down(keypoints, bbox_height: float, bh) -> bool:
    """Mũi hạ ngang (hoặc thấp hơn) vai. False nếu keypoint không tin cậy."""
    nose, left_shoulder, right_shoulder = keypoints[0], keypoints[5], keypoints[6]
    if nose[2] <= 0.5 or left_shoulder[2] <= 0.5 or right_shoulder[2] <= 0.5:
        return False
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
    return nose[1] - shoulder_center_y > -bh.head_down_threshold * bbox_height


def _is_writing(keypoints, bbox_height: float, bh) -> bool:
    """Cúi đầu + cổ tay đặt trên bàn trước người (đo theo bề rộng vai)."""
    if not is_head_down(keypoints, bbox_height, bh):
        return False
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    shoulder_width = float(np.hypot(right_shoulder[0] - left_shoulder[0],
                                    right_shoulder[1] - left_shoulder[1]))
    if shoulder_width < 1e-3:
        return False
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    for wrist in (keypoints[9], keypoints[10]):
        if wrist[2] <= 0.5:
            continue
        drop = (wrist[1] - shoulder_y) / shoulder_width
        offset = abs(wrist[0] - center_x) / shoulder_width
        if (bh.writing_min_drop <= drop <= bh.writing_max_drop
                and offset <= bh.writing_max_offset):
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
    if ls[2] <= 0.5 or rs[2] <= 0.5:
        return None
    sw = float(np.hypot(rs[0] - ls[0], rs[1] - ls[1]))
    head = _head_point(keypoints)
    if sw < 1e-3 or head is None:
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


class StudentStateTracker:
    """Trạng thái hành vi theo từng học sinh qua nhiều khung."""

    def __init__(self, eye_detector=None):
        self.head_position = defaultdict(lambda: deque(maxlen=30))
        self.eye_openness = defaultdict(lambda: deque(maxlen=30))
        self.gaze_position = defaultdict(lambda: deque(maxlen=30))
        self.head_down_timer = defaultdict(float)
        self.eye_closed_timer = defaultdict(float)
        self.stillness_timer = defaultdict(float)
        self.head_rotation_timer = defaultdict(float)
        self.desk_sleep_timer = defaultdict(float)
        self.desk_sleep_miss = defaultdict(int)
        self.desk_sleep_head: dict = {}
        self.desk_sleep_strong: set = set()  # đợt hiện tại đã thấy nằm nghiêng / thân ngang
        self.eye_detector = eye_detector

    def update(self, student_id, keypoints, fps: float, face_crop=None) -> None:
        if keypoints is None or len(keypoints) == 0:
            return
        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        if nose[2] > 0.5:
            self.head_position[student_id].append(nose[:2])
        ear = self._get_ear(left_eye, right_eye, face_crop)
        self.eye_openness[student_id].append(ear)
        if left_eye[2] > 0.5 and right_eye[2] > 0.5:
            gaze_x = (left_eye[0] + right_eye[0]) / 2
            gaze_y = (left_eye[1] + right_eye[1]) / 2
            self.gaze_position[student_id].append((gaze_x, gaze_y))

    def _get_ear(self, left_eye, right_eye, face_crop=None) -> float:
        """EAR từ Face Landmarker nếu có, không thì dùng độ tin cậy keypoint mắt."""
        if self.eye_detector is not None and face_crop is not None:
            try:
                ear = self.eye_detector.get_ear(face_crop)
                if ear is not None:
                    return ear
            except Exception:
                pass
        return self._calculate_ear(left_eye, right_eye)

    @staticmethod
    def _calculate_ear(left_eye, right_eye) -> float:
        try:
            if left_eye[2] > 0.5 and right_eye[2] > 0.5:
                return (left_eye[2] + right_eye[2]) / 2
            return 1.0
        except Exception:
            return 1.0

    def forget(self, student_id) -> None:
        for d in (self.head_position, self.eye_openness, self.gaze_position,
                  self.head_down_timer, self.eye_closed_timer,
                  self.stillness_timer, self.head_rotation_timer,
                  self.desk_sleep_timer, self.desk_sleep_miss, self.desk_sleep_head):
            d.pop(student_id, None)
        self.desk_sleep_strong.discard(student_id)

    def update_desk_sleep(self, student_id, keypoints, fps: float, bh) -> float | None:
        """Số giây đã nằm lên bàn liên tục (≥ ngưỡng thì trả về, chưa đủ trả None).

        Bỏ qua 1 lần lệch tư thế (pose chập chờn); đầu dịch mạnh → không phải ngủ, đếm lại.
        """
        head = _head_point(keypoints)
        ls, rs = keypoints[5], keypoints[6]
        sw = float(np.hypot(rs[0] - ls[0], rs[1] - ls[1])) if min(ls[2], rs[2]) > 0.5 else 0.0
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

    def check_long_term_behaviors(self, student_id, keypoints, fps: float,
                                  config: Config, bbox_height: float = 0,
                                  writing: bool = False) -> list:
        """Hành vi kéo dài (cúi đầu, nhắm mắt, thẫn thờ, quay đầu).

        Khi đang viết bài thì không cộng thời gian cúi đầu / thẫn thờ.
        """
        behaviors = []
        bh = config.behavior

        if keypoints is not None and len(keypoints) > 0 and bh.is_on("Cúi đầu lâu"):
            if not writing and is_head_down(keypoints, bbox_height, bh):
                self.head_down_timer[student_id] += 1 / fps
                if self.head_down_timer[student_id] >= bh.head_down_duration:
                    behaviors.append(
                        f"Cúi đầu lâu ({self.head_down_timer[student_id]:.1f}s)")
            else:
                self.head_down_timer[student_id] = 0
        else:
            self.head_down_timer[student_id] = 0

        if len(self.eye_openness[student_id]) > 0 and bh.is_on("Nhắm mắt"):
            ear = self.eye_openness[student_id][-1]
            if ear < bh.eye_closed_threshold:
                self.eye_closed_timer[student_id] += 1 / fps
                if self.eye_closed_timer[student_id] >= bh.eye_closed_duration:
                    behaviors.append(
                        f"Nhắm mắt ({self.eye_closed_timer[student_id]:.1f}s)")
            else:
                self.eye_closed_timer[student_id] = 0
        else:
            self.eye_closed_timer[student_id] = 0

        if len(self.head_position[student_id]) >= 5 and bh.is_on("Thẫn thờ"):
            recent = list(self.head_position[student_id])[-5:]
            head_movement = 0.0
            for i in range(1, len(recent)):
                dx = recent[i][0] - recent[i - 1][0]
                dy = recent[i][1] - recent[i - 1][1]
                head_movement += float(np.sqrt(dx * dx + dy * dy))

            if not writing and head_movement < bh.stillness_threshold:
                self.stillness_timer[student_id] += 1 / fps
                if self.stillness_timer[student_id] >= bh.stillness_duration:
                    behaviors.append(
                        f"Thẫn thờ ({self.stillness_timer[student_id]:.1f}s)")
            else:
                self.stillness_timer[student_id] = 0
        else:
            self.stillness_timer[student_id] = 0

        if keypoints is not None and len(keypoints) > 0 and bh.is_on("Quay đầu"):
            nose = keypoints[0]
            left_ear = keypoints[3]
            right_ear = keypoints[4]
            if nose[2] > 0.5 and left_ear[2] > 0.5 and right_ear[2] > 0.5:
                ear_mid_x = (left_ear[0] + right_ear[0]) / 2
                inter_ear = abs(right_ear[0] - left_ear[0])
                if inter_ear > 1e-3:
                    nose_offset = nose[0] - ear_mid_x
                    yaw_ratio = max(-1.0, min(1.0, nose_offset / inter_ear))
                    yaw_angle = abs(np.degrees(np.arcsin(yaw_ratio)))
                    if yaw_angle > bh.head_rotation_threshold:
                        self.head_rotation_timer[student_id] += 1 / fps
                        if self.head_rotation_timer[student_id] >= bh.head_rotation_duration:
                            direction = "phải" if nose_offset > 0 else "trái"
                            behaviors.append(
                                f"Quay đầu ({direction} "
                                f"{self.head_rotation_timer[student_id]:.1f}s)")
                    else:
                        self.head_rotation_timer[student_id] = 0
                else:
                    self.head_rotation_timer[student_id] = 0
            else:
                self.head_rotation_timer[student_id] = 0
        else:
            self.head_rotation_timer[student_id] = 0

        return behaviors


def calculate_attention_score(keypoints, bbox_height: float, config: Config,
                              state_tracker: StudentStateTracker,
                              student_id, fps: float,
                              face_crop=None) -> tuple[int, list[str]]:
    """Trả về (điểm 0–100, danh sách lý do)."""
    if keypoints is None or len(keypoints) < 17:
        return 0, []

    score = 100
    reasons: list[str] = []
    bh = config.behavior

    try:
        state_tracker.update(student_id, keypoints, fps, face_crop)
        desk_sleep = (state_tracker.update_desk_sleep(student_id, keypoints, fps, bh)
                      if bh.is_on("Ngủ gục") else None)
        writing = bh.is_on("Viết bài") and _is_writing(keypoints, bbox_height, bh)
        long_term = state_tracker.check_long_term_behaviors(
            student_id, keypoints, fps, config, bbox_height, writing=writing)

        for behavior in long_term:
            if "Cúi đầu lâu" in behavior:
                score -= bh.head_down_penalty
            elif "Nhắm mắt" in behavior:
                score -= bh.eye_closed_penalty
            elif "Thẫn thờ" in behavior:
                score -= bh.stillness_penalty
            elif "Quay đầu" in behavior:
                score -= bh.head_rotation_penalty
            reasons.append(behavior)

        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hand = keypoints[9]
        right_hand = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        if writing:
            reasons.append("Viết bài")

        if (bh.is_on("Cúi đầu thoáng qua")
                and not writing
                and not any("Cúi đầu lâu" in r for r in reasons)
                and is_head_down(keypoints, bbox_height, bh)):
            score -= bh.short_head_down_penalty
            reasons.append("Cúi đầu thoáng qua")

        if bh.is_on("Nghiêng người") and left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]
            angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
            if angle > bh.shoulder_tilt_threshold:
                score -= bh.shoulder_tilt_penalty
                reasons.append(f"Nghiêng người ({int(angle)}°)")

        hand_low = False
        if left_hand[2] > 0.5 and left_hip[2] > 0.5:
            if left_hand[1] > left_hip[1] + bh.hand_below_hip_threshold:
                hand_low = True
        if right_hand[2] > 0.5 and right_hip[2] > 0.5:
            if right_hand[1] > right_hip[1] + bh.hand_below_hip_threshold:
                hand_low = True
        if hand_low and not writing and bh.is_on("Tay bất thường"):
            score -= bh.hand_below_hip_penalty
            reasons.append("Tay bất thường")

        if bh.is_on("Giơ tay"):
            left_raised = (left_hand[2] > 0.5 and left_shoulder[2] > 0.5
                           and nose[2] > 0.5
                           and left_hand[1] < left_shoulder[1]
                           and left_hand[1] < nose[1])
            right_raised = (right_hand[2] > 0.5 and right_shoulder[2] > 0.5
                            and nose[2] > 0.5
                            and right_hand[1] < right_shoulder[1]
                            and right_hand[1] < nose[1])
            if left_raised or right_raised:
                reasons.append("Giơ tay")

        if desk_sleep is not None:
            # Cùng 1 tư thế nằm bàn sinh ra cúi đầu / thẫn thờ / nghiêng / quay → chỉ tính ngủ gục
            return max(0, 100 - bh.desk_sleep_penalty), [f"Ngủ gục ({desk_sleep:.1f}s)"]

    except Exception:
        return max(0, score), reasons

    return max(0, min(100, score)), reasons
