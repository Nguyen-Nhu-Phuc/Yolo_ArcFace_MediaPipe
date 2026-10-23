"""
Nhận diện hành vi lớp học bằng MediaPipe (Pose + Face Landmarker) — rule-based.

Luồng riêng cho webcam test. Không dùng YOLO.
Hành vi:
  - hand_raise: giơ tay phát biểu
  - turn_away: quay đi / quay ra sau quá lâu (>= TURN_AWAY_SEC)
  - drowsy: ngủ gật (cúi đầu + mắt hẹp hơn baseline + ít chuyển động)
  - writing: cúi xuống nhưng tay còn động → không coi là ngủ
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "mediapipe"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"

# Thời gian confirmed
TURN_AWAY_SEC = 10.0
DROWSY_SEC = 2.5
HAND_RAISE_SEC = 0.8
# Cho phép tín hiệu ngắt quãng ngắn (mất face khi quay) mà không reset đếm
TURN_GRACE_SEC = 2.5
DROWSY_GRACE_SEC = 1.2

WRITING_MOTION_EPS = 0.015

# Ngưỡng góc / EAR (nới hơn bản đầu — phù hợp webcam gần)
YAW_TURN_DEG = 28.0
NOSE_OFF_TURN = 0.32
PITCH_DOWN_DELTA = 12.0  # lệch pitch so với baseline khi cúi
EAR_ABS_DROWSY = 0.28  # mắt khép (EAR mở thường ~0.45–0.55)
EAR_RATIO_DROWSY = 0.62  # ear < baseline * ratio
POSE_HEAD_DOWN = 0.07  # mũi thấp hơn vai (chuẩn hóa)

# Pose landmark indices
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ear(landmarks: list, idxs: tuple[int, ...]) -> float:
    pts = [np.array([landmarks[i].x, landmarks[i].y], dtype=np.float32) for i in idxs]
    vert = _dist(pts[1], pts[5]) + _dist(pts[2], pts[4])
    horiz = _dist(pts[0], pts[3]) + 1e-6
    return vert / (2.0 * horiz)


def _lm_xy(lm, w: int, h: int) -> np.ndarray:
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def estimate_head_pose_from_face(landmarks, w: int, h: int) -> tuple[float, float]:
    try:
        nose = _lm_xy(landmarks[1], w, h)
        chin = _lm_xy(landmarks[199], w, h)
        le = _lm_xy(landmarks[33], w, h)
        re = _lm_xy(landmarks[263], w, h)
        ml = _lm_xy(landmarks[61], w, h)
        mr = _lm_xy(landmarks[291], w, h)
    except Exception:
        return 0.0, 0.0

    model = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0],
            [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1],
            [28.9, -28.9, -24.1],
        ],
        dtype=np.float64,
    )
    image_pts = np.array([nose, chin, le, re, ml, mr], dtype=np.float64)
    focal = float(w)
    center = (w / 2.0, h / 2.0)
    cam = np.array(
        [[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64
    )
    ok, rvec, _tvec = cv2.solvePnP(
        model, image_pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    if sy < 1e-6:
        pitch = math.degrees(math.atan2(-rot[1, 2], rot[1, 1]))
        yaw = 0.0
    else:
        pitch = math.degrees(math.atan2(-rot[2, 1], rot[2, 2]))
        yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    while pitch > 90.0:
        pitch -= 180.0
    while pitch < -90.0:
        pitch += 180.0
    while yaw > 90.0:
        yaw -= 180.0
    while yaw < -90.0:
        yaw += 180.0
    return float(yaw), float(pitch)


class BehaviorMonitor:
    """Stateful tracker cho 1 session webcam."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pose = None
        self._face = None
        self._init_models()
        self.reset()

    def _init_models(self) -> None:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not POSE_MODEL.exists() or not FACE_MODEL.exists():
            raise FileNotFoundError(
                f"Thiếu model MediaPipe trong {MODEL_DIR}. "
                "Cần pose_landmarker_lite.task và face_landmarker.task"
            )
        base_pose = python.BaseOptions(model_asset_path=str(POSE_MODEL))
        base_face = python.BaseOptions(model_asset_path=str(FACE_MODEL))
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=base_pose,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        face_opts = vision.FaceLandmarkerOptions(
            base_options=base_face,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_facial_transformation_matrixes=False,
        )
        self._pose = vision.PoseLandmarker.create_from_options(pose_opts)
        self._face = vision.FaceLandmarker.create_from_options(face_opts)

    def reset(self) -> None:
        with self._lock:
            self._turn_since: float | None = None
            self._turn_last_hit: float | None = None
            self._drowsy_since: float | None = None
            self._drowsy_last_hit: float | None = None
            self._raise_since: float | None = None
            self._last_wrists: tuple[float, float] | None = None
            self._wrist_motion = 0.0
            self._last_nose: tuple[float, float] | None = None
            self._head_motion = 0.0
            self._ear_baseline = 0.45
            self._pitch_baseline: float | None = None
            self._saw_face = False
            self._face_lost_since: float | None = None
            self._last_behaviors: list[str] = []
            self._last_signals: dict = {}

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        import mediapipe as mp

        if frame_bgr is None or frame_bgr.size == 0:
            return {"ok": False, "error": "empty frame"}

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_res = self._pose.detect(mp_img)
        face_res = self._face.detect(mp_img)

        signals = {
            "pose": False,
            "face": False,
            "yaw": 0.0,
            "pitch": 0.0,
            "ear": 1.0,
            "ear_baseline": 0.0,
            "nose_off": 0.0,
            "hand_raise_raw": False,
            "turn_raw": False,
            "head_down_raw": False,
            "eyes_drowsy_raw": False,
            "wrist_motion": 0.0,
            "head_motion": 0.0,
            "writing_like": False,
            "face_lost": False,
        }
        pose_pts = None
        now = time.monotonic()

        if pose_res.pose_landmarks:
            signals["pose"] = True
            lms = pose_res.pose_landmarks[0]
            pose_pts = lms

            def vis(i: int) -> bool:
                return float(getattr(lms[i], "visibility", 1.0) or 0.0) > 0.35

            raise_l = raise_r = False
            if vis(L_WRIST) and vis(L_SHOULDER):
                raise_l = lms[L_WRIST].y < lms[L_SHOULDER].y - 0.08
            if vis(R_WRIST) and vis(R_SHOULDER):
                raise_r = lms[R_WRIST].y < lms[R_SHOULDER].y - 0.08
            if raise_l and vis(L_ELBOW):
                raise_l = lms[L_ELBOW].y < lms[L_SHOULDER].y + 0.05
            if raise_r and vis(R_ELBOW):
                raise_r = lms[R_ELBOW].y < lms[R_SHOULDER].y + 0.05
            signals["hand_raise_raw"] = bool(raise_l or raise_r)

            if vis(NOSE) and vis(L_SHOULDER) and vis(R_SHOULDER):
                mid_x = 0.5 * (lms[L_SHOULDER].x + lms[R_SHOULDER].x)
                mid_y = 0.5 * (lms[L_SHOULDER].y + lms[R_SHOULDER].y)
                shoulder_w = abs(lms[L_SHOULDER].x - lms[R_SHOULDER].x) + 1e-6
                nose_off = (lms[NOSE].x - mid_x) / shoulder_w
                signals["nose_off"] = round(float(nose_off), 3)
                if abs(nose_off) >= NOSE_OFF_TURN:
                    signals["turn_raw"] = True
                # Cúi đầu: mũi thấp hơn vai
                if (lms[NOSE].y - mid_y) >= POSE_HEAD_DOWN:
                    signals["head_down_raw"] = True

            # Asymmetry tai (profile): một tai thấy rõ / lệch mạnh
            if vis(L_EAR) and vis(R_EAR) and vis(NOSE):
                ear_span = abs(lms[L_EAR].x - lms[R_EAR].x) + 1e-6
                # Khi quay nghiêng, khoảng cách 2 tai trên ảnh thu hẹp mạnh
                if ear_span < 0.045:
                    signals["turn_raw"] = True

            # Wrist + head motion
            wx = wy = 0.0
            n = 0
            if vis(L_WRIST):
                wx += lms[L_WRIST].x
                wy += lms[L_WRIST].y
                n += 1
            if vis(R_WRIST):
                wx += lms[R_WRIST].x
                wy += lms[R_WRIST].y
                n += 1
            with self._lock:
                if n:
                    wx /= n
                    wy /= n
                    if self._last_wrists is not None:
                        motion = math.hypot(
                            wx - self._last_wrists[0], wy - self._last_wrists[1]
                        )
                        self._wrist_motion = 0.65 * self._wrist_motion + 0.35 * motion
                    self._last_wrists = (wx, wy)
                    signals["wrist_motion"] = round(self._wrist_motion, 4)
                    if wy > 0.52 and self._wrist_motion > WRITING_MOTION_EPS:
                        signals["writing_like"] = True

                if vis(NOSE):
                    nx, ny = float(lms[NOSE].x), float(lms[NOSE].y)
                    if self._last_nose is not None:
                        hm = math.hypot(nx - self._last_nose[0], ny - self._last_nose[1])
                        self._head_motion = 0.7 * self._head_motion + 0.3 * hm
                    self._last_nose = (nx, ny)
                    signals["head_motion"] = round(self._head_motion, 4)

        face_ok = bool(face_res.face_landmarks)
        if face_ok:
            signals["face"] = True
            fl = face_res.face_landmarks[0]
            try:
                ear = 0.5 * (_ear(fl, LEFT_EYE) + _ear(fl, RIGHT_EYE))
            except Exception:
                ear = 1.0
            signals["ear"] = round(float(ear), 4)
            yaw, pitch = estimate_head_pose_from_face(fl, w, h)
            signals["yaw"] = round(yaw, 1)
            signals["pitch"] = round(pitch, 1)

            with self._lock:
                self._saw_face = True
                self._face_lost_since = None
                # Baseline khi nhìn gần thẳng (không quay / không cúi mạnh)
                facing = abs(yaw) < 18.0 and not signals["head_down_raw"]
                if facing and ear > 0.22:
                    self._ear_baseline = max(self._ear_baseline * 0.92, ear)
                    self._ear_baseline = min(max(self._ear_baseline, 0.28), 0.65)
                    if self._pitch_baseline is None:
                        self._pitch_baseline = pitch
                    else:
                        self._pitch_baseline = 0.9 * self._pitch_baseline + 0.1 * pitch
                signals["ear_baseline"] = round(self._ear_baseline, 3)

                if abs(yaw) >= YAW_TURN_DEG:
                    signals["turn_raw"] = True

                # Cúi đầu so với baseline pitch (webcam thường pitch âm hơn khi cúi)
                if self._pitch_baseline is not None:
                    # nhìn xuống: pitch giảm (âm hơn) hoặc tăng — lấy độ lệch tuyệt đối theo hướng "xuống"
                    delta_down = self._pitch_baseline - pitch  # >0 nếu pitch âm hơn baseline
                    signals["pitch_delta"] = round(delta_down, 1)
                    if delta_down >= PITCH_DOWN_DELTA or (-delta_down) >= (
                        PITCH_DOWN_DELTA + 8
                    ):
                        # lệch lớn theo 2 chiều đều có thể là cúi (tùy giải PnP)
                        if abs(delta_down) >= PITCH_DOWN_DELTA:
                            signals["head_down_raw"] = True
                elif pitch <= -28.0:
                    signals["head_down_raw"] = True

                # Mắt ngủ: EAR tuyệt đối thấp HOẶC thấp so với baseline
                eyes_drowsy = ear <= EAR_ABS_DROWSY or ear <= (
                    self._ear_baseline * EAR_RATIO_DROWSY
                )
                signals["eyes_drowsy_raw"] = bool(eyes_drowsy)
        else:
            with self._lock:
                if self._saw_face:
                    if self._face_lost_since is None:
                        self._face_lost_since = now
                    # Mất mặt >0.6s trong khi vẫn còn pose → thường là quay đi / cúi sâu
                    lost_for = now - self._face_lost_since
                    signals["face_lost"] = True
                    signals["face_lost_sec"] = round(lost_for, 2)
                    if lost_for >= 0.6 and signals["pose"]:
                        signals["turn_raw"] = True
                signals["ear_baseline"] = round(self._ear_baseline, 3)

        behaviors: list[str] = []
        timers: dict[str, float] = {}

        with self._lock:
            # --- Hand raise ---
            if signals["hand_raise_raw"]:
                if self._raise_since is None:
                    self._raise_since = now
                timers["hand_raise"] = round(now - self._raise_since, 2)
                if now - self._raise_since >= HAND_RAISE_SEC:
                    behaviors.append("hand_raise")
            else:
                self._raise_since = None

            # --- Turn away (grace: không reset ngay khi tín hiệu chớp tắt) ---
            turn_hit = signals["turn_raw"] and not signals["hand_raise_raw"]
            if turn_hit:
                if self._turn_since is None:
                    self._turn_since = now
                self._turn_last_hit = now
            if self._turn_since is not None:
                last = self._turn_last_hit or self._turn_since
                if (now - last) <= TURN_GRACE_SEC:
                    timers["turn_away"] = round(now - self._turn_since, 2)
                    if now - self._turn_since >= TURN_AWAY_SEC:
                        behaviors.append("turn_away")
                else:
                    self._turn_since = None
                    self._turn_last_hit = None

            # --- Drowsy vs writing ---
            still = signals["wrist_motion"] < WRITING_MOTION_EPS * 1.8
            head_still = signals.get("head_motion", 0.0) < 0.012
            drowsy_hit = (
                signals["head_down_raw"]
                and signals.get("eyes_drowsy_raw", False)
                and not signals["hand_raise_raw"]
                and not signals["writing_like"]
                and still
            )
            # Cúi sâu + đứng yên lâu (mặt có thể mất) cũng coi là ứng viên ngủ
            if (
                not drowsy_hit
                and signals["head_down_raw"]
                and still
                and head_still
                and not signals["writing_like"]
                and not signals["hand_raise_raw"]
                and (signals.get("eyes_drowsy_raw") or signals.get("face_lost"))
            ):
                drowsy_hit = True

            if (
                signals["head_down_raw"]
                and signals["writing_like"]
                and not signals.get("eyes_drowsy_raw", False)
            ):
                behaviors.append("writing")

            if drowsy_hit:
                if self._drowsy_since is None:
                    self._drowsy_since = now
                self._drowsy_last_hit = now
            if self._drowsy_since is not None:
                last_d = self._drowsy_last_hit or self._drowsy_since
                if (now - last_d) <= DROWSY_GRACE_SEC:
                    timers["drowsy"] = round(now - self._drowsy_since, 2)
                    if now - self._drowsy_since >= DROWSY_SEC:
                        behaviors.append("drowsy")
                else:
                    self._drowsy_since = None
                    self._drowsy_last_hit = None

            self._last_behaviors = behaviors
            self._last_signals = signals

        skeleton = []
        if pose_pts is not None:
            for i in (
                NOSE,
                L_SHOULDER,
                R_SHOULDER,
                L_ELBOW,
                R_ELBOW,
                L_WRIST,
                R_WRIST,
                L_EAR,
                R_EAR,
            ):
                lm = pose_pts[i]
                skeleton.append(
                    {
                        "i": i,
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "v": float(getattr(lm, "visibility", 1.0) or 0.0),
                    }
                )

        return {
            "ok": True,
            "behaviors": behaviors,
            "timers": timers,
            "signals": signals,
            "skeleton": skeleton,
            "labels": {
                "hand_raise": "Giơ tay phát biểu",
                "turn_away": f"Quay đi / nói chuyện (>={int(TURN_AWAY_SEC)}s)",
                "drowsy": "Ngủ gật",
                "writing": "Cúi viết bài (không tính ngủ)",
            },
        }


_monitors: dict[str, BehaviorMonitor] = {}
_monitors_lock = threading.Lock()


def get_monitor(session_id: str) -> BehaviorMonitor:
    sid = (session_id or "default").strip() or "default"
    with _monitors_lock:
        mon = _monitors.get(sid)
        if mon is None:
            mon = BehaviorMonitor()
            _monitors[sid] = mon
        return mon


def reset_monitor(session_id: str) -> None:
    sid = (session_id or "default").strip() or "default"
    with _monitors_lock:
        mon = _monitors.get(sid)
        if mon is not None:
            mon.reset()
