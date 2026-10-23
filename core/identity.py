"""Gắn tên học sinh (ArcFace) vào track YOLO và giữ tên ổn định qua nhiều khung.

Tách 2 phần để chạy ở 2 luồng:
  - update_tracks(): nhanh — lọc OSD, ghi vị trí, chuyển tên khi track ID đổi
  - recognize():     nặng — InsightFace quét mặt toàn khung + crop từng người
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from core.face_gallery import FaceRecognizer, get_face_app, sim_to_pct

RECOG_EVERY_UNKNOWN = 1  # thử lại sớm — phủ đều cả lớp qua nhiều vòng
RECOG_EVERY_KNOWN = 12  # đã khóa: kiểm tra lại thưa hơn (tránh gỡ tên khi cúi/xa)
# Crop thất bại liên tiếp → giãn nhịp thử lại (mỗi crop tốn nhiều lượt detect mặt)
RECOG_BACKOFF_MAX = 8
MAX_RECOG_PER_ROUND = 22  # nhận diện N người / vòng (luân phiên cả lớp)
MAX_REVERIFY_PER_ROUND = 4  # số track đã khóa được kiểm tra lại / vòng
RECOG_BANDS = 3  # chia khung: trên (cuối lớp) / giữa / dưới (gần cam)
MIN_BOX_W = 22
MIN_BOX_H = 36
STICKY_TTL = 120  # giữ tên ~N lần cập nhật track sau khi track ID đổi / mất tạm
STICKY_IOU = 0.20  # IoU tối thiểu để gắn lại tên vào box mới cùng chỗ
# Ngưỡng khóa cứng nằm trong FaceRecognizer (hard_lock_sim / hard_lock_margin, chỉnh trên giao diện)
HARD_STICKY_TTL = 900  # giữ mã HS khóa cứng khi mất tạm (~15 phút ở ~1 lần cập nhật/giây)
HARD_RECOG_EVERY = 24  # khóa cứng: kiểm tra lại thưa hơn nữa
# Track đang giữ mã không thấy trong N lần cập nhật → coi như vắng, track mới được nhận lại mã
OWNER_ABSENT_UPDATES = 3
# OSD camera (EZVIZ…): góc trên thường là logo/timestamp — không phải người
OSD_TOP_FRAC = 0.18
OSD_SIDE_FRAC = 0.28
OSD_MAX_AREA_FRAC = 0.04
NO_FACE_HIDE_AFTER = 2
FACE_SWEEP_EVERY = 2
FACE_SWEEP_MAX = 40
# Detector InsightFace luôn resize về det_size; ảnh nguồn lớn hơn chỉ làm crop mặt căn chỉnh nét hơn
# (đo trên video lớp: 960px → top-1 trung vị 40%, 1280px → 47%)
FACE_SWEEP_MAX_SIDE = 1280
HEAD_POSE_KEEP = 30  # bỏ góc đầu cũ hơn N lần cập nhật track

Track = tuple[int, int, int, int, int]  # (tid, x1, y1, x2, y2)


def _pick_fair_recog(
    candidates: list[tuple[int, int, int, int, int, float]],
    max_n: int,
    frame_h: int,
    id_last_recog: dict,
) -> list[tuple[int, int, int, int, int, float]]:
    """
    Luân phiên đều theo vị trí dọc khung (cuối lớp / giữa / gần cam)
    + ưu tiên người lâu chưa được nhận diện — không chỉ front hay back.
    """
    if not candidates or max_n <= 0:
        return []
    n_bands = max(1, RECOG_BANDS)
    bands: list[list[tuple[int, int, int, int, int, float]]] = [
        [] for _ in range(n_bands)
    ]
    for t in candidates:
        _tid, _x1, y1, _x2, y2, _area = t
        cy = 0.5 * (y1 + y2)
        bi = int(cy / max(frame_h, 1) * n_bands)
        bi = min(n_bands - 1, max(0, bi))
        bands[bi].append(t)

    for bi in range(n_bands):
        bands[bi].sort(
            key=lambda t: (
                id_last_recog.get(t[0], -10**9),
                abs(t[5] - 8000.0),  # ưu tiên kích thước "vừa" trong lớp
            )
        )

    selected: list[tuple[int, int, int, int, int, float]] = []
    seen: set[int] = set()
    idx = [0] * n_bands
    while len(selected) < max_n:
        progressed = False
        for bi in range(n_bands):
            while idx[bi] < len(bands[bi]):
                t = bands[bi][idx[bi]]
                idx[bi] += 1
                if t[0] in seen:
                    continue
                selected.append(t)
                seen.add(t[0])
                progressed = True
                break
            if len(selected) >= max_n:
                break
        if not progressed:
            break
    return selected


def is_osd_false_person(
    x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int
) -> bool:
    """Lọc logo/watermark camera (góc trên) bị YOLO nhầm thành người."""
    if frame_w <= 0 or frame_h <= 0:
        return False
    bw, bh = max(0, x2 - x1), max(0, y2 - y1)
    if bw < 2 or bh < 2:
        return True
    area_frac = float(bw * bh) / float(frame_w * frame_h)
    ar = bw / max(bh, 1)
    top = y2 <= frame_h * OSD_TOP_FRAC
    left = x2 <= frame_w * OSD_SIDE_FRAC
    right = x1 >= frame_w * (1.0 - OSD_SIDE_FRAC)
    if top and (left or right) and area_frac <= OSD_MAX_AREA_FRAC:
        return True
    if y2 <= frame_h * 0.12 and ar >= 2.0 and area_frac <= 0.05:
        return True
    if ar > 2.8 or ar < 0.18:
        return True
    if area_frac < 0.0008:
        return True
    return False


def _face_center_in_person(
    fx1: float, fy1: float, fx2: float, fy2: float,
    px1: int, py1: int, px2: int, py2: int,
) -> bool:
    """Mặt nằm trong nửa trên box người (vùng đầu)."""
    fcx = 0.5 * (fx1 + fx2)
    fcy = 0.5 * (fy1 + fy2)
    if not (px1 <= fcx <= px2 and py1 <= fcy <= py2):
        return False
    return fcy <= py1 + 0.62 * (py2 - py1)


def _best_track_for_face(
    fx1: float, fy1: float, fx2: float, fy2: float, parsed: list[Track],
) -> int | None:
    """Gắn face bbox vào track YOLO gần nhất (ưu tiên mặt trong vùng đầu)."""
    best_tid = None
    best_score = 0.0
    f_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
    for tid, x1, y1, x2, y2 in parsed:
        hy2 = y1 + int(0.55 * (y2 - y1))
        ix1, iy1 = max(fx1, x1), max(fy1, y1)
        ix2, iy2 = min(fx2, x2), min(fy2, hy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            if _face_center_in_person(fx1, fy1, fx2, fy2, x1, y1, x2, y2):
                score = 0.15
            else:
                continue
        else:
            head_area = max(1.0, (x2 - x1) * max(1, hy2 - y1))
            score = inter / min(f_area, head_area)
            if _face_center_in_person(fx1, fy1, fx2, fy2, x1, y1, x2, y2):
                score += 0.35
        if score > best_score:
            best_score = score
            best_tid = tid
    return best_tid if best_score >= 0.12 else None


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class IdentityTracker:
    """Trạng thái tên ↔ track cho 1 phiên video. Mọi map được bảo vệ bởi self.lock."""

    def __init__(self, recognizer: FaceRecognizer):
        self.rec = recognizer
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.id_names: dict[int, str] = {}
            self.id_student: dict[int, str] = {}
            self.id_locked: set[int] = set()
            self.id_hard: set[int] = set()
            self.id_hits: dict[int, dict[str, int]] = {}
            self.id_best_sim: dict[int, float] = {}
            self.id_last_pct: dict[int, float] = {}
            self.id_noface: dict[int, int] = {}
            self.id_has_face: set[int] = set()
            self.id_last_recog: dict[int, int] = {}
            self.id_fail: dict[int, int] = {}
            self.id_last_box: dict[int, tuple[int, int, int, int]] = {}
            self.id_last_seen: dict[int, int] = {}
            self.id_head_pose: dict[int, tuple[float, float, int]] = {}
            self.sid_owner: dict[str, int] = {}
            self.frame_i = 0
            self.recog_i = 0

    # ---------- thao tác trên map (gọi khi đã giữ lock) ----------

    def _clear_identity(self, tid: int) -> None:
        sid_old = self.id_student.pop(tid, None)
        self.id_names.pop(tid, None)
        self.id_locked.discard(tid)
        self.id_hard.discard(tid)
        self.id_hits.pop(tid, None)
        self.id_best_sim.pop(tid, None)
        self.id_last_pct.pop(tid, None)
        self.id_noface.pop(tid, None)
        self.id_has_face.discard(tid)
        if sid_old and self.sid_owner.get(sid_old) == tid:
            self.sid_owner.pop(sid_old, None)

    def _transfer_lock(self, old_tid: int, new_tid: int) -> None:
        """Chuyển tên đã khóa từ track cũ sang track mới (cùng vị trí)."""
        sid = self.id_student.get(old_tid)
        name = self.id_names.get(old_tid)
        if not sid or not name:
            return
        if new_tid in self.id_locked and self.id_student.get(new_tid) not in (None, sid):
            return
        self.id_student[new_tid] = sid
        self.id_names[new_tid] = name
        self.id_locked.add(new_tid)
        self.sid_owner[sid] = new_tid
        self.id_best_sim[new_tid] = max(
            self.id_best_sim.get(new_tid, 0.0), self.id_best_sim.get(old_tid, 0.0)
        )
        if old_tid in self.id_last_pct:
            self.id_last_pct[new_tid] = max(
                self.id_last_pct.get(new_tid, 0.0), self.id_last_pct[old_tid]
            )
        if old_tid in self.id_has_face:
            self.id_has_face.add(new_tid)
        if old_tid in self.id_noface:
            self.id_noface[new_tid] = self.id_noface[old_tid]
        if old_tid in self.id_hits:
            self.id_hits[new_tid] = self.id_hits[old_tid]
        if old_tid in self.id_hard:
            self.id_hard.add(new_tid)
        if old_tid != new_tid:
            self._clear_identity(old_tid)
            self.id_last_box.pop(old_tid, None)
            self.id_last_seen.pop(old_tid, None)
            self.sid_owner[sid] = new_tid

    def _is_hard(self, sim: float, margin: float) -> bool:
        return sim >= self.rec.hard_lock_sim and margin >= self.rec.hard_lock_margin

    def drop_below(self, match_threshold: float, hard_lock_sim: float) -> int:
        """Sau khi nâng ngưỡng: bỏ tên các track khớp tốt nhất dưới ngưỡng mới. Trả về số track bị bỏ."""
        with self.lock:
            dropped = [t for t in self.id_locked if self.id_best_sim.get(t, 0.0) < match_threshold]
            for tid in dropped:
                self._clear_identity(tid)
            for tid in [t for t in self.id_hard if self.id_best_sim.get(t, 0.0) < hard_lock_sim]:
                self.id_hard.discard(tid)
            return len(dropped)

    def _owner_absent(self, owner: int) -> bool:
        return self.frame_i - self.id_last_seen.get(owner, -10**9) > OWNER_ABSENT_UPDATES

    def _pin_identity(self, tid: int, sid: str, name: str, sim: float, hard: bool) -> None:
        """Ghim SID vào track tại vị trí hiện tại."""
        old_sid = self.id_student.get(tid)
        if old_sid and old_sid != sid and self.sid_owner.get(old_sid) == tid:
            self.sid_owner.pop(old_sid, None)
        if old_sid != sid:
            self.id_hard.discard(tid)
        if hard:
            self.id_hard.add(tid)
        self.id_student[tid] = sid
        self.id_names[tid] = name
        self.sid_owner[sid] = tid
        self.id_best_sim[tid] = float(sim)
        self.id_last_pct[tid] = sim_to_pct(sim)
        self.id_locked.add(tid)
        self.id_has_face.add(tid)
        self.id_noface[tid] = 0
        hits = self.id_hits.setdefault(tid, {})
        hits[sid] = hits.get(sid, 0) + 1

    def _try_pin_from_match(
        self, tid: int, sid: str | None, name: str | None, sim: float, margin: float = 0.0
    ) -> None:
        """Ghim / nâng cấp theo %: track đã có mã chỉ đổi sang HS khác khi % cao hơn."""
        if sid is None or name is None or not self.rec.is_compatible(sim):
            return
        hard = self._is_hard(sim, margin)
        if tid in self.id_locked:
            pinned = float(self.id_best_sim.get(tid, 0.0))
            cur = self.id_student.get(tid)
            if sid == cur:
                if sim > pinned:
                    self.id_best_sim[tid] = sim
                    self.id_last_pct[tid] = sim_to_pct(sim)
                if hard:
                    self.id_hard.add(tid)
                return
            if sim <= pinned:
                return
        owner = self.sid_owner.get(sid)
        if owner is not None and owner != tid and owner in self.id_locked:
            owner_best = float(self.id_best_sim.get(owner, 0.0))
            if (self._owner_absent(owner)
                    and sim >= self.rec.lock_sim and tid not in self.id_locked):
                # HS xuất hiện lại với track mới (track cũ đã mất) → nhận lại mã (kể cả khóa cứng)
                self._transfer_lock(owner, tid)
                if sim > float(self.id_best_sim.get(tid, 0.0)):
                    self.id_best_sim[tid] = sim
                self.id_last_pct[tid] = sim_to_pct(sim)
                self.id_has_face.add(tid)
                self.id_noface[tid] = 0
                return
            if sim <= owner_best:
                return
        if owner is not None and owner != tid:
            if owner in self.id_locked:
                self._clear_identity(owner)
            else:
                self.id_names.pop(owner, None)
                self.id_student.pop(owner, None)
                self.sid_owner.pop(sid, None)
        self._pin_identity(tid, sid, name, sim, hard)

    # ---------- luồng phân tích (nhanh) ----------

    def update_tracks(self, raw: list[Track], width: int, height: int) -> list[Track]:
        """Lọc OSD, ghi vị trí track, chuyển tên khi track ID đổi. Trả về track hợp lệ."""
        with self.lock:
            frame_i = self.frame_i
            self.frame_i += 1
            parsed: list[Track] = []
            present: set[int] = set()
            for tid, x1, y1, x2, y2 in raw:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                if is_osd_false_person(x1, y1, x2, y2, width, height):
                    if tid in self.id_locked or tid in self.id_student:
                        self._clear_identity(tid)
                    continue
                parsed.append((tid, x1, y1, x2, y2))
                present.add(tid)
                self.id_last_box[tid] = (x1, y1, x2, y2)
                self.id_last_seen[tid] = frame_i

            orphans = [
                otid for otid in list(self.id_locked)
                if otid not in present and otid in self.id_last_box
            ]
            taken_new: set[int] = set()
            for otid in orphans:
                ttl = HARD_STICKY_TTL if otid in self.id_hard else STICKY_TTL
                if frame_i - self.id_last_seen.get(otid, -999) > ttl:
                    self._clear_identity(otid)
                    self.id_last_box.pop(otid, None)
                    self.id_last_seen.pop(otid, None)
                    continue
                obox = self.id_last_box[otid]
                best_tid = None
                best_iou = STICKY_IOU
                for ntid, x1, y1, x2, y2 in parsed:
                    if ntid in self.id_locked or ntid in taken_new:
                        continue
                    ov = box_iou(obox, (x1, y1, x2, y2))
                    if ov > best_iou:
                        best_iou = ov
                        best_tid = ntid
                if best_tid is not None:
                    self._transfer_lock(otid, best_tid)
                    taken_new.add(best_tid)
            for tid in [t for t, p in self.id_head_pose.items() if frame_i - p[2] > HEAD_POSE_KEEP]:
                del self.id_head_pose[tid]
            return parsed

    def head_pose(self, tid: int, max_age: int) -> tuple[float, float] | None:
        """(yaw, pitch) độ từ InsightFace nếu đo trong N lần cập nhật gần nhất.

        yaw dương = quay sang phải ảnh, pitch dương = cúi xuống.
        """
        with self.lock:
            p = self.id_head_pose.get(tid)
            if p is None or self.frame_i - 1 - p[2] > max_age:
                return None
            return p[0], p[1]

    def describe(self, tid: int) -> dict:
        """Tên hiện tại của track (không chạy ArcFace)."""
        with self.lock:
            known = tid in self.id_locked and bool(self.id_names.get(tid))
            pct = self.id_last_pct.get(tid)
            if pct is None and tid in self.id_best_sim:
                pct = sim_to_pct(self.id_best_sim[tid])
            no_face_suspect = (
                tid not in self.id_locked
                and tid not in self.id_has_face
                and self.id_noface.get(tid, 0) >= NO_FACE_HIDE_AFTER
            )
            return {
                "known": known,
                "student_id": self.id_student.get(tid) if known else None,
                "name": self.id_names.get(tid) if known else None,
                "match_pct": round(pct, 1) if pct is not None else None,
                "hard_locked": known and tid in self.id_hard,
                "no_face_suspect": no_face_suspect,
            }

    def identified_count(self) -> int:
        with self.lock:
            return len([t for t in self.id_locked if t in self.id_names])

    def hard_locked_count(self) -> int:
        with self.lock:
            return len([t for t in self.id_hard if t in self.id_names])

    # ---------- luồng ArcFace (nặng) ----------

    def recognize(self, frame: np.ndarray, parsed: list[Track]) -> None:
        """1 vòng nhận diện trên khung + các track tương ứng của khung đó."""
        if not parsed:
            return
        height, width = frame.shape[:2]

        with self.lock:
            recog_i = self.recog_i
            self.recog_i += 1
            recog_candidates = []
            reverify_candidates = []
            for tid, x1, y1, x2, y2 in parsed:
                locked = tid in self.id_locked
                if locked:
                    every = HARD_RECOG_EVERY if tid in self.id_hard else RECOG_EVERY_KNOWN
                else:
                    every = min(RECOG_BACKOFF_MAX,
                                RECOG_EVERY_UNKNOWN + self.id_fail.get(tid, 0))
                last = self.id_last_recog.get(tid, -999)
                bw, bh = x2 - x1, y2 - y1
                if recog_i - last >= every and bw >= MIN_BOX_W and bh >= MIN_BOX_H:
                    item = (tid, x1, y1, x2, y2, float(bw * bh))
                    (reverify_candidates if locked else recog_candidates).append(item)

            reverify_candidates.sort(
                key=lambda t: (self.id_last_recog.get(t[0], -10**9), t[5])
            )
            unnamed = [t for t in recog_candidates if t[0] not in self.id_student]
            named_unlocked = [t for t in recog_candidates if t[0] in self.id_student]
            fair_unnamed = _pick_fair_recog(
                unnamed, MAX_RECOG_PER_ROUND, height, self.id_last_recog
            )
            remain = max(0, MAX_RECOG_PER_ROUND - len(fair_unnamed))
            fair_rest = _pick_fair_recog(
                named_unlocked, remain, height, self.id_last_recog
            )
            to_recog = [t[:5] for t in fair_unnamed + fair_rest]
            to_reverify = [t[:5] for t in reverify_candidates[:MAX_REVERIFY_PER_ROUND]]

        if recog_i % FACE_SWEEP_EVERY == 0:
            self._face_sweep(frame, parsed, recog_i)

        for tid, x1, y1, x2, y2 in to_reverify:
            sid, full_name, sim, margin = self.rec.match_person_crop(frame[y1:y2, x1:x2])
            with self.lock:
                self.id_last_recog[tid] = recog_i
                if sim >= 0.05:
                    self.id_has_face.add(tid)
                    self.id_noface[tid] = 0
                pinned = float(self.id_best_sim.get(tid, 0.0))
                if sim >= 0.05 and sid is not None:
                    self._try_pin_from_match(tid, sid, full_name, sim, margin)
                self.id_last_pct[tid] = sim_to_pct(self.id_best_sim.get(tid, pinned))

        for tid, x1, y1, x2, y2 in to_recog:
            with self.lock:
                if tid in self.id_locked:
                    continue
            sid, full_name, sim, margin = self.rec.match_person_crop(frame[y1:y2, x1:x2])
            with self.lock:
                self.id_last_recog[tid] = recog_i
                self.id_last_pct[tid] = sim_to_pct(sim)
                if sim >= 0.05:
                    self.id_has_face.add(tid)
                    self.id_noface[tid] = 0
                else:
                    self.id_noface[tid] = self.id_noface.get(tid, 0) + 1
                self._try_pin_from_match(tid, sid, full_name, sim, margin)
                if tid in self.id_locked:
                    self.id_fail.pop(tid, None)
                else:
                    self.id_fail[tid] = self.id_fail.get(tid, 0) + 1

    def _face_sweep(self, frame: np.ndarray, parsed: list[Track], recog_i: int) -> None:
        """Quét mặt toàn khung rồi gán vào track (phủ lớp tốt hơn crop từng người)."""
        try:
            face_app = get_face_app()
            sh, sw = frame.shape[:2]
            sweep = frame
            sx = sy = 1.0
            if max(sw, sh) > FACE_SWEEP_MAX_SIDE:
                s = FACE_SWEEP_MAX_SIDE / max(sw, sh)
                sweep = cv2.resize(
                    frame, (int(sw * s), int(sh * s)), interpolation=cv2.INTER_AREA
                )
                sx = sw / sweep.shape[1]
                sy = sh / sweep.shape[0]
            faces = face_app.get(sweep) or []
        except Exception:
            return
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )[:FACE_SWEEP_MAX]
        for face in faces:
            fx1, fy1, fx2, fy2 = [float(v) for v in face.bbox]
            fx1, fx2 = fx1 * sx, fx2 * sx
            fy1, fy2 = fy1 * sy, fy2 * sy
            tid = _best_track_for_face(fx1, fy1, fx2, fy2, parsed)
            if tid is None:
                continue
            emb = np.asarray(face.normed_embedding, dtype=np.float32)
            sid, full_name, sim, margin = self.rec._match_embedding(emb)
            pose = face.get("pose")
            with self.lock:
                if pose is not None:
                    # InsightFace: pose = [pitch, yaw, roll], pitch âm khi cúi
                    self.id_head_pose[tid] = (float(pose[1]), -float(pose[0]), self.frame_i - 1)
                self.id_has_face.add(tid)
                self.id_noface[tid] = 0
                self.id_last_recog[tid] = recog_i
                if sid is not None:
                    self._try_pin_from_match(tid, sid, full_name, sim, margin)
                elif sim > 0:
                    self.id_last_pct[tid] = sim_to_pct(sim)
