"""Kiểm thử tự động Giai đoạn 1 (không cần GPU / video thật).

Chạy từ thư mục gốc dự án:
  python -m tools.run_stage1_tests
  python -m tools.run_stage1_tests --out evidence/tests/stage1_unit_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from core.analysis import MAX_BEHAVIOR_STEP_SEC, effective_fps
from core.behavior_log import (
    ATTENDANCE_MODE,
    PRESENT_MIN_FACE_MATCHES,
    PRESENT_MIN_FRAMES,
    BehaviorRecorder,
)
from core.face_gallery import FaceRecognizer
from core.identity import IOU_TRANSFER_LOCK, IdentityTracker
from core.video_clock import VideoClock


def _ok(name: str, cond: bool, detail: str = "") -> dict:
    return {"name": name, "pass": bool(cond), "detail": detail}


def test_arcface_margin() -> list[dict]:
    rec = FaceRecognizer.__new__(FaceRecognizer)
    rec.match_threshold = 0.45
    rec.min_compat_pct = 45.0
    rec.min_margin = 0.004
    rec.lock_sim = 0.55
    out = []
    # Trường hợp CHINHCODE: top1=0.460, margin=0.001 → từ chối
    out.append(_ok(
        "AF01_ambiguous_reject",
        not rec._accept_match(0.460, 0.001, 2),
        "best=0.460 margin=0.001 phải từ chối",
    ))
    out.append(_ok(
        "AF02_lock_sim_accept",
        rec._accept_match(0.560, 0.001, 2),
        "best >= lock_sim chấp nhận dù margin nhỏ",
    ))
    out.append(_ok(
        "AF03_margin_accept",
        rec._accept_match(0.460, 0.010, 2),
        "margin >= min_margin chấp nhận",
    ))
    out.append(_ok(
        "AF04_single_gallery",
        rec._accept_match(0.460, 0.460, 1),
        "gallery 1 HS: chỉ cần tương thích",
    ))
    out.append(_ok(
        "AF05_below_threshold",
        not rec._accept_match(0.400, 0.100, 2),
        "dưới ngưỡng từ chối",
    ))
    # Quy tắc cũ near_ok (margin>=0) sẽ chấp nhận AF01 — xác nhận đã bỏ
    old_near_ok = 0.460 >= (0.45 + 0.04) or 0.001 >= 0.0
    out.append(_ok(
        "AF06_old_near_ok_would_accept",
        old_near_ok is True,
        "minh chứng quy tắc cũ chấp nhận trường hợp mơ hồ",
    ))
    return out


def test_identity_iou() -> list[dict]:
    out = []
    out.append(_ok(
        "ID01_iou_transfer_disabled",
        IOU_TRANSFER_LOCK is False,
        "không chuyển khóa chỉ vì IoU",
    ))

    class _DummyRec:
        match_threshold = 0.45
        lock_sim = 0.55
        hard_lock_sim = 0.70
        hard_lock_margin = 0.02

        def is_compatible(self, sim: float) -> bool:
            return sim >= self.match_threshold

    tr = IdentityTracker(_DummyRec())  # type: ignore[arg-type]
    with tr.lock:
        tr.id_student[10] = "HS08"
        tr.id_names[10] = "Hoc Sinh 08"
        tr.id_locked.add(10)
        tr.id_best_sim[10] = 0.60
        tr.id_last_box[10] = (100, 100, 200, 300)
        tr.id_last_seen[10] = 0
        tr.sid_owner["HS08"] = 10
        tr.frame_i = 5

    # Track 10 biến mất, track 27 xuất hiện cùng chỗ → chỉ pending, không khóa
    parsed = tr.update_tracks([(27, 105, 105, 205, 305)], 640, 480)
    out.append(_ok("ID02_new_track_present", any(t[0] == 27 for t in parsed), "track mới được giữ"))
    with tr.lock:
        pending = tr.id_pending.get(27)
        locked_27 = 27 in tr.id_locked
        still_orphan = 10 in tr.id_locked
    out.append(_ok(
        "ID03_pending_not_locked",
        pending is not None and pending[0] == "HS08" and not locked_27,
        f"pending={pending} locked_27={locked_27}",
    ))
    out.append(_ok(
        "ID04_orphan_kept_until_verify",
        still_orphan,
        "track cũ giữ khóa đến khi ArcFace xác minh / hết TTL",
    ))

    # ArcFace xác minh cùng HS trên track 27 → chuyển khóa
    with tr.lock:
        tr._try_pin_from_match(27, "HS08", "Hoc Sinh 08", 0.62, 0.02)
        locked_27 = 27 in tr.id_locked
        owner = tr.sid_owner.get("HS08")
        orphan_cleared = 10 not in tr.id_locked
    out.append(_ok(
        "ID05_arcface_confirms_transfer",
        locked_27 and owner == 27 and orphan_cleared,
        f"locked_27={locked_27} owner={owner} orphan_cleared={orphan_cleared}",
    ))

    # HS09 ngồi vào chỗ HS08: IoU đề cử nhưng ArcFace khác người → không giữ tên HS08
    tr2 = IdentityTracker(_DummyRec())  # type: ignore[arg-type]
    with tr2.lock:
        tr2.id_student[12] = "HS08"
        tr2.id_names[12] = "Hoc Sinh 08"
        tr2.id_locked.add(12)
        tr2.id_best_sim[12] = 0.60
        tr2.id_last_box[12] = (100, 100, 200, 300)
        tr2.id_last_seen[12] = 0
        tr2.sid_owner["HS08"] = 12
        tr2.frame_i = 3
    tr2.update_tracks([(40, 100, 100, 200, 300)], 640, 480)
    with tr2.lock:
        tr2._try_pin_from_match(40, "HS09", "Hoc Sinh 09", 0.58, 0.03)
        sid40 = tr2.id_student.get(40) if 40 in tr2.id_locked else None
    out.append(_ok(
        "ID06_seat_swap_no_hs08",
        sid40 == "HS09",
        f"sau ArcFace track mới là {sid40}, không giữ HS08 từ IoU",
    ))
    return out


def _det(tid: int, sid: str | None, reason: str | None = None, *,
         known: bool = True, posed: bool = True, face_events: int = 0,
         x1=10, y1=10, x2=100, y2=200) -> dict:
    reasons = [reason] if reason else []
    return {
        "track_id": tid,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "known": known and bool(sid),
        "student_id": sid if known else None,
        "name": sid or "",
        "match_pct": 50.0 if sid else None,
        "face_match_events": face_events,
        "posed": posed,
        "reasons": reasons,
        "group": "activity" if reason and "Giơ" in reason else (
            "unknown" if reason and "Không" in reason else "clear"
        ),
    }


def test_behavior_log_e01_e06() -> list[dict]:
    out = []
    clock = VideoClock(25.0, False)
    names = {"HS01": "A", "HS02": "B", "HS03": "C"}

    # E01: có mặt đúng
    r = BehaviorRecorder(25.0, clock, "test", student_names=names)
    for i in range(5):
        r.record(i, [_det(1, "HS01", face_events=1)], wall=i / 25.0)
    s = r.summary()
    present = {p["student_id"] for p in s["present"]}
    out.append(_ok("E01_present", "HS01" in present, f"present={present}"))

    # E02: vắng
    out.append(_ok("E02_absent", "HS03" in {a["student_id"] for a in s["absent"]}, "HS03 vắng"))

    # E03: có mặt nhưng không nhận diện → không gán nhầm
    r3 = BehaviorRecorder(25.0, clock, "test", student_names=names)
    for i in range(5):
        r3.record(i, [_det(7, None, known=False, posed=True, reason="Không xác định")], wall=i / 25.0)
    s3 = r3.summary()
    out.append(_ok(
        "E03_no_false_id",
        all(p["student_id"] in names for p in s3["present"]) and not s3["present"],
        f"present={s3['present']}",
    ))

    # E04: một HS hai track → không nhân đôi
    r4 = BehaviorRecorder(25.0, clock, "test", student_names=names)
    for i in range(4):
        r4.record(i, [
            _det(1, "HS01", face_events=2),
            _det(2, "HS01", face_events=1),
        ], wall=i / 25.0)
    s4 = r4.summary()
    n_hs01 = sum(1 for p in s4["present"] if p["student_id"] == "HS01")
    out.append(_ok("E04_no_duplicate", n_hs01 == 1, f"count HS01={n_hs01}"))

    # E05: hành vi liên tục → 1 đợt (không tách vô lý trong gap)
    r5 = BehaviorRecorder(25.0, clock, "test", student_names=names)
    for i in range(0, 50, 5):
        r5.record(i, [_det(1, "HS01", "Giơ tay (1.0s)", face_events=3)], wall=i / 25.0)
    with r5._lock:
        eps = r5._closed + list(r5._open.values())
        raise_eps = [e for e in eps if e.behavior == "Giơ tay"]
    out.append(_ok(
        "E05_single_episode",
        len(raise_eps) == 1,
        f"episodes={len(raise_eps)} starts={[e.start_frame for e in raise_eps]}",
    ))

    # E06: mất dấu giữa sự kiện — khoảng gap lớn kết thúc đợt, không cộng thời gian thiếu
    r6 = BehaviorRecorder(25.0, clock, "test", student_names=names)
    for i in range(0, 30, 5):
        r6.record(i, [_det(1, "HS01", "Giơ tay (1.0s)", face_events=3)], wall=i / 25.0)
    # Gap > EPISODE_GAP_SEC (1s = 25 frames)
    r6.record(80, [_det(1, "HS01", "Giơ tay (1.0s)", face_events=3)], wall=80 / 25.0)
    with r6._lock:
        eps = r6._closed + list(r6._open.values())
        raise_eps = [e for e in eps if e.behavior == "Giơ tay"]
        durs = [(e.last_frame - e.start_frame) / 25.0 for e in raise_eps]
    out.append(_ok(
        "E06_gap_splits_episode",
        len(raise_eps) >= 2 and all(d < 3.0 for d in durs[:-1]),
        f"n={len(raise_eps)} durs={durs}",
    ))

    # Gộp theo sự kiện: track đổi HS giữa chừng không gán hết cho 1 người
    r7 = BehaviorRecorder(25.0, clock, "test", student_names=names)
    for i in range(0, 20, 5):
        r7.record(i, [_det(12, "HS01", "Viết bài (1.0s)", face_events=2)], wall=i / 25.0)
    for i in range(40, 60, 5):
        r7.record(i, [_det(12, "HS02", "Giơ tay (1.0s)", face_events=2)], wall=i / 25.0)
    with r7._lock:
        eps = r7._closed + list(r7._open.values())
        by_sid = {}
        for e in eps:
            by_sid.setdefault(e.student_id, []).append(e.behavior)
    out.append(_ok(
        "E07_episode_sid_split",
        "Viết bài" in by_sid.get("HS01", []) and "Giơ tay" in by_sid.get("HS02", []),
        f"by_sid={by_sid}",
    ))

    # Điểm danh face_match_events vs identity_observations
    r8 = BehaviorRecorder(
        25.0, clock, "test", student_names=names, attendance_mode="face_match_events",
    )
    # 5 quan sát có danh tính nhưng chỉ 1 lần ArcFace → không đủ nếu min=3
    for i in range(5):
        r8.record(i, [_det(1, "HS01", face_events=1)], wall=i / 25.0)
    s8 = r8.summary()
    out.append(_ok(
        "E08_strict_face_match_absent",
        "HS01" not in {p["student_id"] for p in s8["present"]},
        "1 face_match_event không đủ PRESENT_MIN_FACE_MATCHES=3",
    ))
    r9 = BehaviorRecorder(
        25.0, clock, "test", student_names=names, attendance_mode="face_match_events",
    )
    for i in range(5):
        r9.record(i, [_det(1, "HS01", face_events=3)], wall=i / 25.0)
    s9 = r9.summary()
    out.append(_ok(
        "E09_strict_face_match_present",
        "HS01" in {p["student_id"] for p in s9["present"]},
        "≥3 face_match_events → có mặt",
    ))
    out.append(_ok(
        "E10_default_mode",
        ATTENDANCE_MODE == "identity_observations",
        f"mode={ATTENDANCE_MODE}",
    ))
    return out


def test_timing() -> list[dict]:
    out = []
    # Video 25 FPS, bỏ 50 khung → fps/50=0.5 nhưng sàn 1.0
    fps_eff = effective_fps(25.0, 50)
    out.append(_ok(
        "T01_fps_eff_floor",
        abs(fps_eff - 1.0 / MAX_BEHAVIOR_STEP_SEC) < 1e-9,
        f"fps_eff={fps_eff}",
    ))
    out.append(_ok(
        "T02_every_frame",
        abs(effective_fps(25.0, 1) - 25.0) < 1e-9,
        f"fps_eff={effective_fps(25.0, 1)}",
    ))
    out.append(_ok(
        "T03_skip_5",
        abs(effective_fps(25.0, 5) - 5.0) < 1e-9,
        f"fps_eff={effective_fps(25.0, 5)}",
    ))
    # Thời lượng sự kiện theo khung video, không theo bộ đếm
    clock = VideoClock(25.0, False)
    r = BehaviorRecorder(25.0, clock, "test", student_names={"HS01": "A"})
    # Hai quan sát cách 20 khung (0.8s) < EPISODE_GAP_SEC → cùng một đợt
    r.record(250, [_det(1, "HS01", "Cúi đầu lâu (5.2s)", face_events=3)], wall=10.0)
    r.record(270, [_det(1, "HS01", "Cúi đầu lâu (5.2s)", face_events=3)], wall=10.8)
    with r._lock:
        ep = (r._closed + list(r._open.values()))[0]
        dur = (ep.last_frame - ep.start_frame) / 25.0
    out.append(_ok(
        "T04_no_backdate_start",
        ep.start_frame == 250,
        f"start={ep.start_frame} (không nội suy 5.2s ngược)",
    ))
    out.append(_ok(
        "T05_video_duration",
        abs(dur - 0.8) < 1e-6,
        f"dur={dur}s từ khung 250→270 (thời gian video)",
    ))
    return out


def test_missing_keypoint_constants() -> list[dict]:
    from core.behavior import UNKNOWN_BEHAVIOR, observation_group
    src = (BASE_DIR / "core" / "analysis.py").read_text(encoding="utf-8")
    return [
        _ok("KP01_unknown_label", UNKNOWN_BEHAVIOR == "Không xác định", UNKNOWN_BEHAVIOR),
        _ok(
            "KP02_decay_on_missing",
            "state_tracker.decay" in src and "no_keypoints" in src,
            "analysis.py gọi decay và gắn no_keypoints",
        ),
        _ok(
            "KP03_unknown_group",
            observation_group([UNKNOWN_BEHAVIOR]) == "unknown",
            "Nhóm quan sát = unknown",
        ),
    ]


def test_ernu_matrix_and_metrics() -> list[dict]:
    """Ma trận A×B, C_min→U, xung đột B2+B4 ở A2, ví dụ 5 phút trong bộ tiêu chí."""
    from core.behavior import ACTIVITY_BEHAVIORS, NEUTRAL_BEHAVIORS, observation_group
    from core.config import ErnuConfig
    from core.ernu import (
        BehaviorInterval,
        ObservationInterval,
        WindowLabel,
        analyze_session,
        label_window,
        matrix_label,
        metrics_from_windows,
    )

    out = []
    out.append(_ok("ER01_B1_A1_E", matrix_label("B1", "A1") == "E", matrix_label("B1", "A1")))
    out.append(_ok("ER02_B2_A2_E", matrix_label("B2", "A2") == "E", matrix_label("B2", "A2")))
    out.append(_ok("ER03_B2_A1_E_auto", matrix_label("B2", "A1") == "E", matrix_label("B2", "A1")))
    out.append(_ok("ER04_B3_always_N", matrix_label("B3", "A5") == "N", matrix_label("B3", "A5")))
    out.append(_ok("ER05_B4_A1_N", matrix_label("B4", "A1") == "N", matrix_label("B4", "A1")))
    out.append(_ok("ER06_B4_A2_N", matrix_label("B4", "A2") == "N", matrix_label("B4", "A2")))
    out.append(_ok("ER07_B5_A3_R", matrix_label("B5", "A3") == "R", matrix_label("B5", "A3")))
    out.append(_ok(
        "ER08_quay_dau_not_activity",
        "Quay đầu" not in ACTIVITY_BEHAVIORS and "Quay đầu" in NEUTRAL_BEHAVIORS,
        f"activity={ACTIVITY_BEHAVIORS}",
    ))
    out.append(_ok(
        "ER09_quay_dau_group_neutral",
        observation_group(["Quay đầu (2.1s)"]) == "neutral",
        observation_group(["Quay đầu (2.1s)"]),
    ))

    cfg = ErnuConfig()
    # Coverage thấp → U
    low = label_window(
        video_id="VID", student_id="S01", start_sec=0, end_sec=10, activity_code="A1",
        behavior_events=[], observation_spans=[(0.0, 2.0)], cfg=cfg,
    )
    out.append(_ok("ER10_cmin_to_U", low.pred_label == "U", f"{low.pred_label} cov={low.valid_coverage}"))

    # B2+B4 trong A2 với coverage đủ → E (không R chỉ vì cúi)
    conflict = label_window(
        video_id="VID", student_id="S01", start_sec=0, end_sec=10, activity_code="A2",
        behavior_events=[
            BehaviorInterval("S01", "Viết bài", 0, 7),
            BehaviorInterval("S01", "Cúi đầu lâu", 1, 7),
        ],
        observation_spans=[(0.0, 10.0)],
        cfg=cfg,
    )
    out.append(_ok(
        "ER11_B2_B4_A2_E",
        conflict.pred_label == "E" and not conflict.mixed,
        f"{conflict.pred_label} mixed={conflict.mixed} {conflict.reason}",
    ))

    # T_valid=0 → Chưa đủ dữ liệu
    empty = metrics_from_windows("S01", [
        WindowLabel("V", "S01", 0, 10, "A1", [], 0.2, "U", "low", False, "ernu-v1"),
        WindowLabel("V", "S01", 10, 20, "A1", [], 0.1, "U", "low", False, "ernu-v1"),
    ], cfg)
    out.append(_ok(
        "ER12_tvalid0_status",
        empty.data_status == "Chưa đủ dữ liệu" and empty.P_E is None,
        f"status={empty.data_status} P_E={empty.P_E}",
    ))

    # Ví dụ 5 phút: 18E + 5R + 4N + 3U cửa sổ 10s
    wins = []
    labels = ["E"] * 18 + ["R"] * 5 + ["N"] * 4 + ["U"] * 3
    for i, lab in enumerate(labels):
        wins.append(WindowLabel(
            "V", "S01", i * 10, (i + 1) * 10, "A1", [], 1.0, lab, lab, False, "ernu-v1",
        ))
    m = metrics_from_windows("S01", wins, cfg)
    out.append(_ok("ER13_ex5_TE", abs(m.T_E - 180) < 1e-6, f"T_E={m.T_E}"))
    out.append(_ok("ER14_ex5_TR", abs(m.T_R - 50) < 1e-6, f"T_R={m.T_R}"))
    out.append(_ok("ER15_ex5_TN", abs(m.T_N - 40) < 1e-6, f"T_N={m.T_N}"))
    out.append(_ok("ER16_ex5_TU", abs(m.T_U - 30) < 1e-6, f"T_U={m.T_U}"))
    out.append(_ok("ER17_ex5_C", m.C == 90.0, f"C={m.C}"))
    out.append(_ok("ER18_ex5_PE", m.P_E == 66.7, f"P_E={m.P_E}"))
    out.append(_ok("ER19_ex5_PR", m.P_R == 18.5, f"P_R={m.P_R}"))
    out.append(_ok("ER20_ex5_PN", m.P_N == 14.8, f"P_N={m.P_N}"))
    out.append(_ok("ER21_ex5_PU", m.P_U == 10.0, f"P_U={m.P_U}"))
    # BEEI mặc định w_N=0.5: 100*(180+0+20)/270 = 74.1
    out.append(_ok("ER22_ex5_BEEI", m.BEEI == 74.1, f"BEEI={m.BEEI}"))
    out.append(_ok("ER22b_ex5_publishable", m.publishable is True, f"pub={m.publishable} C={m.C}"))
    out.append(_ok(
        "ER22c_ex5_interp",
        "66,7%" in m.interpretation.replace(".", ",") or "66.7%" in m.interpretation,
        m.interpretation[:80],
    ))

    # Ví dụ 10 phút: T_E=350, T_R=80, T_N=120, T_U=50 → C=91.7; P_E=63.6; P_R=14.5; P_N=21.8; P_U=8.3
    from core.ernu import beei_score
    wins10 = []
    # 35E + 8R + 12N + 5U cửa sổ 10s
    for i, lab in enumerate(["E"] * 35 + ["R"] * 8 + ["N"] * 12 + ["U"] * 5):
        wins10.append(WindowLabel(
            "V", "S01", i * 10, (i + 1) * 10, "A1", [], 1.0, lab, lab, False, "ernu-v1",
        ))
    m10 = metrics_from_windows("S01", wins10, cfg)
    out.append(_ok("ER26_ex10_C", m10.C == 91.7, f"C={m10.C}"))
    out.append(_ok("ER27_ex10_PE", m10.P_E == 63.6, f"P_E={m10.P_E}"))
    out.append(_ok("ER28_ex10_PR", m10.P_R == 14.5, f"P_R={m10.P_R}"))
    out.append(_ok("ER29_ex10_PN", m10.P_N == 21.8, f"P_N={m10.P_N}"))
    out.append(_ok("ER30_ex10_PU", m10.P_U == 8.3, f"P_U={m10.P_U}"))

    # BEEI độ nhạy (ví dụ tài liệu gốc phút): T_E=28, T_N=4, T_R=8 → 70; 75; 80 với w_N=0;0.5;1
    out.append(_ok("ER31_beei_wN0", beei_score(28, 8, 4, 1, 0, 0) == 70.0, beei_score(28, 8, 4, 1, 0, 0)))
    out.append(_ok("ER32_beei_wN05", beei_score(28, 8, 4, 1, 0, 0.5) == 75.0, beei_score(28, 8, 4, 1, 0, 0.5)))
    out.append(_ok("ER33_beei_wN1", beei_score(28, 8, 4, 1, 0, 1) == 80.0, beei_score(28, 8, 4, 1, 0, 1)))

    # B1 dưới ngưỡng 1s không đủ để thành E
    short_b1 = label_window(
        video_id="VID", student_id="S01", start_sec=0, end_sec=10, activity_code="A1",
        behavior_events=[BehaviorInterval("S01", "Giơ tay", 0, 0.5)],
        observation_spans=[(0.0, 10.0)],
        cfg=cfg,
    )
    out.append(_ok(
        "ER34_B1_short_N",
        short_b1.pred_label == "N" and "B1" not in short_b1.behavior_events,
        f"{short_b1.pred_label} {short_b1.behavior_events}",
    ))

    # analyze_session mặc định A1, có observation → N nếu không có E/R
    windows, metrics, class_m = analyze_session(
        video_id="VID01",
        duration_sec=20.0,
        student_ids=["S01"],
        behaviors=[],
        observations=[ObservationInterval("S01", 0.0, 20.0)],
        activity_segments=None,
        cfg=cfg,
    )
    out.append(_ok("ER23_default_A1", all(w.activity_code == "A1" for w in windows),
                   str([w.activity_code for w in windows])))
    out.append(_ok(
        "ER24_sitting_N",
        all(w.pred_label == "N" for w in windows),
        str([w.pred_label for w in windows]),
    ))
    out.append(_ok("ER25_class_enough", class_m["data_status"] == "Đủ dữ liệu", class_m["data_status"]))

    # Tách cửa sổ tại mốc đổi A
    windows_a, _, _ = analyze_session(
        video_id="VID02",
        duration_sec=30.0,
        student_ids=["S01"],
        behaviors=[BehaviorInterval("S01", "Viết bài", 10, 25)],
        observations=[ObservationInterval("S01", 0.0, 30.0)],
        activity_segments=[
            {"start_sec": 0, "end_sec": 15, "activity_code": "A1"},
            {"start_sec": 15, "end_sec": 30, "activity_code": "A2"},
        ],
        cfg=cfg,
    )
    acts = [(round(w.start_sec), round(w.end_sec), w.activity_code, w.pred_label) for w in windows_a]
    out.append(_ok(
        "ER35_split_at_A",
        any(a[0] == 10 and a[1] == 15 and a[2] == "A1" for a in acts)
        and any(a[2] == "A2" and a[3] == "E" for a in acts),
        str(acts),
    ))

    # Nhiệm vụ chưa xác nhận: B1 không thành E
    unc = label_window(
        video_id="VID", student_id="S01", start_sec=0, end_sec=10,
        activity_code="CHUA_XAC_MINH",
        behavior_events=[BehaviorInterval("S01", "Giơ tay", 0, 3)],
        observation_spans=[(0.0, 10.0)],
        cfg=cfg,
    )
    out.append(_ok(
        "ER36_unconfirmed_no_E",
        unc.pred_label == "N" and unc.activity_code == "CHUA_XAC_MINH",
        f"{unc.pred_label} act={unc.activity_code}",
    ))

    # Proxy tự động: E→ON, R→OFF; hết phiên focus_verdict
    from core.ernu import compute_focus_verdict, proxy_from_label
    from core.config import ErnuConfig as _EC, DEFAULT_ERNU_RULES

    out.append(_ok("ER37_proxy_E_ON", proxy_from_label("E") == "ON", proxy_from_label("E")))
    out.append(_ok("ER38_proxy_R_OFF_auto", proxy_from_label("R") == "OFF", proxy_from_label("R")))
    out.append(_ok("ER40_low_proxy_NO_DATA", low.proxy_label == "NO_DATA", low.proxy_label))
    out.append(_ok("ER41_conflict_proxy_ON", conflict.proxy_label == "ON", conflict.proxy_label))

    rw = WindowLabel("V", "S01", 0, 10, "A1", ["B5"], 1.0, "R", "review", False, "ernu-v1")
    out.append(_ok("ER42_R_default_OFF", rw.proxy_label == "OFF", rw.proxy_label))

    out.append(_ok("ER44_ex5_TON", abs(m.T_ON - 180) < 1e-6, f"T_ON={m.T_ON}"))
    out.append(_ok("ER45_ex5_TOFF", abs(m.T_OFF - 50) < 1e-6, f"T_OFF={m.T_OFF}"))
    out.append(_ok("ER46_ex5_TNO_DATA", abs(m.T_NO_DATA - 30) < 1e-6, f"T_NO_DATA={m.T_NO_DATA}"))
    out.append(_ok(
        "ER47_ex5_verdict_tap_trung",
        m.focus_verdict == "Tập trung" and m.focus_score == 74.1,
        f"{m.focus_verdict} score={m.focus_score}",
    ))

    # Load rules JSON (R→OFF, B2→E mọi A)
    loaded = _EC.load_rules(DEFAULT_ERNU_RULES)
    out.append(_ok(
        "ER48_load_rules_json",
        loaded.rules_version == "ernu-v1" and loaded.window_sec == 10.0 and loaded.c_min == 0.70,
        f"ver={loaded.rules_version} win={loaded.window_sec} c={loaded.c_min}",
    ))
    out.append(_ok(
        "ER49_load_matrix_B2_A1",
        (loaded.matrix.get("B2") or {}).get("A1") == "E",
        str(loaded.matrix.get("B2")),
    ))
    out.append(_ok(
        "ER50_load_proxy_mapping",
        (loaded.proxy_mapping or {}).get("R") == "OFF",
        str(loaded.proxy_mapping),
    ))

    # Session: gục bàn → OFF → Không tập trung
    _, metrics_c, class_c = analyze_session(
        video_id="VID_R",
        duration_sec=10.0,
        student_ids=["S01"],
        behaviors=[BehaviorInterval("S01", "Gục xuống bàn", 0, 8)],
        observations=[ObservationInterval("S01", 0.0, 10.0)],
        cfg=cfg,
    )
    out.append(_ok(
        "ER51_session_auto_OFF",
        metrics_c[0].T_OFF > 0
        and metrics_c[0].focus_verdict == "Không tập trung"
        and any(w.proxy_label == "OFF" for w in metrics_c[0].windows),
        f"verdict={metrics_c[0].focus_verdict} T_OFF={metrics_c[0].T_OFF}",
    ))
    out.append(_ok(
        "ER52_class_counts",
        class_c.get("n_khong_tap_trung", 0) >= 1,
        str({k: class_c.get(k) for k in ("n_tap_trung", "n_khong_tap_trung", "n_chua_du_du_lieu")}),
    ))
    v, sc = compute_focus_verdict(
        publishable=True, T_valid=100, T_ON=80, T_OFF=20, BEEI=80.0, focus_threshold=50,
    )
    out.append(_ok("ER53_verdict_fn", v == "Tập trung" and sc == 80.0, f"{v} {sc}"))
    return out


def run_all() -> dict:
    suites = [
        ("arcface_margin", test_arcface_margin),
        ("identity_iou", test_identity_iou),
        ("behavior_log_E01_E06", test_behavior_log_e01_e06),
        ("timing", test_timing),
        ("missing_keypoint", test_missing_keypoint_constants),
        ("ernu", test_ernu_matrix_and_metrics),
    ]
    results = []
    for suite_name, fn in suites:
        try:
            cases = fn()
        except Exception as exc:  # noqa: BLE001
            cases = [_ok(f"{suite_name}_CRASH", False, traceback.format_exc())]
            print(f"CRASH {suite_name}: {exc}")
        for c in cases:
            c["suite"] = suite_name
            status = "PASS" if c["pass"] else "FAIL"
            print(f"  [{status}] {c['name']}: {c['detail']}")
            results.append(c)
    passed = sum(1 for c in results if c["pass"])
    failed = len(results) - passed
    report = {
        "created_at": datetime.now().isoformat(),
        "passed": passed,
        "failed": failed,
        "total": len(results),
        "present_min_frames": PRESENT_MIN_FRAMES,
        "present_min_face_matches": PRESENT_MIN_FACE_MATCHES,
        "attendance_mode_default": ATTENDANCE_MODE,
        "iou_transfer_lock": IOU_TRANSFER_LOCK,
        "cases": results,
    }
    print(f"\n→ {passed}/{len(results)} passed, {failed} failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=BASE_DIR / "evidence" / "tests" / "stage1_unit_results.json",
    )
    args = parser.parse_args()
    report = run_all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {args.out}")
    raise SystemExit(0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
