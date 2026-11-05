"""
Web app theo dõi lớp học (giống luồng classroom_attention):
  nguồn video / camera RTSP → server đọc khung → YOLOv12 (track người)
  → YOLO11-pose (keypoint) → nhận biết hành vi → ArcFace gắn tên
  → vẽ box + nhãn → đẩy JPEG + thống kê qua WebSocket.

Chạy: python app.py   →   http://127.0.0.1:5000
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from ultralytics import YOLO

from core.analysis import FrameAnalyzer, effective_fps, fit_width, work_size
from core.behavior import ALWAYS_ON, REASON_SHORT, behavior_legend
from core.camera_hub import CameraHub
from core.camera_registry import CameraError, source_from_body
from core.behavior_log import BehaviorRecorder
from core.config import HIDDEN_BEHAVIORS, Config, load_config, save_behavior_settings
from core.face_gallery import (
    FaceRecognizer,
    display_code,
    load_student_classes,
    reset_display_codes,
    roster_names,
)
from core.roster_import import import_student_rows, student_template_bytes
from core.roster_store import RosterError, add_class, add_photos, add_student, delete_class, delete_reports, delete_student, first_photo_jpeg, list_classes, list_reports, list_students, overview, update_student
from core.identity import IdentityTracker
from core.keypoint_log import LOG_DIR as KEYPOINT_LOG_DIR
from core.keypoint_log import KeypointLogger
from core.video_source import (
    capture_fps,
    create_camera_capture,
    is_stream_url,
    mask_url,
    parse_camera_source,
)
from core.video_clock import RESYNC_SEC, VideoClock, preload_ocr, read_osd
from core.visualizer import draw_annotations

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
# Video tải lên mà không có phiên nào mở trong khoảng này thì bị dọn
UPLOAD_ORPHAN_SEC = 3600
REPORTS_DIR = BASE_DIR / "reports"
MAX_KEPT_REPORTS = 10
# Đọc toàn khung tìm ngày giờ tốn ~5s CPU → thử tối đa N lần rồi bỏ
OSD_FULL_TRIES = 3
OSD_RETRY_SEC = 10.0
CAMERA_RETRY_SEC = 0.2
CAMERA_REOPEN_EVERY = 10  # ~2 giây lỗi liên tiếp → mở lại camera
CAMERA_MAX_FAILS = 150  # ~30 giây không có khung → báo mất tín hiệu

ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mts", ".m2ts"}
MAX_UPLOAD_BYTES = 2 * 1024 ** 3
UPLOAD_CHUNK = 1024 * 1024

DISPLAY_MAX_FPS = 15.0
SUMMARY_INTERVAL_SEC = 1.0
JPEG_QUALITY = 80
# ArcFace nghỉ = tỉ lệ × thời gian vòng vừa chạy, nhường CPU cho YOLO/pose
ARCFACE_IDLE_RATIO = 0.5

_camera_hub = CameraHub()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _camera_hub.start_all()
    yield
    _camera_hub.stop_all()


app = FastAPI(title="YOLOv12 + ArcFace — theo dõi lớp học", lifespan=_lifespan)

_config: Config = load_config()
_recognizer: FaceRecognizer | None = None
_recognizer_lock = threading.Lock()
_reports: dict[str, BehaviorRecorder] = {}
_reports_lock = threading.Lock()
_uploads_in_use: list[Path] = []  # video tạm đang được phiên phân tích đọc (có thể lặp nếu mở 2 tab)
_uploads_lock = threading.Lock()
_sessions: set["_StreamSession"] = set()
_sessions_lock = threading.Lock()


def get_recognizer() -> FaceRecognizer:
    global _recognizer
    with _recognizer_lock:
        if _recognizer is None:
            print("Đang nạp gallery ArcFace...")
            _recognizer = FaceRecognizer(force_rebuild=False)
            print(f"Gallery: {len(_recognizer.ids)} học sinh")
        return _recognizer


def _read_template(name: str) -> HTMLResponse:
    path = TEMPLATES_DIR / name
    if not path.exists():
        return HTMLResponse(f"<h1>Không thấy {name}</h1>", status_code=404)
    return HTMLResponse(path.read_text(encoding="utf-8"))


def _render_frame(frame: np.ndarray, detections: list[dict]) -> bytes:
    canvas = frame.copy()
    draw_annotations(canvas, detections, _config.show_labels)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


class _StreamSession:
    """Giữ video phát mượt trong khi phân tích (chậm hơn nhiều) chạy song song.

    - Reader: giải mã đúng tốc độ video, chỉ giữ khung mới nhất.
    - Analyzer: YOLOv12 track + YOLO11-pose + nhận biết hành vi trên khung mới nhất.
    - ArcFace: nhận diện tên trên (khung, track) của lần phân tích gần nhất.
    Mỗi khung hiển thị được vẽ đè kết quả phân tích gần nhất.
    """

    def __init__(self, cap, fps: float, is_camera: bool, recognizer: FaceRecognizer,
                 source_label: str, reopen=None):
        self.cap = cap
        self.fps = fps
        self.is_camera = is_camera
        self.reopen = reopen
        self.cfg = _config
        self.source_label = source_label
        self._student_names = roster_names(recognizer.ids, recognizer.names)
        classes = load_student_classes()
        known_classes = [c for c in classes.values() if c]
        self._student_classes = classes
        self._default_class = (
            max(set(known_classes), key=known_classes.count) if known_classes else ""
        )
        self.session_id = uuid.uuid4().hex[:10]
        self.clock = VideoClock(fps, is_camera)
        self.recorder = self._make_recorder()
        self.identity = IdentityTracker(recognizer)
        self.analyzer = FrameAnalyzer(self.cfg, self.identity)
        self._frame_wh = work_size(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.kp_log = self._open_keypoint_log()
        self.recording = True
        self._work = threading.Lock()
        self._gen = 0

        self.stop = threading.Event()
        self.eof = threading.Event()
        self.error: str | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_idx = -1
        self._detections: list[dict] = []
        self._analysis_idx = -1
        self._analysis_ms = 0
        self._analysis_version = 0
        self._face_job: tuple[int, np.ndarray, list] | None = None
        self._threads = [
            threading.Thread(target=self._read_loop, daemon=True, name="StreamReader"),
            threading.Thread(target=self._analyze_loop, daemon=True, name="StreamAnalyzer"),
            threading.Thread(target=self._face_loop, daemon=True, name="ArcFaceWorker"),
            threading.Thread(target=self._clock_loop, daemon=True, name="OsdClock"),
        ]

    def _make_recorder(self) -> BehaviorRecorder:
        return BehaviorRecorder(
            self.fps, self.clock, self.source_label,
            self._student_names,
            legend_fn=lambda: behavior_legend(_config),
            student_classes=self._student_classes,
            default_class=self._default_class,
            ernu_cfg=_config.ernu,
        )

    def _open_keypoint_log(self) -> KeypointLogger | None:
        if not self.cfg.record_keypoints:
            return None
        stem = f"{datetime.now():%Y%m%d_%H%M%S}_{self.session_id}"
        w, h = self._frame_wh
        return KeypointLogger(
            KEYPOINT_LOG_DIR / f"{stem}.jsonl", self.source_label, self.fps, w, h, self.cfg.behavior,
        )

    def end_session(self) -> tuple[str, BehaviorRecorder] | None:
        """Đóng sổ phiên đang ghi. Video hoặc camera chạy tiếp, chưa mở phiên mới."""
        with self._work:
            if not self.recording:
                return None
            self.recording = False
            self._gen += 1
            old_id = self.session_id
            old_rec = self.recorder
            old_log = self.kp_log
            self.kp_log = None
            if old_log is not None:
                old_log.close()
        return old_id, old_rec

    def start_session(self) -> str | None:
        """Mở phiên mới trên cùng nguồn sau khoảng giao thoa."""
        with self._work:
            if self.recording:
                return None
            self._gen += 1
            self.session_id = uuid.uuid4().hex[:10]
            self.recorder = self._make_recorder()
            self.identity.reset()
            self.analyzer.reset_tracks()
            with self._lock:
                self._detections = []
                self._face_job = None
                self._analysis_version += 1
            self.kp_log = self._open_keypoint_log()
            self.recording = True
        return self.session_id

    def start(self) -> None:
        for t in self._threads:
            t.start()

    def close(self) -> None:
        self.stop.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=10)
        self.cap.release()
        self.analyzer.close()
        if self.kp_log is not None:
            self.kp_log.close()

    def snapshot(self):
        with self._lock:
            return (
                self._frame,
                self._frame_idx,
                self._detections,
                self._analysis_idx,
                self._analysis_version,
                self._analysis_ms,
            )

    def _read_loop(self) -> None:
        interval = 1.0 / self.fps
        start = time.monotonic()
        idx = 0
        fails = 0
        try:
            while not self.stop.is_set():
                try:
                    ok, frame = self.cap.read()
                except cv2.error:
                    ok, frame = False, None
                if not ok or frame is None:
                    if not self.is_camera:
                        break
                    # Camera ảo (OBS) / RTSP hay lỗi đọc tạm thời → thử lại, định kỳ mở lại camera
                    fails += 1
                    if fails >= CAMERA_MAX_FAILS:
                        self.error = ("Mất tín hiệu camera. Nếu dùng OBS: kiểm tra Virtual Camera "
                                      "còn bật và cửa sổ nguồn không bị thu nhỏ.")
                        break
                    if self.reopen is not None and fails % CAMERA_REOPEN_EVERY == 0:
                        try:
                            self.cap.release()
                            self.cap = self.reopen()
                        except Exception:  # noqa: BLE001
                            pass
                    self.stop.wait(CAMERA_RETRY_SEC)
                    continue
                fails = 0
                frame = fit_width(frame)
                with self._lock:
                    self._frame, self._frame_idx = frame, idx
                idx += 1
                if self.is_camera:
                    continue
                delay = start + idx * interval - time.monotonic()
                if delay > 0:
                    self.stop.wait(delay)
                    continue
                # Chậm hơn lịch: bỏ khung (grab không giải mã màu) để giữ tốc độ thật
                while delay < -interval and not self.stop.is_set() and self.cap.grab():
                    idx += 1
                    delay = start + idx * interval - time.monotonic()
        finally:
            self.eof.set()

    def _analyze_loop(self) -> None:
        last_idx = -1
        while not self.stop.is_set():
            with self._lock:
                frame, idx = self._frame, self._frame_idx
            if frame is None or idx == last_idx:
                if self.eof.is_set():
                    return
                self.stop.wait(0.01)
                continue
            frames_elapsed = idx - last_idx if last_idx >= 0 else 1
            fps_eff = effective_fps(self.fps, frames_elapsed)
            t0 = time.perf_counter()
            try:
                with self._work:
                    if self.stop.is_set():
                        return
                    gen = self._gen
                    detections, parsed = self.analyzer.analyze(frame, fps_eff)
                    if gen != self._gen:
                        continue
                    if self.recording:
                        self.recorder.record(idx, detections, time.time(), frame)
                        if self.kp_log is not None:
                            self.kp_log.log(idx, fps_eff, detections)
            except Exception as exc:  # noqa: BLE001
                self.error = f"Lỗi phân tích: {exc}"
                self.stop.set()
                return
            last_idx = idx
            with self._lock:
                self._detections = detections
                self._analysis_idx = idx
                self._analysis_ms = round((time.perf_counter() - t0) * 1000)
                self._analysis_version += 1
                self._face_job = (idx, frame, parsed)

    def _face_loop(self) -> None:
        """ArcFace chạy riêng để tên không làm chậm YOLO/pose."""
        last_idx = -1
        while not self.stop.is_set():
            with self._lock:
                job = self._face_job
            if job is None or job[0] == last_idx:
                if self.eof.is_set() and (job is None or job[0] == last_idx):
                    return
                self.stop.wait(0.02)
                continue
            idx, frame, parsed = job
            t0 = time.perf_counter()
            epoch = self.identity.epoch
            try:
                self.identity.recognize(frame, parsed)
            except Exception as exc:  # noqa: BLE001
                self.error = f"Lỗi ArcFace: {exc}"
                self.stop.set()
                return
            took = time.perf_counter() - t0
            last_idx = idx
            if self.identity.epoch != epoch:
                self.stop.wait(max(0.01, took * ARCFACE_IDLE_RATIO))
                continue
            with self._lock:
                for det in self._detections:
                    info = self.identity.describe(det["track_id"])
                    det.update(
                        known=info["known"],
                        student_id=info["student_id"],
                        name=info["name"],
                        match_pct=info["match_pct"],
                        hard_locked=info["hard_locked"],
                    )
                self._analysis_version += 1
            self.stop.wait(max(0.01, took * ARCFACE_IDLE_RATIO))

    def _clock_loop(self) -> None:
        """Đọc ngày giờ in trên video: quét toàn khung tìm vị trí, rồi đọc lại vùng đó định kỳ."""
        next_idx = 0
        full_tries = 0
        while not self.stop.is_set():
            with self._lock:
                frame, idx = self._frame, self._frame_idx
            if frame is None or idx < next_idx:
                if self.eof.is_set():
                    return
                self.stop.wait(0.5)
                continue
            region = self.clock.region
            if region is None:
                if full_tries >= OSD_FULL_TRIES:
                    return
                full_tries += 1
            try:
                reading = read_osd(frame, region)
            except Exception as exc:  # noqa: BLE001
                print(f"Không đọc được ngày giờ trên video (OCR): {exc}")
                return
            if reading is not None:
                self.clock.add_reading(idx, reading, frame.shape)
                next_idx = idx + int(RESYNC_SEC * self.fps)
            else:
                next_idx = idx + int(OSD_RETRY_SEC * self.fps)


# ---------------------------------------------------------------- trang chính


@app.get("/")
async def index():
    return _read_template("index.html")


@app.get("/admin")
async def admin_page():
    return _read_template("admin.html")


def _roster_error(exc: RosterError) -> HTTPException:
    return HTTPException(400, str(exc))


async def _photo_blobs(files: list[UploadFile] | None) -> list[tuple[str, bytes]]:
    blobs: list[tuple[str, bytes]] = []
    for upload in files or []:
        if not upload.filename:
            continue
        blobs.append((upload.filename, await upload.read()))
        await upload.close()
    return blobs


@app.get("/api/admin/overview")
async def admin_overview():
    return await asyncio.to_thread(overview, REPORTS_DIR)


@app.get("/api/admin/classes")
async def admin_classes():
    return {"classes": await asyncio.to_thread(list_classes)}


@app.post("/api/admin/classes")
async def admin_add_class(request: Request):
    body = await request.json()
    try:
        row = await asyncio.to_thread(add_class, str(body.get("class_id") or ""), str(body.get("note") or ""))
    except RosterError as exc:
        raise _roster_error(exc) from exc
    return row


@app.delete("/api/admin/classes/{class_id}")
async def admin_delete_class(class_id: str):
    try:
        await asyncio.to_thread(delete_class, class_id)
    except RosterError as exc:
        raise _roster_error(exc) from exc
    return {"ok": True}


@app.get("/api/admin/students")
async def admin_students(class_id: str = ""):
    return {"students": await asyncio.to_thread(list_students, class_id)}


@app.get("/api/admin/students/template")
async def admin_student_template(class_id: str = ""):
    try:
        data, filename = await asyncio.to_thread(student_template_bytes, class_id)
    except RosterError as exc:
        raise _roster_error(exc) from exc
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/admin/classes/{class_id}/students/import")
async def admin_import_students(class_id: str, file: UploadFile = File(...)):
    raw = await file.read()
    await file.close()
    try:
        result = await asyncio.to_thread(import_student_rows, class_id, raw)
    except RosterError as exc:
        raise _roster_error(exc) from exc
    reset_display_codes()
    return result


@app.post("/api/admin/students")
async def admin_add_student(
    class_name: str = Form(...),
    full_name: str = Form(...),
    gioi_tinh: str = Form(""),
    ngay_sinh: str = Form(""),
    student_id: str = Form(""),
    files: list[UploadFile] | None = File(None),
):
    photos = await _photo_blobs(files)
    try:
        row = await asyncio.to_thread(
            add_student,
            class_name=class_name,
            full_name=full_name,
            gioi_tinh=gioi_tinh,
            ngay_sinh=ngay_sinh,
            student_id=student_id,
            photos=photos,
        )
    except RosterError as exc:
        raise _roster_error(exc) from exc
    reset_display_codes()
    return row


@app.patch("/api/admin/students/{student_id}")
async def admin_update_student(student_id: str, request: Request):
    body = await request.json()
    try:
        row = await asyncio.to_thread(
            update_student,
            student_id,
            full_name=body.get("full_name"),
            class_name=body.get("class_name"),
            gioi_tinh=body.get("gioi_tinh"),
            ngay_sinh=body.get("ngay_sinh"),
        )
    except RosterError as exc:
        raise _roster_error(exc) from exc
    reset_display_codes()
    return row


@app.post("/api/admin/students/{student_id}/photos")
async def admin_add_photos(student_id: str, files: list[UploadFile] | None = File(None)):
    photos = await _photo_blobs(files)
    try:
        row = await asyncio.to_thread(add_photos, student_id, photos)
    except RosterError as exc:
        raise _roster_error(exc) from exc
    return row


@app.delete("/api/admin/students/{student_id}")
async def admin_delete_student(student_id: str):
    try:
        await asyncio.to_thread(delete_student, student_id)
    except RosterError as exc:
        raise _roster_error(exc) from exc
    reset_display_codes()
    return {"ok": True}


_photo_slots = asyncio.Semaphore(4)


@app.get("/api/admin/students/{student_id}/photo")
async def admin_student_photo(student_id: str):
    async with _photo_slots:
        try:
            jpeg = await asyncio.to_thread(first_photo_jpeg, student_id)
        except Exception as exc:
            from pymongo.errors import PyMongoError
            if isinstance(exc, PyMongoError):
                raise HTTPException(503, "Chưa kết nối được cơ sở dữ liệu") from exc
            raise
    if jpeg is None:
        raise HTTPException(404, "Chưa có ảnh")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/admin/reports")
async def admin_reports(page: int = 1, page_size: int = 8):
    return await asyncio.to_thread(list_reports, page, page_size, REPORTS_DIR)


@app.delete("/api/admin/reports")
async def admin_delete_reports(request: Request):
    body = await request.json()
    names = body.get("names") if isinstance(body, dict) else None
    if not isinstance(names, list):
        raise HTTPException(400, "Cần danh sách báo cáo.")
    try:
        deleted = await asyncio.to_thread(delete_reports, [str(item) for item in names], REPORTS_DIR)
    except RosterError as exc:
        raise _roster_error(exc) from exc
    return {"ok": True, "deleted": deleted}


@app.get("/api/admin/reports/{name}/preview")
async def admin_report_preview(name: str):
    if not re.fullmatch(r"bao_cao_hanh_vi_[0-9_]+_[0-9a-f]+\.xlsx", name):
        raise HTTPException(404, "Không thấy báo cáo")
    from core.mongo_roster import preview_report
    try:
        return await asyncio.to_thread(preview_report, name)
    except RosterError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/admin/reports/{name}")
async def admin_report_file(name: str):
    if not re.fullmatch(r"bao_cao_hanh_vi_[0-9_]+_[0-9a-f]+\.xlsx", name):
        raise HTTPException(404, "Không thấy báo cáo")
    data = _read_report(name)
    if data is None:
        raise HTTPException(404, "Không thấy báo cáo")
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.post("/api/admin/rebuild-gallery")
async def admin_rebuild_gallery():
    global _recognizer
    try:
        rec = await asyncio.to_thread(FaceRecognizer, True)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    reset_display_codes()
    with _recognizer_lock:
        _recognizer = rec
    return {"ok": True, "students": len(rec.ids)}


@app.get("/api/legend")
@app.get("/api/behaviors")
async def legend():
    """Chú thích nhãn viết tắt + trạng thái bật/tắt từng hành vi."""
    return {
        "behaviors": behavior_legend(_config),
    }


@app.post("/api/behaviors")
async def set_behaviors(request: Request):
    """Bật/tắt hành vi: {"enabled": {"Cúi đầu lâu": false, ...}} hoặc {"all": true|false}.

    Áp dụng ngay cho video đang chạy và lưu vào behavior_settings.json.
    """
    body = await request.json()
    names = [name for name, _short in REASON_SHORT
             if name not in HIDDEN_BEHAVIORS and name not in ALWAYS_ON]
    disabled = set(_config.behavior.disabled)
    if isinstance(body.get("all"), bool):
        disabled = set() if body["all"] else set(names)
    for name, on in (body.get("enabled") or {}).items():
        if name not in names:
            raise HTTPException(400, f"Không có hành vi: {name}")
        (disabled.discard if on else disabled.add)(name)
    _config.behavior.disabled = frozenset(disabled) | HIDDEN_BEHAVIORS
    save_behavior_settings(_config)
    return {
        "behaviors": behavior_legend(_config),
    }


@app.get("/api/ernu")
def ernu_config():
    """Cấu hình diễn giải E/R/N/U + proxy + đường dẫn ACTIVITY_CONTEXT."""
    e = _config.ernu
    return {
        "rules_version": e.rules_version,
        "rules_path": getattr(e, "rules_path", "") or "",
        "window_sec": e.window_sec,
        "c_min": e.c_min,
        "publish_c_min": e.publish_c_min,
        "default_activity": e.default_activity,
        "activity_context_path": e.activity_context_path,
        "min_duration": {
            "B1": e.min_duration_b1,
            "B2": e.min_duration_b2,
            "B3": e.min_duration_b3,
            "B4": e.min_duration_b4,
            "B5": e.min_duration_b5,
        },
        "beei_weights": {"w_E": e.beei_w_e, "w_R": e.beei_w_r, "w_N": e.beei_w_n},
        "proxy_mapping": getattr(e, "proxy_mapping", None) or {
            "E": "ON", "R": "PENDING", "N": "UNCERTAIN", "U": "NO_DATA",
        },
        "activities": {
            "A1": "Nghe giảng / quan sát bảng",
            "A2": "Ghi chép / làm bài cá nhân",
            "A3": "Đọc sách / tài liệu",
            "A4": "Hỏi đáp / phát biểu",
            "A5": "Thảo luận nhóm / theo cặp",
            "CHUA_XAC_MINH": "Chưa xác nhận nhiệm vụ",
        },
        "disclaimer": (
            "E/R/N/U không phải điểm tập trung nhận thức. "
            "Proxy ON/PENDING/UNCERTAIN/NO_DATA diễn giải theo nhiệm vụ; "
            "OFF chỉ sau xác nhận xem lại. BEEI chỉ là chỉ số phụ. Không xếp hạng học sinh."
        ),
    }


@app.post("/api/ernu/activity-context")
async def set_activity_context(request: Request):
    """Gán đường dẫn ACTIVITY_CONTEXT (.xlsx/.json) hoặc segments JSON.

    Body: {"path": "..."} hoặc {"segments": [{"start_sec", "end_sec", "activity_code", ...}]}.
    activity_code ∈ A1–A5 hoặc CHUA_XAC_MINH.
    Áp dụng cho phiên đang ghi (recorder) và cấu hình toàn cục.
    """
    from core.activity_context import load_activity_context
    from core.ernu import UNCONFIRMED_ACTIVITY, VALID_ACTIVITIES

    body = await request.json()
    path = (body.get("path") or "").strip()
    segments = body.get("segments")
    allowed = set(VALID_ACTIVITIES) | {UNCONFIRMED_ACTIVITY, "UNKNOWN", "UNCONFIRMED", "?"}
    if path:
        _config.ernu.activity_context_path = path
        segs = load_activity_context(path)
    elif isinstance(segments, list):
        segs = []
        for row in segments:
            try:
                s, e = float(row["start_sec"]), float(row["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if e <= s:
                continue
            code = str(row.get("activity_code") or "").strip().upper()
            if code not in allowed:
                continue
            if code in ("UNKNOWN", "UNCONFIRMED", "?"):
                code = UNCONFIRMED_ACTIVITY
            segs.append({
                "start_sec": s, "end_sec": e, "activity_code": code,
                "teacher_task": str(row.get("teacher_task") or row.get("task_note") or ""),
                "annotator_id": str(row.get("annotator_id") or ""),
                "notes": str(row.get("notes") or ""),
            })
    else:
        raise HTTPException(400, "Cần path hoặc segments")

    applied = 0
    with _sessions_lock:
        sessions = list(_sessions)
    for s in sessions:
        s.recorder.activity_segments = list(segs)
        applied += 1
    return {
        "ok": True,
        "n_segments": len(segs),
        "segments": segs,
        "sessions_updated": applied,
        "activity_context_path": _config.ernu.activity_context_path,
        "default_activity": _config.ernu.default_activity,
    }


@app.post("/api/ernu/review-confirm")
async def ernu_review_confirm(request: Request):
    """Xác nhận cửa sổ R → OFF (hoặc giữ PENDING nếu confirmed_off=false).

    Body: {student_id, start_sec, end_sec, confirmed_off?: true}
    Không đổi nhãn E/R/N/U của máy.
    """
    body = await request.json()
    sid = str(body.get("student_id") or "").strip()
    try:
        start_sec = float(body["start_sec"])
        end_sec = float(body["end_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cần student_id, start_sec, end_sec: {exc}") from exc
    if not sid or end_sec <= start_sec:
        raise HTTPException(400, "student_id rỗng hoặc khoảng thời gian không hợp lệ")
    confirmed_off = bool(body.get("confirmed_off", True))

    updated = []
    with _sessions_lock:
        sessions = list(_sessions)
    if not sessions:
        raise HTTPException(404, "Không có phiên đang chạy")
    for s in sessions:
        row = s.recorder.add_review_confirmation(sid, start_sec, end_sec, confirmed_off)
        updated.append(row)
    return {
        "ok": True,
        "confirmed_off": confirmed_off,
        "proxy_if_r": "OFF" if confirmed_off else "PENDING",
        "sessions_updated": len(updated),
        "confirmation": updated[0] if updated else None,
        "note": "Chỉ áp dụng khi cửa sổ khớp có pred_label=R; không tự gán OFF từ máy.",
    }


@app.get("/api/arcface")
def arcface_thresholds():
    """Ngưỡng ArcFace đang dùng (%)."""
    return get_recognizer().thresholds()


@app.post("/api/arcface")
async def set_arcface_thresholds(request: Request):
    """Đổi ngưỡng: {"match_pct", "hard_lock_pct", "hard_lock_margin_pct"}.

    Áp dụng ngay cho video đang chạy (bỏ tên các track khớp dưới ngưỡng mới) và lưu vào meta.json.
    """
    body = await request.json()
    rec = await asyncio.to_thread(get_recognizer)
    cur = rec.thresholds()
    try:
        values = {k: float(body.get(k, cur[k])) for k in ("match_pct", "hard_lock_pct", "hard_lock_margin_pct")}
        await asyncio.to_thread(rec.set_thresholds, **values)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    with _sessions_lock:
        sessions = list(_sessions)
    dropped = sum(
        s.identity.drop_below(rec.match_threshold, rec.hard_lock_sim)
        for s in sessions if s.identity.rec is rec
    )
    return {**rec.thresholds(), "dropped": dropped}


@app.get("/api/sources")
def camera_sources():
    """Camera đã lưu. Link trả về đã che mật khẩu."""
    return {"cameras": _camera_hub.list_public()}


@app.post("/api/sources")
async def add_camera_source(request: Request):
    body = await request.json()
    try:
        name, source = source_from_body(body)
        row = await asyncio.to_thread(_camera_hub.add, name, source)
    except CameraError as exc:
        raise HTTPException(400, str(exc)) from exc
    return row


@app.delete("/api/sources/{camera_id}")
async def delete_camera_source(camera_id: str):
    try:
        await asyncio.to_thread(_camera_hub.remove, camera_id)
    except CameraError as exc:
        status = 409 if "theo dõi" in str(exc) else 404
        raise HTTPException(status, str(exc)) from exc
    return {"ok": True}


@app.post("/api/sources/{camera_id}/retry")
async def retry_camera_source(camera_id: str):
    try:
        row = await asyncio.to_thread(_camera_hub.restart, camera_id)
    except CameraError as exc:
        status = 409 if "theo dõi" in str(exc) else 404
        raise HTTPException(status, str(exc)) from exc
    return row


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Lưu tạm video vào static/uploads/ để mở qua /ws — file bị xóa khi phiên phân tích kết thúc."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(400, f"Định dạng không hỗ trợ: {ext or '(không có)'}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_purge_uploads, UPLOAD_ORPHAN_SEC)
    stem = re.sub(r"[^\w.-]+", "_", Path(file.filename or "video").stem)[:60] or "video"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}_{stem}{ext}"
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File quá lớn (tối đa 2 GB)")
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    cap = cv2.VideoCapture(str(dest))
    readable = cap.isOpened() and cap.read()[0]
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    if not readable:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "File không phải video đọc được")

    return JSONResponse(
        {"path": str(dest), "filename": file.filename, "size": size, "frames": frames, "fps": fps}
    )


def _is_upload(path) -> bool:
    try:
        return Path(path).resolve().parent == UPLOAD_DIR.resolve()
    except (OSError, TypeError):
        return False


def _upload_label(path) -> str:
    """'a1b2c3d4e5f6_Tiet3.mp4' → 'Tiet3.mp4' (tên file gốc lúc tải lên)."""
    name = Path(path).name
    return name[13:] if _is_upload(path) and len(name) > 13 and name[12] == "_" else name


def _purge_uploads(min_age_sec: float = 0.0) -> None:
    """Xóa video tạm không còn phiên nào dùng (tải lên rồi không kết nối, hoặc sót lại khi server tắt)."""
    if not UPLOAD_DIR.is_dir():
        return
    now = time.time()
    with _uploads_lock:
        in_use = set(_uploads_in_use)
    for p in UPLOAD_DIR.iterdir():
        try:
            if p.is_file() and p.resolve() not in in_use and now - p.stat().st_mtime >= min_age_sec:
                p.unlink()
        except OSError:
            pass


@app.websocket("/ws/previews")
async def preview_socket(ws: WebSocket):
    """Ảnh xem trước của mọi camera đã lưu. Không gửi link hay mật khẩu."""
    await ws.accept()
    seen: dict[str, int] = {}
    try:
        next_status = 0.0
        while True:
            now = time.monotonic()
            if now >= next_status:
                await ws.send_json({"type": "previews", "cameras": _camera_hub.list_public()})
                next_status = now + 1.0
            for camera_id, seq, jpeg in _camera_hub.frames_since(seen):
                seen[camera_id] = seq
                header = camera_id.encode("ascii")
                await ws.send_bytes(b"P" + len(header).to_bytes(2, "big") + header + jpeg)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cap = None
    session: _StreamSession | None = None
    upload: Path | None = None
    claimed_id = ""
    held_id = ""
    claimed_name = ""
    listener = None
    try:
        data = await ws.receive_json()
        claimed_id = str(data.get("camera_id") or "").strip()
        if claimed_id:
            try:
                source, claimed_name = await asyncio.to_thread(_camera_hub.claim, claimed_id)
                held_id = claimed_id
            except CameraError as exc:
                await ws.send_json({"error": str(exc)})
                return
        else:
            source = parse_camera_source(data.get("video_source", ""))
            if isinstance(source, int) or source == "":
                await ws.send_json({
                    "error": "Không dùng webcam máy tính. Hãy chọn camera RTSP ở tab Camera hoặc tải video lên.",
                })
                return

        if is_stream_url(source):
            is_camera = True
            await ws.send_json({"status": "connecting", "message": "Đang kết nối camera..."})
            try:
                cap, fps, width, height = await asyncio.to_thread(create_camera_capture, source)
            except ValueError:
                hint = "Kiểm tra địa chỉ IP, mã xác minh và đã bật RTSP chưa."
                await ws.send_json({"error": (
                    f"Không kết nối được camera: {mask_url(source)}. {hint}"
                )})
                return
            total_frames = None
        else:
            is_camera = False
            if not os.path.exists(source):
                await ws.send_json({"error": f"Không tìm thấy video: {source}"})
                return
            if _is_upload(source):
                upload = Path(source).resolve()
                with _uploads_lock:
                    _uploads_in_use.append(upload)
            cap = cv2.VideoCapture(source)
            fps = capture_fps(cap, default=30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        await ws.send_json({
            "status": "loading",
            "message": "Đang tải YOLOv12 + YOLO11-pose + ArcFace...",
            "resolution": [width, height],
            "fps": fps,
            "total_frames": total_frames,
        })
        recognizer = await asyncio.to_thread(get_recognizer)
        source_label = mask_url(source) if is_camera else _upload_label(source)
        reopen = (lambda src=source: create_camera_capture(src)[0]) if is_camera else None
        session = await asyncio.to_thread(
            _StreamSession, cap, fps, is_camera, recognizer, source_label, reopen
        )
        cap = None
        _register_report(session.session_id, session.recorder)

        await ws.send_json({
            "status": "streaming",
            "message": (
                (f"Đang theo dõi {claimed_name} · " if claimed_name else "Đã bắt đầu phát · ")
                + f"YOLO: {Path(_config.yolo_model).name} · "
                f"pose: {Path(_config.pose_model).name} · gallery {len(recognizer.ids)} HS"
            ),
            "gallery_size": len(recognizer.ids),
            "session_id": session.session_id,
            "camera_id": claimed_id or None,
            "report_url": f"/api/report/{session.session_id}",
        })
        session.start()
        with _sessions_lock:
            _sessions.add(session)

        cut_flag = asyncio.Event()
        start_flag = asyncio.Event()
        client_gone = asyncio.Event()

        async def listen_client() -> None:
            try:
                while True:
                    incoming = await ws.receive()
                    if incoming["type"] == "websocket.disconnect":
                        client_gone.set()
                        return
                    text = incoming.get("text")
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    action = payload.get("action")
                    if action == "cut":
                        cut_flag.set()
                    elif action == "start":
                        start_flag.set()
            except Exception:  # noqa: BLE001
                client_gone.set()

        listener = asyncio.create_task(listen_client())

        send_interval = 1.0 / min(fps, DISPLAY_MAX_FPS)
        sent_idx = -1
        sent_version = 0
        next_summary = 0.0

        async def send_summary() -> None:
            summary = await asyncio.to_thread(session.recorder.summary)
            await ws.send_json({"type": "summary", **summary})

        # Lịch tuyệt đối: Windows làm tròn sleep lên ~15.6 ms, sleep cố định sẽ tụt fps
        next_send = time.monotonic()
        while True:
            if client_gone.is_set():
                break
            if cut_flag.is_set():
                cut_flag.clear()
                ended = await asyncio.to_thread(session.end_session)
                if ended is not None:
                    old_id, old_rec = ended
                    name = await asyncio.to_thread(_save_report, old_id)
                    await ws.send_json({
                        "status": "session_cut",
                        "message": (
                            f"Đã ngắt phiên · báo cáo {name} đã lưu trên hệ thống. "
                            "Xem và tải tại trang Quản lý lớp. "
                            "Đoạn giao thoa chưa được ghi. Bấm Bắt đầu phiên khi tiết mới bắt đầu."
                        ),
                        "report_url": f"/api/report/{old_id}",
                        "session_id": old_id,
                    })
            if start_flag.is_set():
                start_flag.clear()
                new_id = await asyncio.to_thread(session.start_session)
                if new_id:
                    _register_report(new_id, session.recorder)
                    await ws.send_json({
                        "status": "session_started",
                        "message": "Đã bắt đầu phiên mới. Khoảng giao thoa vừa rồi không vào báo cáo.",
                        "session_id": new_id,
                        "report_url": f"/api/report/{new_id}",
                    })
                    sent_version = -1
                    next_summary = 0.0
            frame, frame_idx, detections, analysis_idx, version, analysis_ms = session.snapshot()
            if session.error:
                await ws.send_json({"error": session.error})
                break

            if version != sent_version:
                sent_version = version
                await ws.send_json({
                    "type": "stats",
                    "frame_idx": analysis_idx,
                    "analysis_ms": analysis_ms,
                    "identified": session.identity.identified_count(),
                    "hard_locked": session.identity.hard_locked_count(),
                    "students": [
                        {
                            "id": d["track_id"],
                            "student_id": d["student_id"],
                            "code": display_code(d["student_id"]) if d.get("known") else "",
                            "name": d["name"],
                            "known": d["known"],
                            "match_pct": d["match_pct"],
                            "hard_locked": d.get("hard_locked", False),
                            "posed": d.get("posed", False),
                            "group": d.get("group") or "unknown",
                            "reasons": d["reasons"],
                        }
                        for d in detections
                    ],
                })

            if time.monotonic() >= next_summary:
                next_summary = time.monotonic() + SUMMARY_INTERVAL_SEC
                if session.recording:
                    await send_summary()

            if frame is not None and frame_idx != sent_idx:
                jpeg = await asyncio.to_thread(_render_frame, frame, detections)
                await ws.send_bytes(jpeg)
                sent_idx = frame_idx
            elif session.eof.is_set():
                if session.recording:
                    await send_summary()
                    name = await asyncio.to_thread(_save_report, session.session_id)
                    await ws.send_json({
                        "status": "ended",
                        "message": f"Video đã phát xong · báo cáo {name} đã lưu. Xem và tải tại trang Quản lý lớp.",
                        "report_url": f"/api/report/{session.session_id}",
                    })
                else:
                    await ws.send_json({
                        "status": "ended",
                        "message": "Video đã phát xong trong lúc chờ phiên mới. Đoạn giao thoa không được ghi.",
                    })
                break

            next_send = max(next_send + send_interval, time.monotonic() - send_interval)
            await asyncio.sleep(max(0.0, next_send - time.monotonic()))

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_json({"error": mask_url(exc)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        if listener is not None:
            listener.cancel()
        if held_id:
            _camera_hub.release(held_id)
        if session is not None:
            with _sessions_lock:
                _sessions.discard(session)
            await asyncio.to_thread(session.close)
            if session.recording:
                try:
                    name = await asyncio.to_thread(_save_report, session.session_id)
                    print(f"Đã lưu báo cáo hành vi: {name}")
                except Exception as exc:  # noqa: BLE001
                    print(f"Không lưu được báo cáo hành vi: {exc}")
        elif cap is not None:
            cap.release()
        if upload is not None:
            with _uploads_lock:
                _uploads_in_use.remove(upload)
                still_used = upload in _uploads_in_use
            if not still_used:
                try:
                    upload.unlink(missing_ok=True)
                except OSError:
                    pass  # còn bị giữ → _purge_uploads dọn ở lần tải lên / khởi động sau


def _register_report(session_id: str, recorder: BehaviorRecorder) -> None:
    recorder.report_name = f"bao_cao_hanh_vi_{datetime.now():%Y%m%d_%H%M%S}_{session_id}.xlsx"
    with _reports_lock:
        _reports[session_id] = recorder
        while len(_reports) > MAX_KEPT_REPORTS:
            _reports.pop(next(iter(_reports)))


def _write_report(name: str, data: bytes) -> None:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import save_report_bytes
        save_report_bytes(name, data)
        return
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / name).write_bytes(data)


def _read_report(name: str) -> bytes | None:
    from core.db import enabled
    if enabled():
        from core.mongo_roster import get_report_bytes
        return get_report_bytes(name)
    path = (REPORTS_DIR / name).resolve()
    if path.parent != REPORTS_DIR.resolve() or not path.is_file():
        return None
    return path.read_bytes()


def _save_report(session_id: str) -> str:
    with _reports_lock:
        recorder = _reports.get(session_id)
    if recorder is None:
        raise KeyError(session_id)
    _write_report(recorder.report_name, recorder.workbook_bytes())
    return recorder.report_name


@app.get("/api/report/{session_id}")
def download_report(session_id: str):
    """Xuất Excel hành vi của phiên (đang chạy hoặc đã kết thúc)."""
    try:
        name = _save_report(session_id)
    except KeyError:
        raise HTTPException(404, "Không tìm thấy phiên — báo cáo cũ nằm trên MongoDB Atlas hoặc thư mục reports/")
    data = _read_report(name)
    if not data:
        raise HTTPException(404, "Không thấy nội dung báo cáo")
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.post("/rebuild-gallery")
def rebuild_gallery():
    global _recognizer
    try:
        rec = FaceRecognizer(force_rebuild=True)
        with _recognizer_lock:
            _recognizer = rec
        return {"ok": True, "students": len(rec.ids)}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


def main() -> None:
    import argparse

    import torch
    import uvicorn

    parser = argparse.ArgumentParser(description="YOLOv12 + ArcFace — theo dõi lớp học")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 3))
    _purge_uploads()  # video tạm sót lại từ lần chạy trước
    # ultralytics và easyocr import torchvision lười ở 2 luồng khác nhau → lỗi circular import
    import torchvision  # noqa: F401

    print(f"YOLO (track người): {_config.yolo_model}")
    print(f"Pose:               {_config.pose_model}")
    YOLO(_config.yolo_model)
    YOLO(_config.pose_model)
    get_recognizer()
    print("Nạp EasyOCR (đọc ngày giờ trên video)...")
    try:
        preload_ocr()
    except Exception as exc:  # noqa: BLE001
        print(f"Không nạp được EasyOCR — báo cáo sẽ dùng vị trí trong video: {exc}")
    print(f"Mở trình duyệt: http://{args.host}:{args.port}")
    try:
        from lxml import etree
        etree.fromstring(b"<a/>")
    except Exception:
        pass
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
