"""
Web app theo dõi lớp học (giống luồng classroom_attention):
  nguồn video / webcam / RTSP → server đọc khung → YOLOv12 (track người)
  → YOLO11-pose (keypoint) → chấm điểm tập trung → ArcFace gắn tên
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
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from ultralytics import YOLO

from core.analysis import FrameAnalyzer, effective_fps, fit_width, work_size
from core.behavior import REASON_SHORT, behavior_legend
from core.behavior_log import BehaviorRecorder
from core.behavior_monitor import get_monitor, reset_monitor
from core.config import HIDDEN_BEHAVIORS, Config, load_config, save_behavior_settings
from core.face_gallery import FaceRecognizer, load_student_classes
from core.identity import IdentityTracker
from core.keypoint_log import LOG_DIR as KEYPOINT_LOG_DIR
from core.keypoint_log import KeypointLogger
from core.video_source import (
    capture_fps,
    create_camera_capture,
    is_stream_url,
    list_cameras,
    mask_url,
    parse_camera_source,
    resolve_camera_name,
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

app = FastAPI(title="YOLOv12 + ArcFace — theo dõi lớp học")

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
    - Analyzer: YOLOv12 track + YOLO11-pose + chấm điểm trên khung mới nhất.
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
        self.session_id = uuid.uuid4().hex[:10]
        self.clock = VideoClock(fps, is_camera)
        classes = load_student_classes()
        known_classes = [c for c in classes.values() if c]
        self.recorder = BehaviorRecorder(
            fps, self.clock, source_label, dict(zip(recognizer.ids, recognizer.names)),
            legend_fn=lambda: behavior_legend(_config),
            student_classes=classes,
            # students.csv thường chỉ có HS nạp sau → HS còn lại lấy lớp phổ biến nhất
            default_class=max(set(known_classes), key=known_classes.count) if known_classes else "",
        )
        self.identity = IdentityTracker(recognizer)
        self.analyzer = FrameAnalyzer(self.cfg, self.identity)
        self.kp_log: KeypointLogger | None = None
        if self.cfg.record_keypoints:
            stem = f"{datetime.now():%Y%m%d_%H%M%S}_{self.session_id}"
            w, h = work_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.kp_log = KeypointLogger(
                KEYPOINT_LOG_DIR / f"{stem}.jsonl", source_label, fps, w, h, self.cfg.behavior,
            )

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
                detections, parsed = self.analyzer.analyze(frame, fps_eff)
            except Exception as exc:  # noqa: BLE001
                self.error = f"Lỗi phân tích: {exc}"
                self.stop.set()
                return
            last_idx = idx
            self.recorder.record(idx, detections, time.time(), frame)
            if self.kp_log is not None:
                self.kp_log.log(idx, fps_eff, detections)
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
            try:
                self.identity.recognize(frame, parsed)
            except Exception as exc:  # noqa: BLE001
                self.error = f"Lỗi ArcFace: {exc}"
                self.stop.set()
                return
            took = time.perf_counter() - t0
            last_idx = idx
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


@app.get("/api/legend")
@app.get("/api/behaviors")
async def legend():
    """Chú thích nhãn viết tắt + trạng thái bật/tắt từng hành vi."""
    return {
        "attention_threshold": _config.attention_threshold,
        "behaviors": behavior_legend(_config),
    }


@app.post("/api/behaviors")
async def set_behaviors(request: Request):
    """Bật/tắt hành vi: {"enabled": {"Cúi đầu lâu": false, ...}} hoặc {"all": true|false}.

    Áp dụng ngay cho video đang chạy và lưu vào behavior_settings.json.
    """
    body = await request.json()
    names = [name for name, _short in REASON_SHORT if name not in HIDDEN_BEHAVIORS]
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
        "attention_threshold": _config.attention_threshold,
        "behaviors": behavior_legend(_config),
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


@app.get("/api/cameras")
def cameras():
    """Camera trên máy (DirectShow): index dùng cho ô nguồn video; ưu tiên OBS Virtual Camera."""
    cams = list_cameras()
    preferred = next((c["index"] for c in cams if c["preferred"]), None)
    return {"cameras": cams, "preferred": preferred}


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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cap = None
    session: _StreamSession | None = None
    upload: Path | None = None
    try:
        data = await ws.receive_json()
        source = parse_camera_source(data.get("video_source", "0"))
        if isinstance(source, str) and not is_stream_url(source) and not os.path.exists(source):
            cam_index = await asyncio.to_thread(resolve_camera_name, source)
            if cam_index is not None:
                source = cam_index

        if isinstance(source, int) or is_stream_url(source):
            is_camera = True
            await ws.send_json({"status": "connecting", "message": "Đang kết nối camera..."})
            try:
                cap, fps, width, height = await asyncio.to_thread(create_camera_capture, source)
            except ValueError:
                if isinstance(source, int):
                    hint = ("Nếu dùng OBS: bấm Start Virtual Camera trong OBS, rồi Làm mới camera "
                            "và Kết nối lại. Camera đang bị ứng dụng khác dùng cũng không mở được.")
                else:
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
                f"Đã bắt đầu phát · YOLO: {Path(_config.yolo_model).name} · "
                f"pose: {Path(_config.pose_model).name} · gallery {len(recognizer.ids)} HS"
            ),
            "gallery_size": len(recognizer.ids),
            "session_id": session.session_id,
            "report_url": f"/api/report/{session.session_id}",
        })
        session.start()
        with _sessions_lock:
            _sessions.add(session)

        send_interval = 1.0 / min(fps, DISPLAY_MAX_FPS)
        sent_idx = -1
        sent_version = 0
        next_summary = 0.0

        async def send_summary() -> None:
            summary = await asyncio.to_thread(session.recorder.summary)
            await ws.send_json({
                "type": "summary",
                "attention_threshold": _config.attention_threshold,
                **summary,
            })

        # Lịch tuyệt đối: Windows làm tròn sleep lên ~15.6 ms, sleep cố định sẽ tụt fps
        next_send = time.monotonic()
        while True:
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
                            "name": d["name"],
                            "known": d["known"],
                            "match_pct": d["match_pct"],
                            "hard_locked": d.get("hard_locked", False),
                            "score": d["score"],
                            "focused": d["is_focused"],
                            "reasons": d["reasons"],
                        }
                        for d in detections
                    ],
                })

            if time.monotonic() >= next_summary:
                next_summary = time.monotonic() + SUMMARY_INTERVAL_SEC
                await send_summary()

            if frame is not None and frame_idx != sent_idx:
                jpeg = await asyncio.to_thread(_render_frame, frame, detections)
                await ws.send_bytes(jpeg)
                sent_idx = frame_idx
            elif session.eof.is_set():
                await send_summary()
                path = await asyncio.to_thread(_save_report, session.session_id)
                await ws.send_json({
                    "status": "ended",
                    "message": f"Video đã phát xong · đã lưu báo cáo {path.name}",
                    "report_url": f"/api/report/{session.session_id}",
                })
                break

            next_send = max(next_send + send_interval, time.monotonic() - send_interval)
            await asyncio.sleep(max(0.0, next_send - time.monotonic()))

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_json({"error": str(exc)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        if session is not None:
            with _sessions_lock:
                _sessions.discard(session)
            await asyncio.to_thread(session.close)
            try:
                path = await asyncio.to_thread(_save_report, session.session_id)
                print(f"Đã lưu báo cáo hành vi: {path}")
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


def _save_report(session_id: str) -> Path:
    with _reports_lock:
        recorder = _reports.get(session_id)
    if recorder is None:
        raise KeyError(session_id)
    return recorder.export_excel(REPORTS_DIR / recorder.report_name)


@app.get("/api/report/{session_id}")
def download_report(session_id: str):
    """Xuất Excel hành vi của phiên (đang chạy hoặc đã kết thúc)."""
    try:
        path = _save_report(session_id)
    except KeyError:
        raise HTTPException(404, "Không tìm thấy phiên — báo cáo cũ nằm trong thư mục reports/")
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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


# ---------------------------------------------------------------- nạp học sinh / webcam


def _decode_upload(f: UploadFile | None) -> np.ndarray | None:
    if f is None or not f.filename:
        return None
    data = np.frombuffer(f.file.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@app.get("/enroll")
async def enroll_page():
    """Luồng riêng: nạp học sinh + test webcam (không dùng YOLO/video session)."""
    return _read_template("enroll.html")


@app.get("/enroll/students")
def enroll_students():
    try:
        rec = get_recognizer()
        return {
            "ok": True,
            "students": rec.list_students(),
            "next_id": rec.next_student_id(),
            "count": len(rec.ids),
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/enroll/save")
def enroll_save(
    student_id: str = Form(""),
    full_name: str = Form(""),
    class_name: str = Form("LOP07"),
    img_front: UploadFile | None = File(None),
    img_left: UploadFile | None = File(None),
    img_right: UploadFile | None = File(None),
    img_up: UploadFile | None = File(None),
    img_down: UploadFile | None = File(None),
):
    """Lưu học sinh mới / cập nhật từ ảnh webcam — không đụng session video."""
    student_id = student_id.strip()
    full_name = full_name.strip() or student_id
    class_name = class_name.strip() or "LOP07"
    if not student_id:
        return JSONResponse({"ok": False, "error": "Thiếu mã học sinh."}, status_code=400)

    captures: dict[str, np.ndarray] = {}
    for angle, f in (
        ("front", img_front), ("left", img_left), ("right", img_right),
        ("up", img_up), ("down", img_down),
    ):
        if f is None or not f.filename:
            continue
        img = _decode_upload(f)
        if img is None:
            return JSONResponse({"ok": False, "error": f"Ảnh {angle} không hợp lệ."}, status_code=400)
        captures[angle] = img

    if not captures:
        return JSONResponse(
            {"ok": False, "error": "Cần chụp ít nhất 1 góc (ưu tiên front)."}, status_code=400
        )
    try:
        result = get_recognizer().enroll_student(
            student_id=student_id, full_name=full_name, captures=captures, class_name=class_name,
        )
        return {"ok": True, **result}
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/enroll/match")
def enroll_match(
    frame: UploadFile | None = File(None),
    session_id: str = Form("webcam"),
    identify: str = Form("1"),
    behavior: str = Form("1"),
):
    """Test webcam: InsightFace + ArcFace + MediaPipe hành vi (không đụng YOLO)."""
    if frame is None:
        return JSONResponse({"ok": False, "error": "Thiếu frame."}, status_code=400)
    img = _decode_upload(frame)
    if img is None:
        return JSONResponse({"ok": False, "error": "Không đọc được frame."}, status_code=400)

    t0 = time.perf_counter()
    faces: list = []
    beh: dict = {"ok": True, "behaviors": [], "timers": {}, "signals": {}}
    try:
        if identify != "0":
            faces = get_recognizer().detect_and_match(img, max_faces=5)
        if behavior != "0":
            beh = get_monitor(session_id.strip() or "webcam").analyze(img)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    return {
        "ok": True,
        "faces": faces,
        "count": len(faces),
        "identified": sum(1 for f in faces if f.get("known")),
        "behavior": beh,
        "ms": round((time.perf_counter() - t0) * 1000),
    }


@app.post("/enroll/behavior/reset")
async def enroll_behavior_reset(request: Request):
    """Reset bộ đếm thời gian hành vi (luồng webcam)."""
    session_id = "webcam"
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
        if isinstance(body, dict):
            session_id = str(body.get("session_id") or session_id)
    else:
        form = await request.form()
        session_id = str(form.get("session_id") or session_id)
    reset_monitor(session_id)
    return {"ok": True}


@app.post("/enroll/behavior/log")
async def enroll_behavior_log(request: Request):
    """Ghi 1 dòng log hành vi (webcam) ra logs/behavior_webcam.jsonl."""
    form = await request.form()
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        **{k: str(form.get(k) or "") for k in
           ("session_id", "behavior", "label", "student_id", "yaw", "pitch", "ear")},
    }
    with (log_dir / "behavior_webcam.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"ok": True}


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
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
