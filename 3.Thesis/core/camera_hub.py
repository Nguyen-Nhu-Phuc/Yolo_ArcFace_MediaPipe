"""Đọc song song nhiều camera để xem trước. YOLO chỉ chạy khi một camera được chọn theo dõi.

Mất tín hiệu thì kết nối lại với thời gian chờ tăng dần, tối đa 30 giây. Không ghi link có mật khẩu.
"""

from __future__ import annotations

import threading
import time

import cv2

from core.camera_registry import CameraError, CameraRegistry, source_label
from core.video_source import create_camera_capture, is_stream_url, resolve_camera_name

PREVIEW_MAX_WIDTH = 640
PREVIEW_INTERVAL = 0.2  # 5 khung/giây — đủ để xem, nhẹ hơn luồng phân tích
OPEN_WAIT_SEC = 12.0
READ_FAILS_BEFORE_RECONNECT = 25
BACKOFF_START = 1.0
BACKOFF_MAX = 30.0


def _preview_jpeg(frame) -> bytes:
    height, width = frame.shape[:2]
    if width > PREVIEW_MAX_WIDTH:
        scale = PREVIEW_MAX_WIDTH / float(width)
        frame = cv2.resize(frame, (PREVIEW_MAX_WIDTH, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return buf.tobytes() if ok else b""


def _open_source(source):
    """Tên camera trên máy (OBS…) được đổi sang số ngay trước lúc mở."""
    if isinstance(source, str) and not is_stream_url(source):
        index = resolve_camera_name(source)
        if index is None:
            raise ValueError("Không thấy camera trên máy")
        source = index
    return create_camera_capture(source)


class _Preview:
    def __init__(self, camera_id: str, source, name: str, epoch: int):
        self.camera_id = camera_id
        self.source = source
        self.name = name
        self.epoch = epoch
        self.stop = threading.Event()
        self.pause = threading.Event()
        self.idle = threading.Event()
        self.idle.set()
        self._lock = threading.Lock()
        self.status = "offline"
        self.error: str | None = None
        self.jpeg = b""
        self.seq = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"CamPreview-{camera_id}")

    def start(self) -> None:
        self.thread.start()

    def hold_for_analysis(self) -> None:
        """Nhả thiết bị để luồng phân tích mở đúng một kết nối."""
        self.pause.set()
        self.idle.wait(OPEN_WAIT_SEC)

    def resume(self) -> None:
        self.pause.clear()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.camera_id,
                "name": self.name,
                "epoch": self.epoch,
                "label": source_label(self.source),
                "status": "analyzing" if self.pause.is_set() else self.status,
                "error": None if self.pause.is_set() else self.error,
                "fps": round(self.fps, 1),
                "width": self.width,
                "height": self.height,
            }

    def take_jpeg(self, seen: int) -> tuple[int, bytes] | None:
        with self._lock:
            if not self.jpeg or self.seq == seen:
                return None
            return self.seq, self.jpeg

    def _set(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)

    def _run(self) -> None:
        delay = BACKOFF_START
        while not self.stop.is_set():
            if self.pause.is_set():
                self._set(status="analyzing", error=None)
                self.idle.set()
                if self.stop.wait(0.2):
                    return
                continue
            self.idle.clear()
            self._set(status="connecting", error=None)
            try:
                cap, _fps, width, height = _open_source(self.source)
            except Exception:  # noqa: BLE001 — không đưa URL trong thông báo
                self.idle.set()
                self._set(status="error", error="Không kết nối được. Kiểm tra địa chỉ, mã xác minh và đã bật RTSP chưa.")
                if self.stop.wait(delay):
                    return
                delay = min(delay * 2, BACKOFF_MAX)
                continue
            delay = BACKOFF_START
            self._set(status="online", error=None, width=width, height=height, fps=0.0)
            fails = 0
            reads = 0
            window = time.monotonic()
            last_jpeg = 0.0
            try:
                while not self.stop.is_set() and not self.pause.is_set():
                    try:
                        ok, frame = cap.read()
                    except cv2.error:
                        ok, frame = False, None
                    if not ok or frame is None:
                        fails += 1
                        if fails >= READ_FAILS_BEFORE_RECONNECT:
                            break
                        if self.stop.wait(0.2):
                            return
                        continue
                    fails = 0
                    reads += 1
                    now = time.monotonic()
                    if now - window >= 2.0:
                        self._set(fps=reads / (now - window))
                        reads = 0
                        window = now
                    if now - last_jpeg >= PREVIEW_INTERVAL:
                        jpeg = _preview_jpeg(frame)
                        if jpeg:
                            with self._lock:
                                self.jpeg = jpeg
                                self.seq += 1
                        last_jpeg = now
            finally:
                cap.release()
                self.idle.set()
            if self.stop.is_set() or self.pause.is_set():
                continue
            self._set(status="connecting", error="Mất tín hiệu, đang kết nối lại.")
            if self.stop.wait(delay):
                return
            delay = min(delay * 2, BACKOFF_MAX)
        self._set(status="offline")
        self.idle.set()


class CameraHub:
    def __init__(self, registry: CameraRegistry | None = None):
        self.registry = registry or CameraRegistry()
        self._lock = threading.Lock()
        self._workers: dict[str, _Preview] = {}
        self._claimed: set[str] = set()
        self._epoch = 0

    def start_all(self) -> None:
        for cam in self.registry.list():
            self._start(cam)

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop.set()
            worker.pause.clear()
        for worker in workers:
            worker.thread.join(timeout=3)

    def list_public(self) -> list[dict]:
        with self._lock:
            workers = dict(self._workers)
        rows = []
        for cam in self.registry.list():
            worker = workers.get(cam["id"])
            if worker is None:
                rows.append({
                    "id": cam["id"],
                    "name": cam["name"],
                    "epoch": 0,
                    "label": source_label(cam["source"]),
                    "status": "offline",
                    "error": None,
                    "fps": 0,
                    "width": 0,
                    "height": 0,
                })
            else:
                rows.append(worker.snapshot())
        return rows

    def add(self, name: str, source) -> dict:
        cam = self.registry.add(name, source)
        self._start(cam)
        return self._public(cam["id"])

    def remove(self, camera_id: str) -> None:
        with self._lock:
            if camera_id in self._claimed:
                raise CameraError("Camera đang được theo dõi. Bấm Dừng trước khi xóa.")
        self.registry.remove(camera_id)
        self._stop(camera_id)

    def restart(self, camera_id: str) -> dict:
        cam = self.registry.get(camera_id)
        if cam is None:
            raise CameraError("Không thấy camera.")
        with self._lock:
            if camera_id in self._claimed:
                raise CameraError("Camera đang được theo dõi.")
        self._stop(camera_id)
        self._start(cam)
        return self._public(camera_id)

    def claim(self, camera_id: str) -> tuple:
        """Tạm dừng xem trước và trả (nguồn, tên) cho luồng phân tích. Nguồn không ra khỏi server."""
        cam = self.registry.get(camera_id)
        if cam is None:
            raise CameraError("Không thấy camera.")
        with self._lock:
            if camera_id in self._claimed:
                raise CameraError("Camera này đang được theo dõi ở một phiên khác.")
            self._claimed.add(camera_id)
            worker = self._workers.get(camera_id)
        try:
            if worker is not None:
                worker.hold_for_analysis()
        except Exception:
            with self._lock:
                self._claimed.discard(camera_id)
            raise
        return cam["source"], cam["name"]

    def release(self, camera_id: str) -> None:
        with self._lock:
            self._claimed.discard(camera_id)
            worker = self._workers.get(camera_id)
        if worker is not None:
            worker.resume()

    def frames_since(self, seen: dict[str, int]) -> list[tuple[str, int, bytes]]:
        with self._lock:
            workers = list(self._workers.values())
        ready = []
        for worker in workers:
            item = worker.take_jpeg(seen.get(worker.camera_id, 0))
            if item is not None:
                ready.append((worker.camera_id, item[0], item[1]))
        return ready

    def _public(self, camera_id: str) -> dict:
        for row in self.list_public():
            if row["id"] == camera_id:
                return row
        raise CameraError("Không thấy camera.")

    def _start(self, cam: dict) -> None:
        self._epoch += 1
        worker = _Preview(cam["id"], cam["source"], cam["name"], self._epoch)
        with self._lock:
            previous = self._workers.pop(cam["id"], None)
            self._workers[cam["id"]] = worker
        if previous is not None:
            previous.stop.set()
            previous.pause.clear()
            previous.thread.join(timeout=3)
        worker.start()

    def _stop(self, camera_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker is None:
            return
        worker.stop.set()
        worker.pause.clear()
        worker.thread.join(timeout=3)
