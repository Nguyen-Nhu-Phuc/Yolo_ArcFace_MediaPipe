"""Mở nguồn video: webcam / OBS Virtual Camera (số), camera IP (rtsp://...), hoặc file."""

from __future__ import annotations

import os
import re
import subprocess

import cv2

STREAM_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")
PREFERRED_CAMERA = re.compile(r"obs\s*virtual", re.I)
CAMERA_SIZE = (1280, 720)
_DSHOW_DEVICE_RE = re.compile(r'\]\s+"(.+)"\s+\((video|none)\)')


def list_cameras() -> list[dict]:
    """Camera DirectShow theo đúng thứ tự index của cv2.CAP_DSHOW (Windows).

    OBS Virtual Camera chưa Start thì ffmpeg báo kiểu "(none)" — vẫn liệt kê để giữ đúng index.
    """
    if os.name != "nt":
        return []
    import imageio_ffmpeg

    try:
        out = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-list_devices", "true",
             "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
    except Exception:  # noqa: BLE001
        return []
    names = [m.group(1) for m in _DSHOW_DEVICE_RE.finditer(out.stderr)]
    return [
        {"index": i, "name": name, "preferred": bool(PREFERRED_CAMERA.search(name))}
        for i, name in enumerate(names)
    ]


def resolve_camera_name(text: str) -> int | None:
    """'OBS Virtual Camera' / 'obs' → index DirectShow (khớp một phần, không phân biệt hoa thường)."""
    needle = text.strip().lower()
    if not needle:
        return None
    for cam in list_cameras():
        if needle in cam["name"].lower():
            return cam["index"]
    return None


def is_stream_url(source) -> bool:
    return isinstance(source, str) and source.strip().lower().startswith(STREAM_PREFIXES)


def parse_camera_source(value):
    """'0' → 0 (webcam); URL / đường dẫn file giữ nguyên."""
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def mask_url(url) -> str:
    """Che mật khẩu (mã xác minh EZVIZ) trong URL camera."""
    return re.sub(r"(://[^:/@]+):[^@/]*@", r"\1:***@", str(url))


def open_stream(url: str, timeout_ms: int = 10000) -> cv2.VideoCapture:
    # RTSP qua UDP mất gói trên Wi-Fi → khung xám/nhòe; ép TCP
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    return cv2.VideoCapture(url.strip(), cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
    ])


def capture_fps(cap, default: float = 25.0) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and 0 < fps <= 120 else default


def create_camera_capture(source=0) -> tuple:
    """Trả về (VideoCapture, fps, width, height)."""
    if is_stream_url(source):
        cap = open_stream(source)
        label = mask_url(source)
    else:
        # OBS Virtual Camera chỉ có trên DirectShow; index phải khớp list_cameras() → thử DSHOW trước
        backends = (cv2.CAP_DSHOW, cv2.CAP_ANY) if os.name == "nt" else (cv2.CAP_ANY,)
        cap = None
        for backend in backends:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                break
            cap.release()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
        label = source
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera: {label}")

    fps = capture_fps(cap, default=30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        cap.release()
        raise ValueError(f"Invalid camera resolution: {width}x{height}")
    return cap, fps, width, height
