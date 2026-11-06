"""Danh sách camera đã lưu. File chứa link RTSP (có thể có mật khẩu) — không trả raw URL ra API."""

from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from urllib.parse import quote, urlparse

from core.config import DATA_DIR
from core.video_source import STREAM_PREFIXES, mask_url

CAMERAS_PATH = DATA_DIR / "cameras.json"
MAX_CAMERAS = 8
_NAME_MAX = 40
_SOURCE_MAX = 500
_BLOCKED_HOSTS = {"169.254.169.254"}
_HOST_RE = re.compile(
    r"(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$",
    re.I,
)
_PATH_RE = re.compile(r"/[A-Za-z0-9._~!$&'()*+,;=:@%/-]{0,200}")
DEFAULT_REMOTE_PORT = 15554


class CameraError(ValueError):
    """Dữ liệu camera không dùng được. Thông báo không chứa mật khẩu."""


def normalize_source(value) -> int | str:
    """Số thiết bị, link RTSP/HTTP, hoặc tên camera trên máy (ví dụ OBS Virtual Camera)."""
    if isinstance(value, bool) or value is None:
        raise CameraError("Nguồn camera không hợp lệ.")
    if isinstance(value, int):
        if not 0 <= value <= 32:
            raise CameraError("Số camera phải từ 0 đến 32.")
        return value
    text = str(value).strip()
    if not text or len(text) > _SOURCE_MAX:
        raise CameraError("Nhập link RTSP hoặc số camera.")
    if text.isdigit():
        return normalize_source(int(text))
    lower = text.lower()
    if lower.startswith(STREAM_PREFIXES):
        parsed = urlparse(text)
        host = (parsed.hostname or "").strip("[]").lower()
        if not host or host in _BLOCKED_HOSTS:
            raise CameraError("Địa chỉ camera không hợp lệ.")
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise CameraError("Cổng camera không hợp lệ.")
        return text
    if any(part in text for part in ("/", "\\", "..")) or lower.startswith("file:"):
        raise CameraError("Chỉ nhận link RTSP/HTTP hoặc camera trên máy, không nhận đường dẫn file.")
    if not re.fullmatch(r"[^<>:\"|?*\x00-\x1f]{1,120}", text):
        raise CameraError("Tên camera trên máy không hợp lệ.")
    return text


def build_remote_rtsp(host, port=DEFAULT_REMOTE_PORT, username="", password="", path="") -> str:
    """Ghép URL RTSP công khai: cổng ngoài (Port Forward) + hostname DDNS.

    Mật khẩu được mã hóa trong URL và không xuất hiện trong thông báo lỗi.
    """
    raw_host = str(host or "").strip()
    if not raw_host or "://" in raw_host or "@" in raw_host or "/" in raw_host or ":" in raw_host:
        raise CameraError("Ô địa chỉ chỉ nhận hostname DDNS, ví dụ your-ddns.ddns.net.")
    checked = _public_host(raw_host)
    port_n = _remote_port(port)
    user = str(username or "")
    secret = str(password or "")
    if not user.strip():
        raise CameraError("Nhập tên đăng nhập camera.")
    if any(ord(ch) < 32 for ch in user + secret):
        raise CameraError("Tài khoản camera không hợp lệ.")
    stream = _stream_path(path)
    url = f"rtsp://{quote(user, safe='')}:{quote(secret, safe='')}@{checked}:{port_n}{stream}"
    return str(normalize_source(url))


def source_from_body(body) -> tuple[str, int | str]:
    """Nhận form DDNS (host, port, username, password, path) hoặc link RTSP có sẵn."""
    if not isinstance(body, dict):
        raise CameraError("Dữ liệu camera không hợp lệ.")
    name = body.get("name", "")
    host = str(body.get("host") or "").strip()
    if host:
        source = build_remote_rtsp(
            host,
            DEFAULT_REMOTE_PORT if body.get("port") in (None, "") else body.get("port"),
            body.get("username", ""),
            body.get("password", ""),
            body.get("path", ""),
        )
        return name, source
    return name, body.get("source", "")


def _public_host(host: str) -> str:
    if _is_ipv4(host):
        checked = host
    else:
        try:
            checked = host.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise CameraError("Hostname DDNS không hợp lệ.") from exc
        if not _HOST_RE.fullmatch(checked):
            raise CameraError("Hostname DDNS không hợp lệ.")
    if checked in _BLOCKED_HOSTS:
        raise CameraError("Địa chỉ camera không hợp lệ.")
    return checked


def _is_ipv4(host: str) -> bool:
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit() or part != str(int(part)) or not 0 <= int(part) <= 255:
            return False
    return True


def _remote_port(port) -> int:
    if isinstance(port, bool):
        raise CameraError("Cổng ngoài không hợp lệ.")
    try:
        port_n = int(port)
    except (TypeError, ValueError) as exc:
        raise CameraError("Cổng ngoài không hợp lệ.") from exc
    if not 1 <= port_n <= 65535:
        raise CameraError("Cổng ngoài phải từ 1 đến 65535.")
    return port_n


def _stream_path(path) -> str:
    stream = str(path or "").strip()
    if not stream:
        return ""
    if "://" in stream or ".." in stream:
        raise CameraError("Đường dẫn stream không hợp lệ.")
    if not stream.startswith("/"):
        stream = "/" + stream
    if not _PATH_RE.fullmatch(stream):
        raise CameraError("Đường dẫn stream không hợp lệ.")
    return stream


def source_label(source) -> str:
    """Nhãn an toàn để hiện trên giao diện."""
    if isinstance(source, int):
        return f"thiết bị {source}"
    text = str(source)
    if text.lower().startswith(STREAM_PREFIXES):
        return mask_url(text)
    return text


class CameraRegistry:
    def __init__(self, path: Path = CAMERAS_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        self._load()

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._items.values()]

    def get(self, camera_id: str) -> dict | None:
        with self._lock:
            item = self._items.get(camera_id)
            return dict(item) if item else None

    def add(self, name: str, source) -> dict:
        cleaned = _clean_name(name)
        stored = normalize_source(source)
        with self._lock:
            if len(self._items) >= MAX_CAMERAS:
                raise CameraError(f"Tối đa {MAX_CAMERAS} camera.")
            if any(item["source"] == stored for item in self._items.values()):
                raise CameraError("Camera này đã có trong danh sách.")
            if not cleaned:
                cleaned = f"Camera {len(self._items) + 1}"
            item = {"id": "cam_" + uuid.uuid4().hex[:8], "name": cleaned, "source": stored}
            self._items[item["id"]] = item
            self._save()
            return dict(item)

    def remove(self, camera_id: str) -> None:
        with self._lock:
            if camera_id not in self._items:
                raise CameraError("Không thấy camera.")
            del self._items[camera_id]
            self._save()

    def _load(self) -> None:
        from core.db import enabled, get_db
        if enabled():
            doc = get_db().settings.find_one({"_id": "cameras"})
            if doc is None:
                self._load_file()
                self._save()
                return
            items = doc.get("cameras") if isinstance(doc, dict) else None
            if isinstance(items, list):
                self._fill(items)
            return
        self._load_file()

    def _load_file(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        items = raw.get("cameras") if isinstance(raw, dict) else None
        if isinstance(items, list):
            self._fill(items)

    def _fill(self, items: list) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            camera_id = str(item.get("id") or "")
            if not re.fullmatch(r"cam_[0-9a-f]{8}", camera_id):
                continue
            try:
                source = normalize_source(item.get("source"))
                name = _clean_name(item.get("name")) or camera_id
            except CameraError:
                continue
            self._items[camera_id] = {"id": camera_id, "name": name, "source": source}

    def _save(self) -> None:
        from core.db import enabled, get_db
        payload = list(self._items.values())
        if enabled():
            get_db().settings.replace_one(
                {"_id": "cameras"},
                {"_id": "cameras", "cameras": payload},
                upsert=True,
            )
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"cameras": payload}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def _clean_name(name) -> str:
    text = re.sub(r"\s+", " ", str(name or "")).strip()
    return text[:_NAME_MAX]
