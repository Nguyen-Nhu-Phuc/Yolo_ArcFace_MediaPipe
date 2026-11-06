"""Kết nối MongoDB Atlas. Dữ liệu lớp, ảnh, báo cáo và camera không ghi ra đĩa khi đã cấu hình."""

from __future__ import annotations

import os
import threading
from pathlib import Path

_ENV_LOADED = False
_client = None
_lock = threading.Lock()


def load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def enabled() -> bool:
    load_env()
    return bool((os.environ.get("MONGODB_URI") or "").strip())


def database_name() -> str:
    load_env()
    return (os.environ.get("MONGODB_DB") or "luanvan").strip() or "luanvan"


def _patch_tls_for_libressl(client) -> None:
    """Python 3.9 trên macOS dùng LibreSSL 2.8. PyMongo tái sử dụng TLS session
    và lần kết nối sau lỗi BAD_PSK_IDENTITY. Giữ TLS 1.2 và không gửi session cũ."""
    import ssl

    if "LibreSSL" not in getattr(ssl, "OPENSSL_VERSION", ""):
        return
    pool = client.options.pool_options
    getter = pool._ssl_context
    ctx = getter() if callable(getter) else getter
    if ctx is None or getattr(ctx, "_luanvan_tls_patched", False):
        return
    if hasattr(ssl, "TLSVersion"):
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    original = ctx.wrap_socket

    def wrap_socket(*args, **kwargs):
        kwargs.pop("session", None)
        return original(*args, **kwargs)

    ctx.wrap_socket = wrap_socket
    ctx._luanvan_tls_patched = True


def get_db():
    """Database Atlas, hoặc None khi chưa có MONGODB_URI."""
    global _client
    if not enabled():
        return None
    with _lock:
        if _client is None:
            from pymongo import MongoClient

            uri = os.environ["MONGODB_URI"].strip()
            _client = MongoClient(
                uri,
                serverSelectionTimeoutMS=8000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000,
                maxPoolSize=20,
                tz_aware=True,
            )
            _patch_tls_for_libressl(_client)
        return _client[database_name()]


def ping() -> str:
    db = get_db()
    if db is None:
        raise RuntimeError("Chưa có MONGODB_URI trong file .env")
    db.command("ping")
    return database_name()
