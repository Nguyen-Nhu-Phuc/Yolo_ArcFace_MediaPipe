"""Đưa hình camera từ EZVIZ Studio vào OBS Virtual Camera và giữ khung cắt luôn đúng.

Chạy nền khi dùng nguồn "OBS Virtual Camera" trong yolo-arcface:
    python tools/obs_ezviz.py
Cần: OBS đang mở + bật WebSocket (Tools → WebSocket Server Settings), EZVIZ Studio đang Live View.

- Tạo nguồn Window Capture "EZVIZ Studio" (nếu chưa có), cắt đúng vùng video 16:9
- Tự căn lại khi cửa sổ EZVIZ đổi kích thước; cảnh báo khi cửa sổ bị thu nhỏ (OBS mất hình)
- Canvas = output = 1280x720 (đúng cỡ yolo-arcface phân tích), lọc Lanczos → ảnh nét nhất có thể
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import ctypes
import hashlib
import json
import os
import sys
import time
from ctypes import wintypes

import websockets

INPUT_NAME = "EZVIZ Studio"
WINDOW_TITLE = "ezviz studio"
CANVAS = (1280, 720)
FPS = 15
POLL_SEC = 2.0

user32 = ctypes.windll.user32
ctypes.windll.shcore.SetProcessDpiAwareness(2)
_EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _text(h) -> str:
    n = user32.GetWindowTextLengthW(h)
    b = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(h, b, n + 1)
    return b.value


def find_ezviz():
    found = []

    @_EnumProc
    def cb(h, _):
        if user32.IsWindowVisible(h) and _text(h).strip().lower() == WINDOW_TITLE:
            found.append(h)
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else None


def video_rect(top):
    """Vùng video 16:9 (cửa sổ con không tên lớn nhất) so với góc client cửa sổ EZVIZ."""
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(top, ctypes.byref(origin))
    cr = wintypes.RECT()
    user32.GetClientRect(top, ctypes.byref(cr))
    best = []

    @_EnumProc
    def cb(c, _):
        if user32.IsWindowVisible(c) and _text(c) == "":
            r = wintypes.RECT()
            user32.GetWindowRect(c, ctypes.byref(r))
            w, h = r.right - r.left, r.bottom - r.top
            if h > 100 and 1.72 <= w / h <= 1.82:
                best.append((w * h, (r.left - origin.x, r.top - origin.y, w, h)))
        return True

    user32.EnumChildWindows(top, cb, 0)
    best.sort(reverse=True)
    return (best[0][1] if best else None), (cr.right, cr.bottom)


def obs_password() -> str:
    cfg = os.path.join(os.environ["APPDATA"], "obs-studio", "plugin_config", "obs-websocket", "config.json")
    try:
        with open(cfg, encoding="utf-8-sig") as f:
            return json.load(f).get("server_password", "")
    except OSError:
        return ""


class Obs:
    def __init__(self, ws):
        self.ws = ws
        self.n = 0

    @classmethod
    async def connect(cls, url: str, password: str) -> "Obs":
        ws = await websockets.connect(url, max_size=None)
        hello = json.loads(await ws.recv())["d"]
        ident = {"rpcVersion": 1}
        auth = hello.get("authentication")
        if auth:
            secret = base64.b64encode(hashlib.sha256((password + auth["salt"]).encode()).digest())
            ident["authentication"] = base64.b64encode(
                hashlib.sha256(secret + auth["challenge"].encode()).digest()).decode()
        await ws.send(json.dumps({"op": 1, "d": ident}))
        if json.loads(await ws.recv())["op"] != 2:
            raise RuntimeError("OBS từ chối xác thực WebSocket")
        return cls(ws)

    async def req(self, rtype: str, data: dict | None = None, ok=(100,)) -> dict:
        self.n += 1
        rid = str(self.n)
        await self.ws.send(json.dumps({"op": 6, "d": {
            "requestType": rtype, "requestId": rid, "requestData": data or {}}}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg["op"] == 7 and msg["d"]["requestId"] == rid:
                st = msg["d"]["requestStatus"]
                if st["code"] not in ok:
                    raise RuntimeError(f"{rtype}: {st.get('comment') or st['code']}")
                return msg["d"].get("responseData") or {}


async def ensure_setup(obs: Obs) -> tuple[str, int]:
    """Cấu hình video, tạo nguồn (nếu thiếu), bật Virtual Camera. Trả về (scene, sceneItemId)."""
    vc_on = (await obs.req("GetVirtualCamStatus"))["outputActive"]
    vs = await obs.req("GetVideoSettings")
    want = {"fpsNumerator": FPS, "fpsDenominator": 1, "baseWidth": CANVAS[0], "baseHeight": CANVAS[1],
            "outputWidth": CANVAS[0], "outputHeight": CANVAS[1]}
    if any(vs.get(k) != v for k, v in want.items()):
        if vc_on:
            await obs.req("StopVirtualCam")
            await asyncio.sleep(1)
            vc_on = False
        await obs.req("SetVideoSettings", want)
    await obs.req("SetProfileParameter", {"parameterCategory": "Video", "parameterName": "ScaleType",
                                          "parameterValue": "lanczos"})

    scene = (await obs.req("GetCurrentProgramScene"))["currentProgramSceneName"]
    inputs = {i["inputName"] for i in (await obs.req("GetInputList"))["inputs"]}
    if INPUT_NAME not in inputs:
        await obs.req("CreateInput", {
            "sceneName": scene, "inputName": INPUT_NAME, "inputKind": "window_capture",
            "inputSettings": {"method": 2, "cursor": False, "client_area": True},
        })
    for name in ("Desktop Audio", "Mic/Aux"):
        if name in inputs:
            await obs.req("SetInputMute", {"inputName": name, "inputMuted": True}, ok=(100, 600))
    item_id = (await obs.req("GetSceneItemId", {"sceneName": scene, "sourceName": INPUT_NAME}))["sceneItemId"]
    await obs.req("SetSceneItemTransform", {"sceneName": scene, "sceneItemId": item_id, "sceneItemTransform": {
        "positionX": 0, "positionY": 0, "alignment": 5,
        "boundsType": "OBS_BOUNDS_SCALE_INNER", "boundsAlignment": 0,
        "boundsWidth": CANVAS[0], "boundsHeight": CANVAS[1],
    }})
    await obs.req("SetSceneItemEnabled", {"sceneName": scene, "sceneItemId": item_id, "sceneItemEnabled": True})
    if not vc_on:
        await obs.req("StartVirtualCam")
    return scene, item_id


async def bind_window(obs: Obs) -> bool:
    """Chọn lại đúng cửa sổ chính 'Ezviz Studio' (EZVIZ có cả cửa sổ PTZ / đăng nhập trùng exe)."""
    items = (await obs.req("GetInputPropertiesListPropertyItems",
                           {"inputName": INPUT_NAME, "propertyName": "window"}))["propertyItems"]
    target = next((it["itemValue"] for it in items if it["itemValue"].lower().startswith(WINDOW_TITLE + ":")), None)
    if target is None:
        return False
    await obs.req("SetInputSettings", {"inputName": INPUT_NAME, "inputSettings": {"window": ""}})
    await asyncio.sleep(0.4)
    await obs.req("SetInputSettings", {"inputName": INPUT_NAME, "inputSettings": {
        "window": target, "method": 2, "cursor": False, "client_area": True}})
    return True


async def source_size(obs: Obs, scene: str, item_id: int) -> tuple[float, float]:
    tr = (await obs.req("GetSceneItemTransform", {"sceneName": scene, "sceneItemId": item_id}))["sceneItemTransform"]
    return tr["sourceWidth"], tr["sourceHeight"]


async def apply_crop(obs: Obs, scene: str, item_id: int, vr, client) -> str:
    sw, sh = await source_size(obs, scene, item_id)
    if abs(sw - client[0]) > 2 or abs(sh - client[1]) > 2:
        await bind_window(obs)
        for _ in range(20):
            await asyncio.sleep(0.3)
            sw, sh = await source_size(obs, scene, item_id)
            if abs(sw - client[0]) <= 2 and abs(sh - client[1]) <= 2:
                break
    if not sw or not sh:
        return "OBS chưa chụp được cửa sổ EZVIZ"
    kx, ky = sw / client[0], sh / client[1]
    x, y, w, h = vr
    crop = {"cropLeft": round(x * kx), "cropTop": round(y * ky),
            "cropRight": max(0, round(sw - (x + w) * kx)), "cropBottom": max(0, round(sh - (y + h) * ky))}
    await obs.req("SetSceneItemTransform", {"sceneName": scene, "sceneItemId": item_id, "sceneItemTransform": {
        **crop, "positionX": 0, "positionY": 0, "alignment": 5,
        "boundsType": "OBS_BOUNDS_SCALE_INNER", "boundsAlignment": 0,
        "boundsWidth": CANVAS[0], "boundsHeight": CANVAS[1],
    }})
    note = "" if w >= CANVAS[0] else f" — vùng video chỉ {w}x{h} < {CANVAS[0]}: phóng to cửa sổ EZVIZ để nét hơn"
    return f"Đã căn khung: video {w}x{h} tại ({x},{y}){note}"


async def run(url: str, once: bool) -> None:
    obs = await Obs.connect(url, obs_password())
    scene, item_id = await ensure_setup(obs)
    print(f"OBS sẵn sàng: nguồn '{INPUT_NAME}', {CANVAS[0]}x{CANVAS[1]} @ {FPS}fps, Virtual Camera bật")
    last_key = None
    last_state = None
    while True:
        top = find_ezviz()
        if top is None:
            state = "Không thấy cửa sổ EZVIZ Studio — hãy mở EZVIZ Studio và Live View camera"
        elif user32.IsIconic(top):
            state = "Cửa sổ EZVIZ đang THU NHỎ → OBS/yolo-arcface mất hình. Mở lại cửa sổ (không thu nhỏ)."
            last_key = None
        else:
            vr, client = video_rect(top)
            if vr is None:
                state = "EZVIZ chưa có vùng video — mở Live View 1 camera"
            else:
                key = (vr, client)
                if key != last_key:
                    print(time.strftime("%H:%M:%S"), await apply_crop(obs, scene, item_id, vr, client))
                    last_key = key
                state = "ok"
        if state != last_state and state != "ok":
            print(time.strftime("%H:%M:%S"), state)
        last_state = state
        if once:
            break
        await asyncio.sleep(POLL_SEC)


def main() -> None:
    ap = argparse.ArgumentParser(description="EZVIZ Studio → OBS Virtual Camera (tự căn khung)")
    ap.add_argument("--url", default="ws://127.0.0.1:4455")
    ap.add_argument("--once", action="store_true", help="Căn 1 lần rồi thoát (không theo dõi)")
    args = ap.parse_args()
    if sys.platform != "win32":
        raise SystemExit("Chỉ chạy trên Windows")
    try:
        asyncio.run(run(args.url, args.once))
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        raise SystemExit(f"Không kết nối được OBS WebSocket ({args.url}): {exc}. "
                         "Mở OBS và bật Tools → WebSocket Server Settings → Enable.")


if __name__ == "__main__":
    main()
