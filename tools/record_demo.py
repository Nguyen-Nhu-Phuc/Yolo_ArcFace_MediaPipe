"""Quay video demo dashboard (Chrome ẩn qua Playwright) khi đang phân tích 1 video.

Cần server đang chạy (python app.py). Ví dụ:
    python tools/record_demo.py "C:\\path\\video.mp4" --seconds 180
Kết quả: demo/demo_<ngày giờ>.mp4 (+ file Excel tải về trong demo/).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = BASE_DIR / "demo"


def _hold(page, seconds: float) -> None:
    page.wait_for_timeout(int(seconds * 1000))


def record(video: str, seconds: float, url: str, width: int, height: int, channel: str) -> Path:
    DEMO_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = DEMO_DIR / f"_raw_{stamp}"

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        context = browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(raw_dir),
            record_video_size={"width": width, "height": height},
            accept_downloads=True,
        )
        page = context.new_page()
        page.goto(url)
        _hold(page, 2)

        page.fill("#videoSource", video)
        _hold(page, 1)
        page.click("#btnConnect")
        page.wait_for_function(
            "document.getElementById('statusBadge').textContent.includes('Đã kết nối')",
            timeout=180_000,
        )
        t0 = time.monotonic()
        print("Đang phát — bắt đầu kịch bản demo")

        def until(t: float) -> None:
            remain = t - (time.monotonic() - t0)
            if remain > 0:
                _hold(page, remain)

        until(min(25, seconds * 0.15))
        page.keyboard.press("l")  # bật chú thích trên video
        until(min(45, seconds * 0.27))
        page.keyboard.press("l")
        until(min(60, seconds * 0.35))
        page.click("button.tab-btn[data-tab='legend']")
        until(min(75, seconds * 0.43))
        page.click("button.tab-btn[data-tab='students']")

        until(max(seconds - 15, seconds * 0.8))
        try:
            with page.expect_download(timeout=60_000) as dl:
                page.click("#btnExcel")
            excel = DEMO_DIR / dl.value.suggested_filename
            dl.value.save_as(str(excel))
            print(f"Đã tải Excel: {excel}")
        except Exception as exc:  # noqa: BLE001
            print(f"Không tải được Excel trong demo: {exc}")

        until(seconds - 3)
        page.click("#btnStop")
        _hold(page, 3)
        raw_video = Path(page.video.path())
        context.close()
        browser.close()

    out = DEMO_DIR / f"demo_{stamp}.mp4"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw_video),
         "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)],
        check=True,
    )
    shutil.rmtree(raw_dir, ignore_errors=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Quay video demo dashboard")
    ap.add_argument("video", help="Đường dẫn video cần phân tích")
    ap.add_argument("--seconds", type=float, default=180, help="Độ dài phần phát (giây)")
    ap.add_argument("--url", default="http://127.0.0.1:5000/")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--channel", default="chrome", help="chrome | msedge")
    args = ap.parse_args()
    out = record(args.video, args.seconds, args.url, args.width, args.height, args.channel)
    print(f"Video demo: {out}")


if __name__ == "__main__":
    main()
