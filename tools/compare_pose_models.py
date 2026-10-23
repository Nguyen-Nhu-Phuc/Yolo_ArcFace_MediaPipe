"""So sánh OpenPose, MediaPipe Pose Landmarker và YOLO11-pose trên khung video lớp học.

Không có nhãn khung xương, nên đo những gì luật hành vi cần: với mỗi học sinh do YOLOv12 tìm được,
có lấy được mũi, hai vai, hai cổ tay hay không (độ tin cậy > 0,5 như behavior.KP_CONF), chia theo
tam phân vị bề rộng hộp người (xa / giữa / gần camera), cùng thời gian xử lý trên CPU.
Hộp nhận nhầm logo, dòng giờ của camera được lọc như hệ thống (identity.is_osd_false_person).

- OpenPose: mô hình COCO 18 điểm của CMU (bản PyTorch lllyasviel/Annotators body_pose_model.pth),
  bottom-up trên cả khung, nối khớp bằng PAF (mã giải của controlnet_aux.open_pose);
- MediaPipe Pose Landmarker lite và YOLO11n-pose: top-down trên ảnh cắt từng người (cắt như pose_estimator.py).
Đo thêm tỷ lệ MediaPipe Face Landmarker lấy được điểm mốc mắt (EAR) theo bề rộng hộp người.

    .venv/bin/python tools/compare_pose_models.py --video ~/Downloads/hiv00026.mp4 --frames 30
Kết quả: reports/pose_model_compare.json, reports/figures/pose_compare.png (đã làm mờ mặt)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))  # chạy được bằng python tools/<tên>.py

from core.analysis import HEAD_CROP_FRAC, fit_width
from core.config import load_config
from core.identity import is_osd_false_person
from core.pose_estimator import PAD_X, PAD_Y, PoseEstimator

OPENPOSE_WEIGHTS = Path.home() / ".openpose" / "body_pose_model.pth"
MP_POSE_MODEL = BASE_DIR / "models" / "mediapipe" / "pose_landmarker_lite.task"
KP_CONF = 0.5
METHODS = ["OpenPose", "MediaPipe", "YOLO11-pose"]
# Chỉ số khớp: mũi, vai trái, vai phải, cổ tay trái, cổ tay phải
JOINTS = {
    "OpenPose": {"nose": 0, "l_shoulder": 5, "r_shoulder": 2, "l_wrist": 7, "r_wrist": 4},
    "MediaPipe": {"nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_wrist": 15, "r_wrist": 16},
    "YOLO11-pose": {"nose": 0, "l_shoulder": 5, "r_shoulder": 6, "l_wrist": 9, "r_wrist": 10},
}
# Chia theo tam phân vị bề rộng hộp người (camera nhìn chéo nên hộp nhỏ = ngồi xa camera)
BANDS = ["xa camera", "giữa lớp", "gần camera"]
WIDTH_BINS = [(0, 60), (60, 90), (90, 120), (120, 10_000)]
OP_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 0), (0, 14), (0, 15), (14, 16), (15, 17), (1, 8), (1, 11)]
COCO_LIMBS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (0, 1), (0, 2), (1, 3), (2, 4), (5, 11), (6, 12)]
MP_LIMBS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 2), (0, 5), (2, 7), (5, 8), (11, 23), (12, 24)]


def load_openpose_body():
    """Nạp controlnet_aux.open_pose mà không chạy __init__ của controlnet_aux (kéo theo nhiều thư viện).

    Bản gốc cố định scale_search = [0.5] và boxsize = 368 tính theo chiều cao ảnh, tức mạng chỉ nhận
    ảnh cao 184 pixel. Ở đây đặt scale 1,0 và boxsize theo biến module BOXSIZE = chiều cao đưa vào mạng
    (OpenPose mặc định net_resolution -1x368).
    """
    pkg_dir = Path(importlib.util.find_spec("controlnet_aux").submodule_search_locations[0]) / "open_pose"
    pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader("open_pose", loader=None))
    pkg.__path__ = [str(pkg_dir)]
    sys.modules["open_pose"] = pkg
    # body và util import lẫn nhau: nạp body trước (bỏ dòng import util), util sau, rồi gắn lại
    for name in ("model", "body", "util"):
        spec = importlib.util.spec_from_file_location(f"open_pose.{name}", pkg_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"open_pose.{name}"] = mod
        source = (pkg_dir / f"{name}.py").read_text(encoding="utf-8")
        if name == "body":
            source = (source.replace("scale_search = [0.5]", "scale_search = [1.0]")
                      .replace("boxsize = 368", "boxsize = BOXSIZE")
                      .replace("from . import util\n", "")
                      .replace("from scipy.ndimage.filters import", "from scipy.ndimage import"))
            mod.BOXSIZE = 368
        exec(compile(source, str(pkg_dir / f"{name}.py"), "exec"), mod.__dict__)
    body = sys.modules["open_pose.body"]
    body.util = sys.modules["open_pose.util"]
    return body


def crop_boxes(frame, boxes):
    h, w = frame.shape[:2]
    out = []
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, int(x1 - bw * PAD_X)), max(0, int(y1 - bh * PAD_Y))
        cx2, cy2 = min(w, int(x2 + bw * PAD_X)), min(h, int(y2 + bh * PAD_Y))
        out.append(((cx1, cy1), frame[cy1:cy2, cx1:cx2]))
    return out


class MediaPipeRunner:
    def __init__(self):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as base

        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base.BaseOptions(model_asset_path=str(MP_POSE_MODEL)),
            running_mode=vision.RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4))

    def estimate(self, frame, boxes):
        import mediapipe as mp

        out = []
        for (ox, oy), crop in crop_boxes(frame, boxes):
            rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            res = self.pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.pose_landmarks:
                out.append(None)
                continue
            ch, cw = crop.shape[:2]
            out.append(np.array([[lm.x * cw + ox, lm.y * ch + oy, lm.visibility]
                                 for lm in res.pose_landmarks[0]], np.float32))
        return out


def openpose_to_boxes(candidate, subset, boxes):
    """Gán mỗi người OpenPose cho hộp YOLO chứa nhiều khớp của người đó nhất (mỗi hộp một người)."""
    people = []
    for person in subset:
        kp = np.zeros((18, 3), np.float32)
        for j, idx in enumerate(person[:18].astype(int)):
            if idx >= 0:
                kp[j] = [candidate[idx][0], candidate[idx][1], candidate[idx][2]]
        people.append(kp)
    scores = np.zeros((len(people), len(boxes)))
    for p, kp in enumerate(people):
        pts = kp[kp[:, 2] > 0]
        for b, (x1, y1, x2, y2) in enumerate(boxes):
            scores[p, b] = np.sum((pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2))
    assigned: list = [None] * len(boxes)
    while scores.size and scores.max() > 0:
        p, b = np.unravel_index(np.argmax(scores), scores.shape)
        assigned[b] = people[p]
        scores[p, :] = 0
        scores[:, b] = 0
    return assigned


def joint_flags(kpts, method) -> dict:
    if kpts is None:
        return {"pose": False, "nose": False, "shoulders": False, "wrists": False, "core": False}
    j = JOINTS[method]
    ok = {k: bool(kpts[i][2] > KP_CONF) for k, i in j.items()}
    return {"pose": True, "nose": ok["nose"], "shoulders": ok["l_shoulder"] and ok["r_shoulder"],
            "wrists": ok["l_wrist"] or ok["r_wrist"],
            "core": ok["nose"] and ok["l_shoulder"] and ok["r_shoulder"]}


def blur_heads(frame, boxes):
    out = frame.copy()
    for x1, y1, x2, y2 in boxes:
        hy = y1 + int((y2 - y1) * HEAD_CROP_FRAC)
        roi = out[y1:hy, x1:x2]
        if roi.size:
            k = max(15, (x2 - x1) // 2 * 2 + 1)
            out[y1:hy, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def draw(frame, boxes, kps_list, limbs, color):
    img = blur_heads(frame, boxes)
    for kp in kps_list:
        if kp is None:
            continue
        for a, b in limbs:
            if kp[a][2] > KP_CONF and kp[b][2] > KP_CONF:
                cv2.line(img, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])), color, 2, cv2.LINE_AA)
        for x, y, c in kp:
            if c > KP_CONF:
                cv2.circle(img, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh OpenPose / MediaPipe / YOLO11-pose")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--openpose-sizes", default="368,736", help="boxsize OpenPose (chiều cao đưa vào mạng)")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    from core.eye_detector import EyeDetector

    torch.set_num_threads(max(1, os.cpu_count() or 4))
    cfg = load_config()
    detector = YOLO(cfg.yolo_model)
    yolo_pose = PoseEstimator(cfg.pose_model, cfg.pose_imgsz, cfg.pose_conf)
    mp_pose = MediaPipeRunner()
    eye = EyeDetector()
    op_module = load_openpose_body()
    openpose = op_module.Body(str(OPENPOSE_WEIGHTS))
    op_sizes = [int(s) for s in args.openpose_sizes.split(",")]

    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = np.linspace(total * 0.05, total * 0.95, args.frames).astype(int)

    methods = ["YOLO11-pose", "MediaPipe"] + [f"OpenPose@{s}" for s in op_sizes]
    times = {m: [] for m in methods + ["YOLOv12"]}
    rows = []
    eye_rows = []
    sample = None
    for n, idx in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = fit_width(frame)
        fh, fw = frame.shape[:2]

        t = time.perf_counter()
        res = detector.predict(frame, conf=cfg.confidence_threshold, iou=cfg.iou_threshold, classes=[0],
                               imgsz=cfg.yolo_imgsz, verbose=False)[0]
        times["YOLOv12"].append(time.perf_counter() - t)
        boxes = [tuple(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]
        boxes = [b for b in boxes if not is_osd_false_person(*b, fw, fh)]
        if not boxes:
            continue

        out = {}
        t = time.perf_counter()
        out["YOLO11-pose"] = yolo_pose.estimate(frame, boxes)
        times["YOLO11-pose"].append(time.perf_counter() - t)
        t = time.perf_counter()
        out["MediaPipe"] = mp_pose.estimate(frame, boxes)
        times["MediaPipe"].append(time.perf_counter() - t)
        for s in op_sizes:
            op_module.BOXSIZE = s
            t = time.perf_counter()
            cand, subset = openpose(frame)
            times[f"OpenPose@{s}"].append(time.perf_counter() - t)
            out[f"OpenPose@{s}"] = openpose_to_boxes(cand, subset, boxes)

        for b, box in enumerate(boxes):
            row = {"frame": int(idx), "box_w": box[2] - box[0]}
            for m in methods:
                row[m] = joint_flags(out[m][b], m.split("@")[0])
            rows.append(row)
            x1, y1, x2, y2 = box
            face = eye.measure(frame[y1:y1 + int((y2 - y1) * HEAD_CROP_FRAC), x1:x2])
            eye_rows.append({"box_w": x2 - x1, "ok": face is not None})
        if sample is None or len(boxes) > sample[2]:
            sample = (frame, out, len(boxes), boxes)
        print(f"[{n + 1}/{len(picks)}] khung {idx}: {len(boxes)} người", flush=True)

    cuts = np.percentile([r["box_w"] for r in rows], [100 / 3, 200 / 3])
    for r in rows:
        r["band"] = BANDS[int(np.searchsorted(cuts, r["box_w"], side="right"))]

    def rate(rs, m, key):
        return sum(r[m][key] for r in rs) / len(rs) if rs else None

    summary = {
        "video": os.path.basename(args.video),
        "frames": len(times["YOLOv12"]),
        "students": len(rows),
        "kp_conf": KP_CONF,
        "time_ms_per_frame": {m: {"mean": float(np.mean(v) * 1000), "median": float(np.median(v) * 1000)}
                              for m, v in times.items() if v},
        "rates": {m: {k: rate(rows, m, k) for k in ("pose", "nose", "shoulders", "wrists", "core")} for m in methods},
        "band_box_width_cuts": [float(c) for c in cuts],
        "rates_by_band": {band: {"n": sum(r["band"] == band for r in rows),
                                 **{m: rate([r for r in rows if r["band"] == band], m, "core") for m in methods}}
                          for band in BANDS},
        "box_width_median": float(np.median([r["box_w"] for r in rows])),
        "face_landmarker_by_width": [
            {"range": f"{lo}–{hi if hi < 10_000 else '∞'}",
             "n": len(sel := [e for e in eye_rows if lo <= e["box_w"] < hi]),
             "ok_rate": (sum(e["ok"] for e in sel) / len(sel)) if sel else None}
            for lo, hi in WIDTH_BINS
        ],
    }

    if sample is not None:
        frame, out, _, boxes = sample
        panels = [
            ("OpenPose (bottom-up, 368)", draw(frame, boxes, out[f"OpenPose@{op_sizes[0]}"], OP_LIMBS, (0, 0, 255))),
            ("MediaPipe Pose (crop)", draw(frame, boxes, out["MediaPipe"], MP_LIMBS, (255, 128, 0))),
            ("YOLO11-pose (crop)", draw(frame, boxes, out["YOLO11-pose"], COCO_LIMBS, (0, 200, 0))),
        ]
        # Chỉ giữ vùng có học sinh để ba khung xếp chồng vừa một trang A4
        top = max(0, min(b[1] for b in boxes) - 10)
        tiles = []
        for title, img in panels:
            img = img[top:]
            bar = np.full((44, img.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, title, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            tiles.append(np.vstack([bar, img]))
        fig_dir = BASE_DIR / "reports" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / "pose_compare.png"
        cv2.imwrite(str(fig_path), np.vstack(tiles))
        summary["figure"] = str(fig_path)

    out_path = BASE_DIR / "reports" / "pose_model_compare.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("frames", "students", "time_ms_per_frame", "rates",
                                               "rates_by_band", "face_landmarker_by_width")},
                     ensure_ascii=False, indent=1))
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
