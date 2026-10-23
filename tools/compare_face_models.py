"""So sánh ArcFace (InsightFace buffalo_l), FaceNet (Facenet512) và DeepFace trên cùng bộ ảnh lớp.

Giao thức tập đóng như eval_arcface.py: gallery = vectơ trung bình các ảnh enroll của mỗi học sinh,
kiểm tra trên ảnh test. Khuôn mặt được SCRFD phát hiện một lần trên ảnh gốc; ảnh test được thu nhỏ
rồi phóng lại (mô phỏng học sinh ngồi xa) và căn chỉnh bằng đúng điểm mốc đó, nên khác biệt giữa
các mức chỉ do độ phân giải. Mỗi mô hình nhận ảnh theo cách nó được huấn luyện:
ArcFace dùng norm_crop 112x112 theo 5 điểm, FaceNet / DeepFace dùng hộp mặt đã xoay thẳng hai mắt.

Cần môi trường riêng (TensorFlow 2.12 cho DeepFace):
    .venv-compare/bin/python tools/compare_face_models.py --dataset classroom-faces-lop07
Kết quả: reports/face_model_compare.json, reports/figures/face_compare_*.png
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
SCALES = [1.0, 0.5, 0.25, 0.15]
MODELS = ["ArcFace", "FaceNet", "DeepFace"]
DEEPFACE_NAMES = {"FaceNet": "Facenet512", "DeepFace": "DeepFace"}
# Chọn bằng thử trên ảnh gốc: FaceNet không chuẩn hóa (base) chỉ đạt hạng 1 75,8%;
# DeepFace kém ở mọi kiểu chuẩn hóa (base / raw / VGGFace2) nên giữ base.
DEEPFACE_NORMALIZATION = {"Facenet512": "Facenet", "DeepFace": "base"}
WEIGHT_FILES = {
    "ArcFace": Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx",
    "FaceNet": Path.home() / ".deepface/weights/facenet512_weights.h5",
    "DeepFace": Path.home() / ".deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5",
}


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.jpg")) if folder.is_dir() else []


def _angle_of(path: Path) -> str:
    stem = path.stem.lower()
    return next((a for a in ("front", "down", "up", "left", "right") if stem.startswith(a)), "other")


def degrade(img: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def eye_aligned_crop(img: np.ndarray, face) -> np.ndarray:
    """Xoay quanh tâm hộp cho hai mắt nằm ngang rồi cắt hộp mặt (cách deepface căn chỉnh)."""
    (lx, ly), (rx, ry) = face.kps[0], face.kps[1]
    angle = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
    x1, y1, x2, y2 = face.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, rot, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return rotated[y1:y2, x1:x2]


class Embedders:
    def __init__(self):
        from insightface.app import FaceAnalysis
        from deepface import DeepFace

        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.rec = self.app.models["recognition"]
        self.deepface = DeepFace
        self.times: dict[str, list[float]] = {m: [] for m in MODELS}
        dummy = np.full((160, 160, 3), 128, np.uint8)
        for name in DEEPFACE_NAMES.values():
            self._represent(name, dummy)

    def _represent(self, model_name: str, crop: np.ndarray) -> np.ndarray:
        r = self.deepface.represent(crop, model_name=model_name, detector_backend="skip",
                                    enforce_detection=False, align=False,
                                    normalization=DEEPFACE_NORMALIZATION[model_name])
        return np.asarray(r[0]["embedding"], np.float32)

    def detect(self, img: np.ndarray):
        faces = self.app.get(img)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def embed(self, img: np.ndarray, face) -> dict[str, np.ndarray]:
        out = {}
        t = time.perf_counter()
        out["ArcFace"] = np.asarray(self.rec.get(img, face), np.float32)
        self.times["ArcFace"].append(time.perf_counter() - t)
        crop = eye_aligned_crop(img, face)
        for model, name in DEEPFACE_NAMES.items():
            t = time.perf_counter()
            out[model] = self._represent(name, crop)
            self.times[model].append(time.perf_counter() - t)
        return {m: v / (np.linalg.norm(v) + 1e-8) for m, v in out.items()}


def _stats(values) -> dict:
    a = np.asarray(values, np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def roc_points(genuine, impostor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g, i = np.asarray(genuine), np.asarray(impostor)
    thr = np.unique(np.concatenate([g, i, [-1.0, 1.01]]))[::-1]
    tar = np.array([(g >= t).mean() for t in thr])
    far = np.array([(i >= t).mean() for t in thr])
    return thr, far, tar


def verification_metrics(genuine, impostor) -> dict:
    thr, far, tar = roc_points(genuine, impostor)
    frr = 1 - tar
    k = int(np.argmin(np.abs(far - frr)))
    g, i = np.asarray(genuine), np.asarray(impostor)

    def tar_at(target):
        ok = far <= target
        return float(tar[ok].max()) if ok.any() else 0.0

    d_prime = abs(g.mean() - i.mean()) / np.sqrt((g.var() + i.var()) / 2)
    return {"eer": float((far[k] + frr[k]) / 2), "eer_threshold": float(thr[k]),
            "tar_far_1e-2": tar_at(1e-2), "tar_far_1e-3": tar_at(1e-3), "d_prime": float(d_prime)}


def evaluate(gallery: np.ndarray, ids: list[str], probes: list[tuple[str, np.ndarray]]) -> dict:
    index = {sid: k for k, sid in enumerate(ids)}
    genuine, impostor, ranks = [], [], []
    for sid, emb in probes:
        sims = gallery @ emb
        own = index[sid]
        order = np.argsort(sims)[::-1]
        ranks.append(int(np.where(order == own)[0][0]) + 1)
        genuine.append(float(sims[own]))
        impostor.extend(float(s) for k, s in enumerate(sims) if k != own)
    n = len(ranks)
    return {
        "n_probes": n,
        "rank1": sum(r == 1 for r in ranks) / n,
        "rank5": sum(r <= 5 for r in ranks) / n,
        "genuine": _stats(genuine), "impostor": _stats(impostor),
        **verification_metrics(genuine, impostor),
        "genuine_values": genuine, "impostor_values": impostor,
    }


def plot(results: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"ArcFace": "#1f77b4", "FaceNet": "#ff7f0e", "DeepFace": "#2ca02c"}
    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        r = results[model]["1.0"]
        bins = np.linspace(-0.3, 1.0, 40)
        ax.hist(r["impostor_values"], bins=bins, density=True, alpha=0.6, color="#999999", label="Khác người")
        ax.hist(r["genuine_values"], bins=bins, density=True, alpha=0.7, color=colors[model], label="Đúng người")
        ax.set_title(f"{model} (d′ = {r['d_prime']:.2f})")
        ax.set_xlabel("Độ tương đồng cosine")
    axes[0].set_ylabel("Mật độ")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    p = out_dir / "face_compare_hist.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(str(p))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, scale in zip(axes, ["1.0", "0.25"]):
        for model in MODELS:
            r = results[model][scale]
            _, far, tar = roc_points(r["genuine_values"], r["impostor_values"])
            ax.plot(np.clip(far, 1e-4, 1), tar, color=colors[model], label=f"{model} (EER {r['eer']:.1%})")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1)
        ax.set_xlabel("FAR")
        ax.set_title("Ảnh gốc" if scale == "1.0" else f"Thu nhỏ còn {float(scale):.0%}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("TAR")
    fig.tight_layout()
    p = out_dir / "face_compare_roc.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(str(p))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh ArcFace / FaceNet / DeepFace")
    parser.add_argument("--dataset", default="classroom-faces-lop07")
    args = parser.parse_args()

    dataset = BASE_DIR / "data" / args.dataset
    students = sorted(p for p in (dataset / "faces").glob("*/*") if p.is_dir())
    emb = Embedders()

    ids, gallery, n_enroll = [], {m: [] for m in MODELS}, 0
    for sdir in students:
        vecs = {m: [] for m in MODELS}
        for path in _images(sdir / "enroll"):
            img = cv2.imread(str(path))
            face = emb.detect(img) if img is not None else None
            if face is None:
                continue
            n_enroll += 1
            for m, v in emb.embed(img, face).items():
                vecs[m].append(v)
        if not vecs["ArcFace"]:
            continue
        ids.append(sdir.name)
        for m in MODELS:
            mean = np.mean(vecs[m], axis=0)
            gallery[m].append(mean / (np.linalg.norm(mean) + 1e-8))
    gallery = {m: np.stack(v) for m, v in gallery.items()}
    print(f"Gallery: {len(ids)} học sinh từ {n_enroll} ảnh enroll")

    probes = {m: {str(s): [] for s in SCALES} for m in MODELS}
    test_info, no_face = [], []
    for sdir in students:
        if sdir.name not in ids:
            continue
        for path in _images(sdir / "test"):
            img = cv2.imread(str(path))
            face = emb.detect(img)
            if face is None:
                no_face.append(f"{sdir.name}/{path.name}")
                continue
            test_info.append({"student": sdir.name, "image": path.name, "angle": _angle_of(path),
                              "face_width": float(face.bbox[2] - face.bbox[0])})
            for s in SCALES:
                for m, v in emb.embed(degrade(img, s), face).items():
                    probes[m][str(s)].append((sdir.name, v))

    results = {m: {str(s): evaluate(gallery[m], ids, probes[m][str(s)]) for s in SCALES} for m in MODELS}
    face_widths = [t["face_width"] for t in test_info]
    summary = {
        "dataset": args.dataset,
        "students": len(ids),
        "enroll_images": n_enroll,
        "test_images": len(test_info) + len(no_face),
        "test_no_face": no_face,
        "scales": SCALES,
        "deepface_normalization": DEEPFACE_NORMALIZATION,
        "face_width_px_at_scale": {str(s): float(np.median(face_widths) * s) for s in SCALES},
        "models": {
            m: {
                "embedding_dim": int(gallery[m].shape[1]),
                "weights_mb": round(WEIGHT_FILES[m].stat().st_size / 2**20, 1) if WEIGHT_FILES[m].exists() else None,
                "embed_ms_mean": float(np.mean(emb.times[m]) * 1000),
                "embed_ms_median": float(np.median(emb.times[m]) * 1000),
                "gallery_offdiag": _stats((gallery[m] @ gallery[m].T)[~np.eye(len(ids), dtype=bool)]),
            }
            for m in MODELS
        },
        "results": results,
    }
    summary["figures"] = plot(results, BASE_DIR / "reports" / "figures")

    out = BASE_DIR / "reports" / "face_model_compare.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Ảnh test thấy mặt: {len(test_info)} / {summary['test_images']}")
    print(f"{'Mô hình':<9}{'Tỉ lệ':>6}{'Hạng 1':>8}{'EER':>8}{'TAR@1%':>8}{'d′':>7}{'Đúng':>7}{'Khác':>7}")
    for m in MODELS:
        for s in SCALES:
            r = results[m][str(s)]
            print(f"{m:<9}{s:>6.2f}{r['rank1']:>8.1%}{r['eer']:>8.1%}{r['tar_far_1e-2']:>8.1%}"
                  f"{r['d_prime']:>7.2f}{r['genuine']['mean']:>7.3f}{r['impostor']['mean']:>7.3f}")
    for m in MODELS:
        info = summary["models"][m]
        print(f"{m}: {info['embedding_dim']} chiều, {info['weights_mb']} MB, {info['embed_ms_median']:.0f} ms/mặt")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
