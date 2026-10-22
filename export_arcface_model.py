"""
Xuất "model nhận diện" ArcFace từ dataset lớp học.

Cách làm đúng:
  - Backbone ArcFace (InsightFace buffalo_l) là pretrained — KHÔNG train lại trên 33 HS
  - Chạy ArcFace trên ảnh enroll → lưu embedding gallery (= model nhận diện của lớp)

Output:
  models/arcface_lop07/
    gallery.npz   (ids, names, embeddings)
    meta.json

Chạy:
  python export_arcface_model.py
  python export_arcface_model.py --dataset 2_dataset_arcface
  python export_arcface_model.py --dataset classroom-faces-lop07  # bản cũ
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from face_augment import augment_far_variants
from face_gallery import THRESHOLD_KEYS

BASE_DIR = Path(__file__).resolve().parent

SPLIT_WEIGHT = {"enroll": 1.0, "test": 0.85}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ANGLE_WEIGHTS = {
    "front": 1.00,
    "down": 0.95,
    "up": 0.40,
    "left": 0.55,
    "right": 0.55,
}
PREFERRED = ("front", "down")


def _angle_of(path: Path) -> str:
    stem = path.stem.lower()
    for a in ("front", "down", "up", "left", "right"):
        if stem.startswith(a):
            return a
    return "other"


def load_names(csv_path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    if not csv_path.exists():
        return names
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("student_id") or "").strip()
            full = (row.get("full_name") or sid).strip()
            if sid:
                names[sid] = full
    return names


def build_insightface():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    # det_size vừa đủ cho ảnh enroll + crop người trong lớp
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def embed_image(app, img_bgr: np.ndarray) -> np.ndarray | None:
    """Trả embedding L2-normalized 512-d, hoặc None."""
    if img_bgr is None or img_bgr.size == 0:
        return None

    # Thử detect+recognize
    faces = app.get(img_bgr)
    if faces:
        # lấy mặt lớn nhất
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        emb = np.asarray(faces[0].normed_embedding, dtype=np.float32)
        return emb

    # Ảnh enroll đã crop sẵn — pad thêm viền rồi detect lại
    h, w = img_bgr.shape[:2]
    pad = max(h, w) // 4
    padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    faces = app.get(padded)
    if faces:
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return np.asarray(faces[0].normed_embedding, dtype=np.float32)

    # Upscale nếu ảnh quá nhỏ
    if max(h, w) < 160:
        scale = 160.0 / max(h, w)
        up = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return embed_image(app, up)

    return None


def export_gallery(dataset_dir: Path, out_dir: Path) -> dict:
    faces_root = dataset_dir / "faces"
    students_csv = dataset_dir / "students.csv"
    if not faces_root.exists():
        raise FileNotFoundError(f"Không thấy {faces_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    name_map = load_names(students_csv)

    print("Nạp InsightFace ArcFace (buffalo_l)...")
    app = build_insightface()
    print("OK — xuất embedding từ enroll/ + test/ (+ augment xa)")

    ids: list[str] = []
    names: list[str] = []
    embeddings: list[np.ndarray] = []
    details: list[dict] = []

    class_dirs = sorted(p for p in faces_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    t0 = time.time()
    for class_dir in class_dirs:
        for student_dir in sorted(class_dir.glob("HS*")):
            sources: list[tuple[Path, float]] = []
            for split_name in ("enroll", "test"):
                split_dir = student_dir / split_name
                if not split_dir.is_dir():
                    continue
                split_w = SPLIT_WEIGHT.get(split_name, 0.8)
                for p in sorted(split_dir.iterdir()):
                    if p.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    sources.append((p, split_w))
            if not sources:
                continue

            preferred = [t for t in sources if _angle_of(t[0]) in PREFERRED]
            others = [t for t in sources if _angle_of(t[0]) not in PREFERRED]
            use_sources = (preferred + others) if preferred else sources

            weighted: list[tuple[np.ndarray, float]] = []
            used_files: list[str] = []
            for img_path, split_w in use_sources:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                base_w = ANGLE_WEIGHTS.get(_angle_of(img_path), 0.3) * split_w
                views: list[tuple[np.ndarray, float]] = [(img, 1.0)]
                if img_path.parent.name == "enroll":
                    views.extend(augment_far_variants(img))

                added_from_file = False
                for view, view_mult in views:
                    emb = embed_image(app, view)
                    if emb is None:
                        if view_mult >= 1.0:
                            print(
                                f"  ! bỏ ảnh (không detect mặt): "
                                f"{img_path.relative_to(dataset_dir)}"
                            )
                        continue
                    weighted.append((emb, base_w * view_mult))
                    added_from_file = True
                if added_from_file:
                    used_files.append(f"{img_path.parent.name}/{img_path.name}")

            if not weighted:
                print(f"  X bỏ học sinh {student_dir.name}: không embed được")
                continue

            total_w = sum(w for _, w in weighted)
            mean = sum(e * w for e, w in weighted) / total_w
            mean = mean / (np.linalg.norm(mean) + 1e-8)

            sid = student_dir.name
            ids.append(sid)
            names.append(name_map.get(sid, sid))
            embeddings.append(mean.astype(np.float32))
            details.append({"student_id": sid, "images": used_files, "n": len(weighted)})
            print(f"  + {sid}: {len(weighted)} ảnh → embedding OK")

    if not embeddings:
        raise RuntimeError("Không xuất được embedding nào.")

    emb_mat = np.stack(embeddings, axis=0)
    # Kiểm tra độ tách (off-diagonal cosine)
    sims = emb_mat @ emb_mat.T
    np.fill_diagonal(sims, np.nan)
    off = sims[~np.isnan(sims)]
    print(
        f"\nGallery: {len(ids)} HS | emb {emb_mat.shape} | "
        f"sim giữa HS khác nhau: mean={float(off.mean()):.3f} max={float(off.max()):.3f}"
    )
    if float(off.mean()) > 0.85:
        print("CẢNH BÁO: embedding vẫn khá gần nhau — ảnh enroll có thể kém / giống nhau.")

    npz_path = out_dir / "gallery.npz"
    np.savez(
        npz_path,
        ids=np.array(ids, dtype=object),
        names=np.array(names, dtype=object),
        embeddings=emb_mat,
    )

    meta = {
        "backend": "insightface",
        "model": "buffalo_l",
        "task": "arcface_gallery",
        "dataset": str(dataset_dir.resolve()),
        "num_students": len(ids),
        "embedding_dim": int(emb_mat.shape[1]),
        "match_threshold": 0.45,
        "min_compat_pct": 45.0,
        "min_margin": 0.004,
        "lock_sim": 0.45,
        "offdiag_sim_mean": float(off.mean()),
        "offdiag_sim_max": float(off.max()),
        "students": details,
        "gallery_augment": "enroll+test; far_downscale/jpeg/blur on enroll",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        # Giữ ngưỡng đã chỉnh trên giao diện khi dựng lại gallery
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update({k: old[k] for k in THRESHOLD_KEYS if k in old})
        except Exception:  # noqa: BLE001
            pass
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Đồng bộ cache cũ để tương thích
    legacy = dataset_dir / "arcface_gallery.npz"
    np.savez(
        legacy,
        ids=np.array(ids, dtype=object),
        names=np.array(names, dtype=object),
        embeddings=emb_mat,
    )

    print(f"\nĐã xuất model nhận diện:")
    print(f"  {npz_path}")
    print(f"  {out_dir / 'meta.json'}")
    print(f"  {legacy} (cache tương thích)")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="2_dataset_arcface",
        help="Thư mục dataset trong project (mặc định 2_dataset_arcface)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Thư mục output (mặc định models/arcface_<tên dataset>)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = BASE_DIR / dataset_dir
    out_dir = Path(args.out) if args.out else BASE_DIR / "models" / f"arcface_{dataset_dir.name}"
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir

    export_gallery(dataset_dir, out_dir)


if __name__ == "__main__":
    main()
