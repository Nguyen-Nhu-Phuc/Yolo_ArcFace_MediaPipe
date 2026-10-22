"""Đánh giá nhận diện ArcFace kiểu tập đóng: gallery dựng từ ảnh enroll, kiểm tra trên ảnh test.

Gallery dựng đúng như export_arcface_model.py (trọng số góc mặt, ảnh enroll thêm biến thể "ngồi xa"),
mỗi học sinh 1 vectơ trung bình. Ảnh test không đưa vào gallery.

    python eval_arcface.py --dataset classroom-faces-lop07
Kết quả: reports/arcface_eval_<dataset>.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import cv2
import numpy as np

from export_arcface_model import ANGLE_WEIGHTS, IMAGE_SUFFIXES, _angle_of, embed_image
from face_augment import augment_far_variants
from face_gallery import get_face_app

BASE_DIR = Path(__file__).resolve().parent
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if folder.is_dir() else []


def build_gallery(app, students: list[Path]) -> tuple[list[str], np.ndarray, int]:
    ids, vectors, n_images = [], [], 0
    for sdir in students:
        weighted = []
        for img_path in _images(sdir / "enroll"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            base_w = ANGLE_WEIGHTS.get(_angle_of(img_path), 0.3)
            used = False
            for view, mult in [(img, 1.0), *augment_far_variants(img)]:
                emb = embed_image(app, view)
                if emb is not None:
                    weighted.append((emb, base_w * mult))
                    used = True
            n_images += used
        if not weighted:
            print(f"  bỏ {sdir.name}: không có ảnh enroll thấy mặt")
            continue
        mean = sum(e * w for e, w in weighted) / sum(w for _, w in weighted)
        ids.append(sdir.name)
        vectors.append(mean / (np.linalg.norm(mean) + 1e-8))
    return ids, np.stack(vectors).astype(np.float32), n_images


def _stats(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá ArcFace: gallery từ enroll, kiểm tra trên test")
    parser.add_argument("--dataset", default="classroom-faces-lop07")
    args = parser.parse_args()

    dataset = BASE_DIR / args.dataset
    students = sorted(p for p in (dataset / "faces").glob("*/*") if p.is_dir())
    app = get_face_app()
    ids, gallery, n_gallery_images = build_gallery(app, students)
    index = {sid: i for i, sid in enumerate(ids)}
    print(f"Gallery: {len(ids)} học sinh từ {n_gallery_images} ảnh enroll")

    genuine, impostor, rows, no_face = [], [], [], []
    for sdir in students:
        if sdir.name not in index:
            continue
        for img_path in _images(sdir / "test"):
            emb = embed_image(app, cv2.imread(str(img_path)))
            if emb is None:
                no_face.append(f"{sdir.name}/{img_path.name}")
                continue
            sims = gallery @ emb
            own = index[sdir.name]
            order = np.argsort(sims)[::-1]
            genuine.append(float(sims[own]))
            impostor.extend(float(s) for i, s in enumerate(sims) if i != own)
            rows.append({
                "student": sdir.name, "image": img_path.name, "angle": _angle_of(img_path),
                "rank": int(np.where(order == own)[0][0]) + 1,
                "best": ids[int(order[0])], "best_sim": float(sims[order[0]]),
                "own_sim": float(sims[own]), "margin": float(sims[order[0]] - sims[order[1]]),
            })

    n = len(rows)
    thresholds = []
    for t in THRESHOLDS:
        correct = sum(1 for r in rows if r["rank"] == 1 and r["best_sim"] >= t)
        wrong = sum(1 for r in rows if r["rank"] != 1 and r["best_sim"] >= t)
        tar = float(np.mean([g >= t for g in genuine]))
        thresholds.append({
            "threshold": t, "tar": tar, "frr": 1 - tar, "far": float(np.mean([s >= t for s in impostor])),
            "correct_id": correct / n, "wrong_id": wrong / n, "rejected": (n - correct - wrong) / n,
        })

    per_angle = defaultdict(list)
    for r in rows:
        per_angle[r["angle"]].append(r)
    per_student = defaultdict(list)
    for r in rows:
        per_student[r["student"]].append(r["own_sim"])

    result = {
        "dataset": args.dataset,
        "students": len(ids),
        "gallery_images": n_gallery_images,
        "test_images": n + len(no_face),
        "test_with_face": n,
        "test_no_face": no_face,
        "rank1": sum(r["rank"] == 1 for r in rows) / n,
        "rank5": sum(r["rank"] <= 5 for r in rows) / n,
        "genuine": _stats(genuine),
        "impostor": _stats(impostor),
        "gallery_offdiag": _stats((gallery @ gallery.T)[~np.eye(len(ids), dtype=bool)]),
        "thresholds": thresholds,
        "per_angle": {
            a: {"n": len(rs), "rank1": sum(r["rank"] == 1 for r in rs) / len(rs),
                "genuine_mean": float(np.mean([r["own_sim"] for r in rs])),
                "accept_040": sum(r["rank"] == 1 and r["best_sim"] >= 0.40 for r in rs) / len(rs),
                "accept_045": sum(r["rank"] == 1 and r["best_sim"] >= 0.45 for r in rs) / len(rs)}
            for a, rs in sorted(per_angle.items())
        },
        "lowest_students": sorted(
            ({"student": s, "mean_sim": float(np.mean(v)), "n": len(v)} for s, v in per_student.items()),
            key=lambda x: x["mean_sim"])[:5],
        "errors": [r for r in rows if r["rank"] != 1],
        "genuine_values": genuine,
        "impostor_values": impostor,
    }
    out = BASE_DIR / "reports" / f"arcface_eval_{args.dataset}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Ảnh test: {result['test_images']} ({n} thấy mặt) · hạng 1 {result['rank1']:.1%} · hạng 5 {result['rank5']:.1%}")
    print(f"Đúng người: {result['genuine']['mean']:.3f} ± {result['genuine']['std']:.3f} · "
          f"khác người: {result['impostor']['mean']:.3f} ± {result['impostor']['std']:.3f}")
    print(f"{'Ngưỡng':>7}{'TAR':>8}{'FAR':>8}{'FRR':>8}{'Nhận đúng':>11}{'Nhận nhầm':>11}")
    for t in thresholds:
        print(f"{t['threshold']:>7.2f}{t['tar']:>8.1%}{t['far']:>8.2%}{t['frr']:>8.1%}"
              f"{t['correct_id']:>11.1%}{t['wrong_id']:>11.1%}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
