"""Đối chiếu PREDICTED_WINDOWS / hành vi thô với nhãn GT trong mẫu Mau_Gan_Nhan.

Không dùng E/R/N/U của hệ thống làm ground truth của chính nó.
GT tầng 1: BEHAVIOR_GT (B1–B5). GT tầng 2: ENGAGEMENT_GT (ON/OFF/UNCERTAIN).

Chỉ số (mục 8.4 bộ tiêu chí):
  - Precision / Recall / F1 theo B1–B5 (trùng thời gian tối thiểu)
  - Tỷ lệ E trùng ON; hữu ích của R (OFF/UNCERTAIN); tỷ lệ N/U; ON bị bỏ sót
  - Cohen's kappa giữa hai người gán (nếu có annotator_id khác nhau trên cùng đơn vị)
  - Coverage theo seat_zone (nếu có sheet STUDENTS)

Chạy từ 3.Thesis:
  python -m tools.eval_ernu --gt ../Mau_Gan_Nhan_Hanh_Vi_Lop_Hoc.xlsx \\
      --pred reports/bao_cao_hanh_vi_....xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from core.ernu import BEHAVIOR_TO_B, DEFAULT_PROXY_MAPPING, overlap_sec, proxy_from_label


def _read_sheet(path: Path, name: str) -> list[dict]:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    if name not in wb.sheetnames:
        return []
    ws = wb[name]
    rows = ws.iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration:
        return []
    keys = [str(h or "").strip() for h in header]
    out = []
    for vals in rows:
        row = {keys[i]: vals[i] for i in range(min(len(keys), len(vals)))}
        if any(v is not None and str(v).strip() for v in row.values()):
            out.append(row)
    return out


def _sec(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _norm_behavior(code: str | None) -> str | None:
    if code is None:
        return None
    c = str(code).strip().upper()
    if c in ("B1", "B2", "B3", "B4", "B5"):
        return c
    for name, b in BEHAVIOR_TO_B.items():
        if name.lower() == str(code).strip().lower():
            return b
    if c in ("CHUA_GAN", "UNKNOWN", ""):
        return None
    return c


def _norm_gt_label(val: str | None) -> str | None:
    if val is None:
        return None
    g = str(val).strip().upper()
    if g in ("", "CHUA_GAN"):
        return None
    if g in ("ON", "OFF", "UNCERTAIN"):
        return g
    return g


def behavior_prf(
    gt_events: list[dict],
    pred_events: list[dict],
    *,
    min_overlap_ratio: float = 0.3,
) -> dict[str, dict]:
    """Precision/Recall/F1 theo mã B1–B5 với quy tắc trùng thời gian tối thiểu."""
    by_code_gt: dict[str, list] = defaultdict(list)
    by_code_pred: dict[str, list] = defaultdict(list)
    for e in gt_events:
        code = _norm_behavior(e.get("behavior_code") or e.get("behavior"))
        s, en = _sec(e, "start_sec"), _sec(e, "end_sec")
        if code and s is not None and en is not None and en > s:
            by_code_gt[code].append((str(e.get("student_id") or ""), s, en))
    for e in pred_events:
        raw = e.get("behavior_events") or e.get("behavior_code") or e.get("behavior")
        codes = []
        if isinstance(raw, str) and "," in raw:
            codes = [_norm_behavior(x) for x in raw.split(",")]
        else:
            codes = [_norm_behavior(raw)]
        s, en = _sec(e, "start_sec"), _sec(e, "end_sec")
        sid = str(e.get("student_id") or "")
        if s is None or en is None or en <= s:
            continue
        for code in codes:
            if code in ("B1", "B2", "B3", "B4", "B5"):
                by_code_pred[code].append((sid, s, en))

    report = {}
    for code in ("B1", "B2", "B3", "B4", "B5"):
        gts = by_code_gt.get(code, [])
        preds = by_code_pred.get(code, [])
        matched_gt = set()
        matched_pred = set()
        for pi, (psid, ps, pe) in enumerate(preds):
            for gi, (gsid, gs, ge) in enumerate(gts):
                if gi in matched_gt:
                    continue
                if psid and gsid and psid != gsid:
                    continue
                ov = overlap_sec(ps, pe, gs, ge)
                denom = min(pe - ps, ge - gs)
                if denom > 0 and ov / denom >= min_overlap_ratio:
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break
        tp = len(matched_pred)
        fp = len(preds) - tp
        fn = len(gts) - len(matched_gt)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = None
        report[code] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": None if prec is None else round(prec, 3),
            "recall": None if rec is None else round(rec, 3),
            "f1": None if f1 is None else round(f1, 3),
            "n_gt": len(gts), "n_pred": len(preds),
        }
    return report


def engagement_alignment(
    eng_gt: list[dict],
    predicted: list[dict],
) -> dict:
    """Đối chiếu E với ON, R với xem lại — không coi mọi R là OFF; đếm ON bị bỏ sót."""
    e_on = e_total = 0
    r_off = r_uncertain = r_total = 0
    n_u = 0
    on_missed = on_total = 0
    compared = 0
    disagreements: list[dict] = []

    for g in eng_gt:
        gt = _norm_gt_label(g.get("gt_label"))
        if gt is None:
            continue
        sid = str(g.get("student_id") or "")
        s = _sec(g, "start_sec")
        if s is None:
            continue
        best = None
        best_ov = 0.0
        ge = _sec(g, "end_sec") or (s + 10)
        for p in predicted:
            if str(p.get("student_id") or "") != sid:
                continue
            ps, pe = _sec(p, "start_sec"), _sec(p, "end_sec")
            if ps is None or pe is None:
                continue
            ov = overlap_sec(s, ge, ps, pe)
            if ov > best_ov:
                best_ov = ov
                best = str(p.get("pred_label") or "").upper()
        if best is None or best_ov <= 0:
            continue
        compared += 1
        if gt == "ON":
            on_total += 1
            if best not in ("E",):
                on_missed += 1
                disagreements.append({
                    "video_id": g.get("video_id"),
                    "student_id": sid,
                    "start_sec": s,
                    "end_sec": ge,
                    "issue": f"ON bị bỏ sót (pred={best})",
                    "gt_label": gt,
                    "pred_label": best,
                })
        if best == "E":
            e_total += 1
            if gt == "ON":
                e_on += 1
            else:
                disagreements.append({
                    "video_id": g.get("video_id"),
                    "student_id": sid,
                    "start_sec": s,
                    "end_sec": ge,
                    "issue": f"E không khớp ON (gt={gt})",
                    "gt_label": gt,
                    "pred_label": best,
                })
        elif best == "R":
            r_total += 1
            if gt == "OFF":
                r_off += 1
            elif gt == "UNCERTAIN":
                r_uncertain += 1
        elif best in ("N", "U"):
            n_u += 1

    # Proxy alignment: máy E→ON, R→PENDING (không tự OFF); so với GT ON/OFF/UNCERTAIN
    proxy_on_hit = proxy_on_total = 0
    pending_vs_off = pending_vs_unc = pending_total = 0
    for g in eng_gt:
        gt = _norm_gt_label(g.get("gt_label"))
        if gt is None:
            continue
        sid = str(g.get("student_id") or "")
        s = _sec(g, "start_sec")
        if s is None:
            continue
        best = None
        best_proxy = None
        best_ov = 0.0
        ge = _sec(g, "end_sec") or (s + 10)
        for p in predicted:
            if str(p.get("student_id") or "") != sid:
                continue
            ps, pe = _sec(p, "start_sec"), _sec(p, "end_sec")
            if ps is None or pe is None:
                continue
            ov = overlap_sec(s, ge, ps, pe)
            if ov > best_ov:
                best_ov = ov
                best = str(p.get("pred_label") or "").upper()
                raw_px = str(p.get("proxy_label") or "").strip().upper()
                best_proxy = raw_px if raw_px in (
                    "ON", "OFF", "PENDING", "UNCERTAIN", "NO_DATA"
                ) else proxy_from_label(best, False, DEFAULT_PROXY_MAPPING)
        if best is None or best_ov <= 0:
            continue
        if best_proxy == "ON":
            proxy_on_total += 1
            if gt == "ON":
                proxy_on_hit += 1
        elif best_proxy == "PENDING":
            pending_total += 1
            if gt == "OFF":
                pending_vs_off += 1
            elif gt == "UNCERTAIN":
                pending_vs_unc += 1

    return {
        "windows_compared": compared,
        "E_total": e_total,
        "E_and_ON": e_on,
        "E_ON_rate": None if not e_total else round(e_on / e_total, 3),
        "R_total": r_total,
        "R_and_OFF": r_off,
        "R_and_UNCERTAIN": r_uncertain,
        "note": "Không coi mọi R là OFF; R hữu ích khi giáo viên cần xem lại.",
        "N_or_U_count": n_u,
        "ON_total": on_total,
        "ON_missed": on_missed,
        "ON_miss_rate": None if not on_total else round(on_missed / on_total, 3),
        "proxy": {
            "mapping": dict(DEFAULT_PROXY_MAPPING),
            "ON_total": proxy_on_total,
            "ON_and_GT_ON": proxy_on_hit,
            "ON_precision_vs_GT": None if not proxy_on_total else round(proxy_on_hit / proxy_on_total, 3),
            "PENDING_total": pending_total,
            "PENDING_vs_GT_OFF": pending_vs_off,
            "PENDING_vs_GT_UNCERTAIN": pending_vs_unc,
            "note": (
                "PENDING (từ R) không được tính là OFF tự động. "
                "OFF hệ thống chỉ hợp lệ sau review-confirm."
            ),
        },
        "disagreements": disagreements,
    }


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> dict:
    """Cohen's kappa trên cặp nhãn cùng đơn vị (cùng độ dài, đã ghép cặp)."""
    n = min(len(labels_a), len(labels_b))
    if n == 0:
        return {"n": 0, "kappa": None, "agreement": None, "note": "Không đủ cặp nhãn"}
    a = labels_a[:n]
    b = labels_b[:n]
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n
    cats = sorted(set(a) | set(b))
    ca = Counter(a)
    cb = Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if abs(1.0 - pe) < 1e-12:
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return {
        "n": n,
        "agreement": round(po, 3),
        "kappa": round(kappa, 3),
        "categories": cats,
        "note": "Đơn vị so sánh phải cùng cửa sổ/sự kiện; giải thích nhãn trước khi báo cáo.",
    }


def inter_annotator_engagement(eng_gt: list[dict]) -> dict:
    """Ghép cặp cửa sổ theo (video, student, start) giữa hai annotator_id khác nhau."""
    by_key: dict[tuple, dict[str, str]] = defaultdict(dict)
    for g in eng_gt:
        gt = _norm_gt_label(g.get("gt_label"))
        if gt is None:
            continue
        ann = str(g.get("annotator_id") or "").strip() or "G?"
        s = _sec(g, "start_sec")
        if s is None:
            continue
        key = (str(g.get("video_id") or ""), str(g.get("student_id") or ""), round(s, 1))
        by_key[key][ann] = gt

    pairs_a: list[str] = []
    pairs_b: list[str] = []
    for _key, anns in by_key.items():
        if len(anns) < 2:
            continue
        ids = sorted(anns.keys())
        pairs_a.append(anns[ids[0]])
        pairs_b.append(anns[ids[1]])
    return cohens_kappa(pairs_a, pairs_b)


def coverage_by_seat(
    students: list[dict],
    predicted: list[dict],
) -> dict:
    """Độ bao phủ trung bình theo seat_zone / seat_row (tránh diễn giải thiếu quan sát = thiếu tham gia)."""
    seat_of = {}
    for s in students:
        sid = str(s.get("student_id") or "")
        zone = s.get("seat_zone") or s.get("seat_row") or s.get("seat") or "?"
        if sid:
            seat_of[sid] = str(zone)

    by_zone: dict[str, list[float]] = defaultdict(list)
    for p in predicted:
        sid = str(p.get("student_id") or "")
        try:
            cov = float(p.get("valid_coverage"))
        except (TypeError, ValueError):
            continue
        zone = seat_of.get(sid, "?")
        by_zone[zone].append(cov)

    return {
        zone: {
            "n_windows": len(vals),
            "mean_coverage": round(sum(vals) / len(vals), 3) if vals else None,
        }
        for zone, vals in sorted(by_zone.items())
    }


def write_quality_review(path: Path, disagreements: list[dict], reviewer: str = "eval_ernu") -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "QUALITY_REVIEW"
    headers = [
        "video_id", "student_id", "start_sec", "end_sec", "issue",
        "gt_label", "pred_label", "proxy_label", "reviewer", "resolution", "note",
    ]
    ws.append(headers)
    for d in disagreements:
        pred = d.get("pred_label")
        proxy = d.get("proxy_label") or (
            proxy_from_label(str(pred or ""), False, DEFAULT_PROXY_MAPPING) if pred else ""
        )
        ws.append([
            d.get("video_id"), d.get("student_id"), d.get("start_sec"), d.get("end_sec"),
            d.get("issue"), d.get("gt_label"), pred, proxy,
            reviewer, "CHUA_XU_LY", "Sinh tự động từ eval_ernu — không phải kết luận cuối",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Đánh giá ERNU đối chứng (không dùng pred làm GT)")
    ap.add_argument("--gt", type=Path, required=True, help="Workbook mẫu gán nhãn (Mau_Gan_Nhan…)")
    ap.add_argument("--pred", type=Path, required=True, help="Báo cáo Excel có sheet PREDICTED_WINDOWS")
    ap.add_argument("--out", type=Path, default=None, help="Ghi JSON kết quả")
    ap.add_argument("--quality-out", type=Path, default=None,
                    help="Xuất QUALITY_REVIEW (.xlsx) từ bất đồng E↔ON / ON bỏ sót")
    ap.add_argument("--min-overlap", type=float, default=0.3)
    args = ap.parse_args()

    behavior_gt = _read_sheet(args.gt, "BEHAVIOR_GT")
    engagement_gt = _read_sheet(args.gt, "ENGAGEMENT_GT")
    students = _read_sheet(args.gt, "STUDENTS")
    predicted = _read_sheet(args.pred, "PREDICTED_WINDOWS")
    detail = _read_sheet(args.pred, "Chi tiết hành vi")
    pred_beh = []
    for row in detail:
        beh = row.get("Hành vi") or row.get("behavior")
        sid = row.get("Mã HS") or row.get("student_id")
        if row.get("start_sec") is not None:
            pred_beh.append({
                "student_id": sid,
                "behavior": beh,
                "start_sec": row.get("start_sec"),
                "end_sec": row.get("end_sec"),
            })
    if not pred_beh:
        for p in predicted:
            events = str(p.get("behavior_events") or "")
            if not events:
                continue
            for code in events.split(","):
                pred_beh.append({
                    "student_id": p.get("student_id"),
                    "behavior_code": code.strip(),
                    "start_sec": p.get("start_sec"),
                    "end_sec": p.get("end_sec"),
                })

    align = engagement_alignment(engagement_gt, predicted)
    disagreements = list(align.pop("disagreements", []))

    report = {
        "gt": str(args.gt),
        "pred": str(args.pred),
        "behavior_prf": behavior_prf(behavior_gt, pred_beh, min_overlap_ratio=args.min_overlap),
        "engagement_alignment": align,
        "inter_annotator_kappa": inter_annotator_engagement(engagement_gt),
        "coverage_by_seat": coverage_by_seat(students, predicted),
        "n_behavior_gt": len(behavior_gt),
        "n_engagement_gt": len(engagement_gt),
        "n_predicted_windows": len(predicted),
        "n_disagreements": len(disagreements),
        "disclaimer": (
            "Nhãn ON/OFF/UNCERTAIN là GT độc lập; E/R/N/U là đầu ra thuật toán cần đánh giá. "
            "Không dùng E/R/N/U hệ thống làm ground truth của chính nó. "
            "Không xem tất cả R là OFF. Thiếu quan sát theo vị trí ngồi không đồng nghĩa thiếu tham gia."
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    if args.quality_out:
        write_quality_review(args.quality_out, disagreements)
        print(f"Đã ghi QUALITY_REVIEW → {args.quality_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
