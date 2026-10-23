"""Chạy lại luật hành vi trên log keypoint và đo với file nhãn.

    python tools/eval_behavior.py --log logs/keypoints/x.jsonl --labels data/labels/x.csv
    python tools/eval_behavior.py --log ... --labels ... --set head_down_drop=0.4
    python tools/eval_behavior.py --log ... --labels ... --grid head_down_drop=0.3,0.4,0.5 \
        --grid head_down_duration=3,4 --behaviors "Cúi đầu lâu"
    python tools/eval_behavior.py --log ... --dump data/labels/x_du_doan.csv   # đợt dự đoán → CSV nhãn

Nhãn: CSV cột student_id,behavior,start,end[,note]. start/end là giây hoặc mm:ss / hh:mm:ss
(vị trí trong video). student_id = HSxxx, hoặc IDnn cho track chưa nhận diện.
Nên gán đủ mọi hành vi của mọi HS trong đoạn đã gán nhãn — đợt dự đoán không khớp nhãn
nào được tính là báo nhầm.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))  # chạy được bằng python tools/<tên>.py

from core.analysis import apply_chat
from core.behavior import REASON_SHORT, StudentStateTracker, calculate_attention_score
from core.chat_detector import ChatTracker
from core.config import Config
from core.keypoint_log import read_log

BEHAVIORS = [name for name, _short in REASON_SHORT]
_SECONDS_RE = re.compile(r"(\d+(?:\.\d+)?)s\)")


@dataclass
class Episode:
    student: str
    behavior: str
    start: float
    end: float

    @property
    def length(self) -> float:
        return max(1e-6, self.end - self.start)


# ---------------------------------------------------------------- config


def _cast(value: str, like):
    if isinstance(like, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(float(value))
    if isinstance(like, float):
        return float(value)
    return value


def make_config(overrides: dict[str, str]) -> Config:
    cfg = Config()
    cfg.behavior.disabled = frozenset()
    for key, value in overrides.items():
        target = cfg.behavior if hasattr(cfg.behavior, key) else cfg if hasattr(cfg, key) else None
        if target is None:
            raise SystemExit(f"Không có tham số: {key}")
        setattr(target, key, _cast(value, getattr(target, key)))
    return cfg


def _parse_kv(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Cần dạng tên=giá_trị: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------- chạy lại


def _reason_name(reason: str) -> tuple[str | None, float]:
    """'Cúi đầu lâu (3.2s)' → ('Cúi đầu lâu', 3.2)."""
    for name in BEHAVIORS:
        if reason.startswith(name):
            m = _SECONDS_RE.search(reason)
            return name, float(m.group(1)) if m else 0.0
    return None, 0.0


def replay(samples: list[dict], cfg: Config, gap: float) -> tuple[list[Episode], dict[str, float]]:
    """→ (đợt dự đoán, số giây quan sát theo HS)."""
    tracker = StudentStateTracker()
    chat = ChatTracker()
    # (tid, hành vi) → [[start, end], ...]
    spans: dict[tuple[int, str], list[list[float]]] = defaultdict(list)
    votes: dict[int, Counter] = defaultdict(Counter)
    observed: dict[int, float] = defaultdict(float)

    for s in samples:
        t, fps_eff = float(s["t"]), float(s["fps_eff"])
        dt = 1.0 / fps_eff
        dets = []
        for tr in s["tracks"]:
            kpts = np.asarray(tr["kpts"], dtype=np.float32)
            signals = tr.get("signals") or {}
            score, reasons = calculate_attention_score(
                kpts, cfg, tracker, tr["key"], fps_eff, signals=signals)
            x1, y1, x2, y2 = tr["box"]
            dets.append({
                "track_id": tr["tid"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "yaw": tracker.last_turn.get(tr["key"]), "mar": signals.get("mar"),
                "score": score, "reasons": reasons, "is_focused": None,
            })
            if tr.get("sid"):
                votes[tr["tid"]][tr["sid"]] += 1
            observed[tr["tid"]] += dt
        apply_chat(chat, dets, cfg, fps_eff)
        for d in dets:
            for reason in d["reasons"]:
                name, secs = _reason_name(reason)
                if name is None:
                    continue
                lst = spans[(d["track_id"], name)]
                if lst and t - lst[-1][1] <= gap:
                    lst[-1][1] = t + dt
                else:
                    lst.append([t - secs, t + dt])

    def student(tid: int) -> str:
        v = votes.get(tid)
        return v.most_common(1)[0][0] if v else f"ID{tid}"

    merged: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for (tid, name), lst in spans.items():
        merged[(student(tid), name)].extend(lst)
    episodes = []
    for (sid, name), lst in merged.items():
        lst.sort()
        cur = list(lst[0])
        for a, b in lst[1:]:
            if a - cur[1] <= gap:
                cur[1] = max(cur[1], b)
            else:
                episodes.append(Episode(sid, name, max(0.0, cur[0]), cur[1]))
                cur = [a, b]
        episodes.append(Episode(sid, name, max(0.0, cur[0]), cur[1]))

    seen: dict[str, float] = defaultdict(float)
    for tid, secs in observed.items():
        seen[student(tid)] += secs
    return episodes, seen


# ---------------------------------------------------------------- nhãn + đo


def parse_time(text: str) -> float:
    text = text.strip()
    if ":" not in text:
        return float(text)
    secs = 0.0
    for part in text.split(":"):
        secs = secs * 60 + float(part)
    return secs


def read_labels(path: Path) -> list[Episode]:
    out = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            sid = (row.get("student_id") or "").strip()
            name = (row.get("behavior") or "").strip()
            if not sid or not name:
                continue
            if name not in BEHAVIORS:
                print(f"  Bỏ dòng {i}: hành vi không có trong danh sách: {name!r}", file=sys.stderr)
                continue
            out.append(Episode(sid.upper(), name, parse_time(row["start"]), parse_time(row["end"])))
    return out


def _overlap(a: Episode, b: Episode) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    return max(0.0, inter) / min(a.length, b.length)


def score_behaviors(pred: list[Episode], labels: list[Episode], seen: dict[str, float],
                    behaviors: list[str], min_overlap: float) -> dict[str, dict]:
    hours = max(1e-9, sum(seen.values()) / 3600)
    out = {}
    for name in behaviors:
        p = [e for e in pred if e.behavior == name]
        g = [e for e in labels if e.behavior == name]
        tp_pred = sum(1 for e in p if any(
            e.student == l.student and _overlap(e, l) >= min_overlap for l in g))
        tp_label = sum(1 for l in g if any(
            e.student == l.student and _overlap(e, l) >= min_overlap for e in p))
        out[name] = {"pred": len(p), "label": len(g), "tp_pred": tp_pred, "tp_label": tp_label,
                     "fa_per_hour": (len(p) - tp_pred) / hours}
    return out


def _prf(tp_pred: int, n_pred: int, tp_label: int, n_label: int) -> tuple[float | None, float | None, float | None]:
    p = tp_pred / n_pred if n_pred else None
    r = tp_label / n_label if n_label else None
    if p is None or r is None:
        return p, r, None
    if p + r == 0:
        return p, r, 0.0
    return p, r, 1.25 * p * r / (0.25 * p + r)


def _fmt(v: float | None) -> str:
    return "  -  " if v is None else f"{v * 100:5.1f}"


def print_table(metrics: dict[str, dict]) -> None:
    print(f"{'Hành vi':<22}{'Dự đoán':>8}{'Nhãn':>6}{'Prec%':>7}{'Rec%':>7}{'F0.5%':>7}{'Nhầm/HS-giờ':>13}")
    for name, m in metrics.items():
        if not m["pred"] and not m["label"]:
            continue
        p, r, f = _prf(m["tp_pred"], m["pred"], m["tp_label"], m["label"])
        print(f"{name:<22}{m['pred']:>8}{m['label']:>6}{_fmt(p):>7}{_fmt(r):>7}{_fmt(f):>7}"
              f"{m['fa_per_hour']:>13.2f}")


def total(metrics: dict[str, dict]) -> tuple[float | None, float | None, float | None, float]:
    s = {k: sum(m[k] for m in metrics.values()) for k in ("pred", "label", "tp_pred", "tp_label")}
    fa = sum(m["fa_per_hour"] for m in metrics.values())
    return (*_prf(s["tp_pred"], s["pred"], s["tp_label"], s["label"]), fa)


def dump_episodes(path: Path, episodes: list[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "behavior", "start", "end", "note"])
        for e in sorted(episodes, key=lambda e: (e.start, e.student)):
            w.writerow([e.student, e.behavior, _hms(e.start), _hms(e.end), "dự đoán — kiểm tra lại"])


def _hms(secs: float) -> str:
    secs = max(0.0, round(secs, 1))
    whole = int(secs)
    return f"{whole // 3600:d}:{whole % 3600 // 60:02d}:{secs % 60:04.1f}"


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Đo luật hành vi trên log keypoint")
    parser.add_argument("--log", required=True)
    parser.add_argument("--labels", help="CSV nhãn (mặc định labels/<tên log>.csv nếu có)")
    parser.add_argument("--set", action="append", default=[], metavar="tên=giá_trị")
    parser.add_argument("--grid", action="append", default=[], metavar="tên=a,b,c")
    parser.add_argument("--behaviors", help="Chỉ đo các hành vi này (phân tách bằng dấu phẩy)")
    parser.add_argument("--gap", type=float, default=1.5, help="Nối 2 đợt cách nhau ≤ N giây")
    parser.add_argument("--min-overlap", type=float, default=0.3,
                        help="Khớp khi phần chồng / đợt ngắn hơn ≥ N")
    parser.add_argument("--top", type=int, default=15, help="Số bộ tham số in ra khi --grid")
    parser.add_argument("--dump", help="Ghi đợt dự đoán ra CSV cùng định dạng nhãn")
    args = parser.parse_args()

    log_path = Path(args.log)
    header, samples = read_log(log_path)
    if not samples:
        raise SystemExit("Log không có mẫu nào.")
    t_min, t_max = samples[0]["t"], samples[-1]["t"]
    print(f"{log_path.name}: {len(samples)} mẫu, {t_min:.0f}s → {t_max:.0f}s "
          f"(nguồn: {header.get('source', '?')})")

    label_path = Path(args.labels) if args.labels else BASE_DIR / "data" / "labels" / f"{log_path.stem}.csv"
    labels: list[Episode] | None = None
    if label_path.exists():
        labels = [e for e in read_labels(label_path) if e.end >= t_min and e.start <= t_max + 1.0]
        print(f"Nhãn: {label_path} — {len(labels)} đợt trong đoạn của log")
    elif args.labels:
        raise SystemExit(f"Không thấy file nhãn: {label_path}")

    behaviors = ([b.strip() for b in args.behaviors.split(",")] if args.behaviors else BEHAVIORS)
    for b in behaviors:
        if b not in BEHAVIORS:
            raise SystemExit(f"Không có hành vi: {b}")
    base = _parse_kv(args.set)
    grid = {k: [x.strip() for x in v.split(",") if x.strip()] for k, v in _parse_kv(args.grid).items()}

    if not grid:
        cfg = make_config(base)
        pred, seen = replay(samples, cfg, args.gap)
        pred = [e for e in pred if e.behavior in behaviors]
        hours = sum(seen.values()) / 3600
        print(f"Quan sát: {len(seen)} HS/track, {hours:.2f} HS-giờ")
        if args.dump:
            dump_episodes(Path(args.dump), pred)
            print(f"Đã ghi {len(pred)} đợt dự đoán → {args.dump}")
        if labels is None:
            counts = Counter(e.behavior for e in pred)
            secs = Counter()
            for e in pred:
                secs[e.behavior] += e.length
            print(f"\n(Chưa có nhãn — chỉ thống kê dự đoán)\n{'Hành vi':<22}{'Số đợt':>8}{'Tổng giây':>11}{'Đợt/HS-giờ':>12}")
            for b in behaviors:
                if counts[b]:
                    print(f"{b:<22}{counts[b]:>8}{secs[b]:>11.0f}{counts[b] / max(hours, 1e-9):>12.2f}")
            return
        print()
        metrics = score_behaviors(pred, labels, seen, behaviors, args.min_overlap)
        print_table(metrics)
        p, r, f, fa = total(metrics)
        pct = lambda v: "-" if v is None else f"{v * 100:.1f}%"  # noqa: E731
        print(f"\nTổng: precision {pct(p)}, recall {pct(r)}, F0.5 {pct(f)}, báo nhầm {fa:.2f}/HS-giờ")
        return

    if labels is None:
        raise SystemExit("--grid cần file nhãn (--labels).")
    keys = list(grid)
    rows = []
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"Duyệt {len(combos)} bộ tham số...")
    for combo in combos:
        cfg = make_config({**base, **dict(zip(keys, combo))})
        pred, seen = replay(samples, cfg, args.gap)
        pred = [e for e in pred if e.behavior in behaviors]
        p, r, f, fa = total(score_behaviors(pred, labels, seen, behaviors, args.min_overlap))
        rows.append((f if f is not None else -1, p, r, fa, combo))
    rows.sort(key=lambda x: (-x[0], x[3]))
    print(f"\n{'F0.5%':>7}{'Prec%':>7}{'Rec%':>7}{'Nhầm/h':>8}  " + "  ".join(keys))
    for f, p, r, fa, combo in rows[: args.top]:
        print(f"{_fmt(f if f >= 0 else None):>7}{_fmt(p):>7}{_fmt(r):>7}{fa:>8.2f}  " + "  ".join(combo))
    best = dict(zip(keys, rows[0][4]))
    print("\nTốt nhất:", " ".join(f"{k}={v}" for k, v in best.items()))
    print("Ghi các giá trị này vào BehaviorConfig trong config.py để dùng khi chạy web.")


if __name__ == "__main__":
    main()
