"""Tạo / cập nhật workbook gán nhãn đối chứng theo Phụ lục A bộ tiêu chí ERNU.

Sheet bắt buộc:
  HUONG_DAN, VIDEOS, STUDENTS, ACTIVITY_CONTEXT, BEHAVIOR_GT,
  ENGAGEMENT_GT, PREDICTED_WINDOWS (trống — do chương trình điền), QUALITY_REVIEW

Chạy từ thư mục gốc dự án:
  python -m tools.make_annotation_workbook
  python -m tools.make_annotation_workbook --out docs/Mau_Gan_Nhan_Hanh_Vi_Lop_Hoc.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "Mau_Gan_Nhan_Hanh_Vi_Lop_Hoc.xlsx"

# Chia dữ liệu khởi tạo (mục 8.2): 2 lớp × 3 video; phát triển / hiệu chỉnh / kiểm thử cuối
SPLIT_VIDEOS = [
    ("L01_V01", "L01", "phat_trien", "Soạn định nghĩa, kiểm tra tình huống khó"),
    ("L01_V02", "L01", "hieu_chinh", "Chọn ngưỡng, xử lý xung đột"),
    ("L01_V03", "L01", "kiem_thu_cuoi", "Đánh giá với quy tắc đã khóa"),
    ("L02_V01", "L02", "phat_trien", "Soạn định nghĩa, kiểm tra tình huống khó"),
    ("L02_V02", "L02", "hieu_chinh", "Chọn ngưỡng, xử lý xung đột"),
    ("L02_V03", "L02", "kiem_thu_cuoi", "Đánh giá với quy tắc đã khóa"),
]


def _sheet(wb, title: str, headers: list[str], rows: list[list], widths: list[int] | None = None):
    from openpyxl.styles import Font

    ws = wb.create_sheet(title)
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in rows:
        ws.append(row)
    if widths:
        from openpyxl.utils import get_column_letter
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
    return ws


def build_workbook():
    from openpyxl import Workbook

    wb = Workbook()
    # Xóa sheet mặc định
    default = wb.active
    wb.remove(default)

    _sheet(
        wb, "HUONG_DAN",
        ["BƯỚC", "VIỆC LÀM", "VÍ DỤ / LƯU Ý"],
        [
            ["1", "Mở video gốc bằng VLC; kiểm tra âm/hình, thời lượng, FPS và mốc thời gian",
             "Không gán nhãn nếu giải mã lỗi / xám xanh"],
            ["2", "Tạo video_id và bảng student_id ẩn danh; chụp sơ đồ bàn",
             "Không dùng track_id làm nhãn gốc"],
            ["3", "Ghi hoạt động A1–A5 theo đoạn giáo viên giao nhiệm vụ",
             "Không biết nhiệm vụ → CHUA_XAC_MINH"],
            ["4", "Thí điểm: video 5 phút + 5 học sinh rõ; ghi sự kiện B1–B5",
             "B1 giơ tay; B2 viết; B3 quay đầu; B4 cúi lâu; B5 gục bàn"],
            ["5", "Gán ON/OFF/UNCERTAIN từng cửa sổ 10 giây + visibility",
             "Không tự xem ngồi yên là ON"],
            ["6", "Người thứ hai gán độc lập một phần; rà bất đồng; khóa bản GT",
             "Ghi QUALITY_REVIEW trước khi xem kết quả AI"],
            ["7", "Chạy chương trình với cấu hình đã lưu; xuất E/R/N/U; đối chiếu",
             "python -m tools.eval_ernu --gt ... --pred ..."],
            ["QUAN TRỌNG", "ON/OFF/UNCERTAIN ≠ E/R/N/U",
             "E/R/N/U là đầu ra thuật toán cần đánh giá, không phải GT"],
            ["NGUYÊN TẮC", "Không đồng nhất tư thế với tập trung nhận thức",
             "Không dùng kết quả để xếp hạng / kỷ luật học sinh"],
        ],
        [12, 70, 55],
    )

    _sheet(
        wb, "VIDEOS",
        ["video_id", "class_id", "filename", "duration_sec", "fps", "split",
         "quality_notes", "notes"],
        [
            [vid, cls, f"{vid}.mp4", 600, 15, split, "CHUA_XAC_MINH", note]
            for vid, cls, split, note in SPLIT_VIDEOS
        ],
        [14, 10, 18, 12, 8, 16, 18, 45],
    )

    _sheet(
        wb, "STUDENTS",
        ["student_id", "class_id", "seat_zone", "seat_row", "seat_col",
         "present_gt", "visible_notes", "real_name_private"],
        [
            ["S01", "L01", "truoc_trai", 1, 1, "CHUA_XAC_MINH",
             "Ví dụ cấu trúc — cập nhật sau khi xem", ""],
            ["S02", "L01", "truoc_giua", 1, 2, "CHUA_XAC_MINH", "", ""],
            ["S03", "L01", "sau_phai", 3, 4, "CHUA_XAC_MINH",
             "Vị trí xa camera — coverage thường thấp hơn", ""],
        ],
        [12, 10, 14, 10, 10, 16, 45, 18],
    )

    _sheet(
        wb, "ACTIVITY_CONTEXT",
        ["video_id", "start_sec", "end_sec", "activity_code", "task_note",
         "annotator_id", "notes"],
        [
            ["L01_V01", 0, 180, "A1", "Nghe giảng / quan sát bảng", "G01", ""],
            ["L01_V01", 180, 420, "A2", "Ghi chép / làm bài cá nhân", "G01", ""],
            ["L01_V01", 420, 600, "A4", "Hỏi đáp / phát biểu", "G01", ""],
            ["L01_V02", 0, 600, "CHUA_XAC_MINH", "Chưa xác nhận nhiệm vụ", "G01",
             "Không dùng đoạn này để diễn giải E cho đến khi xác nhận A"],
        ],
        [12, 10, 10, 14, 35, 12, 50],
    )

    _sheet(
        wb, "BEHAVIOR_GT",
        ["video_id", "student_id", "start_sec", "end_sec", "behavior_code",
         "visibility", "annotator_id", "confidence", "note"],
        [
            ["L01_V01", "S01", "", "", "CHUA_GAN", "CHUA_KIEM_TRA", "G01",
             "CHUA_DANH_GIA", "Nhập start/end khi xem video; B4 ≠ cúi lúc viết bài"],
        ],
        [12, 12, 10, 10, 14, 16, 12, 14, 55],
    )

    _sheet(
        wb, "ENGAGEMENT_GT",
        ["video_id", "student_id", "window_id", "start_sec", "end_sec",
         "activity_code", "gt_label", "visibility", "evidence", "annotator_id"],
        [
            ["L01_V01", "S01", 1, 0, 10, "A1", "CHUA_GAN", "CHUA_KIEM_TRA",
             "Ví dụ cửa sổ 10s — gt_label ∈ {ON, OFF, UNCERTAIN}", "G01"],
            ["L01_V01", "S01", 2, 10, 20, "A1", "CHUA_GAN", "VISIBLE", "", "G01"],
        ],
        [12, 12, 10, 10, 10, 14, 12, 16, 50, 12],
    )

    _sheet(
        wb, "PREDICTED_WINDOWS",
        ["video_id", "student_id", "start_sec", "end_sec", "activity_code",
         "behavior_events", "valid_coverage", "pred_label", "reason", "mixed",
         "rules_version"],
        [],
        [12, 12, 10, 10, 12, 22, 14, 10, 40, 8, 12],
    )

    _sheet(
        wb, "QUALITY_REVIEW",
        ["video_id", "student_id", "start_sec", "end_sec", "issue",
         "annotator_1", "annotator_2", "reviewer", "resolution", "note"],
        [
            ["L01_V01", "S01", "", "", "Chưa rà soát bất đồng",
             "G01", "G02", "", "CHUA_XU_LY",
             "Khóa GT trước khi xem kết quả kiểm thử cuối"],
        ],
        [12, 12, 10, 10, 35, 12, 12, 12, 14, 45],
    )
    return wb


def main() -> int:
    ap = argparse.ArgumentParser(description="Tạo workbook gán nhãn ERNU (Phụ lục A)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    wb = build_workbook()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f"Đã ghi {args.out}")
    print("Sheets:", wb.sheetnames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
