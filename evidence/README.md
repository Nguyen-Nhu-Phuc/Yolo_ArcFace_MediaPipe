# Minh chứng Giai đoạn 1

| Thư mục | Nội dung |
|---------|----------|
| `config/` | `packages_stage1.txt`, `experiment_config_stage1.json`, `source_manifest_stage1.json` |
| `tests/` | Kết quả unit (gồm suite `ernu`) + checklist R01–R09 (điền tay trên video) |
| `reports/` | Báo cáo rà soát |
| `logs/`, `screenshots/` | Chỗ gắn nhật ký / ảnh chụp khi chạy video thật |

Mã nguồn đang review: `core/`.

## Lớp ERNU (bộ tiêu chí đầy đủ)

| Thành phần | Vị trí |
|------------|--------|
| Diễn giải cửa sổ 10s → E/R/N/U + C/P_E/P_R/P_N/P_U + BEEI | `core/ernu.py`, `ErnuConfig` |
| Ma trận A1–A5 × B1–B5 (đã khóa) | `config/ernu_rules_v1.json` |
| Ngữ cảnh hoạt động | `core/activity_context.py` (`ERNU_ACTIVITY_CONTEXT` hoặc `/api/ernu/activity-context`) |
| Excel phiên | `PREDICTED_WINDOWS`, `TONG_HOP_ERNU`, `DIEN_GIAI_ERNU` |
| Workbook GT (Phụ lục A) | `docs/Mau_Gan_Nhan_Hanh_Vi_Lop_Hoc.xlsx` — tạo lại: `python -m tools.make_annotation_workbook` |
| Đối chứng (Phụ lục 8.4) | `python -m tools.eval_ernu --gt ... --pred ... [--quality-out ...]` |
| Phiếu kiểm tra cuối (Phụ lục B) | `tests/checklist_ERNU_Phu_luc_B.md` |
| Đoạn phương pháp luận văn (Phụ lục C) | `reports/phuong_phap_ERNU_Phu_luc_C.md` |

Nguyên tắc: không đồng nhất tư thế với tập trung nhận thức; không dùng E/R/N/U hệ thống làm GT của chính nó; không xếp hạng / kỷ luật học sinh bằng chỉ số.
