# Báo cáo rà soát Giai đoạn 1 (theo CHINHCODE.docx)

Ngày khóa mã: 2026-10-31  
Thư mục làm việc: thư mục gốc dự án  
Minh chứng: `evidence/`

## 1. Bảng tổng hợp

| STT | Nội dung rà soát | Trạng thái ban đầu | Cách xử lý | Minh chứng |
|-----|------------------|--------------------|------------|------------|
| 1 | ArcFace margin | `near_ok` / `margin >= 0.0` làm mất kiểm tra chênh lệch | `_accept_match`: `lock_sim` hoặc `min_margin`; gallery 1 HS riêng | `core/face_gallery.py` · unit AF01–AF06 |
| 2 | Chuyển ID ByteTrack | `_transfer_lock` theo IoU | `IOU_TRANSFER_LOCK=False` → `id_pending` + ArcFace xác minh | `core/identity.py` · unit ID01–ID06 |
| 3 | Thiếu keypoint | Bỏ qua / không `decay` | Decay + `Không xác định`; `pose_status` | `core/analysis.py` · KP01–KP03 |
| 4 | Thời gian hành vi | Trộn thời gian video / quan sát / xử lý; cộng khoảng mất dấu | Phân biệt 3 loại thời gian; đóng đợt khi gap > `EPISODE_GAP_SEC`; không backdate | `analysis.py`, `behavior_log.py` · T01–T05, E06 |
| 5 | Gộp hành vi theo HS | Đa số phiếu cả track | `student_id` theo sự kiện; đổi SID → tách đợt | `behavior_log.py` · E07 |
| 6 | Điểm danh | 3 quan sát ≠ 3 lần ArcFace | Ghi cả `identity_observations` và `face_match_events`; chế độ mặc định giữ gốc | Excel + E08–E10 |
| 7 | Cấu hình thực nghiệm | Phân tán | JSON + SHA-256 + `pip freeze` | `evidence/config/` |
| 8 | Thuật ngữ % | Dễ hiểu nhầm “độ chính xác” | “Điểm tương đồng quy đổi (%)” trên UI/Excel | `templates/index.html` |

## 2. Kết quả kiểm thử tự động

```text
python -m tools.run_stage1_tests
→ 30/30 passed
```

File: `evidence/tests/stage1_unit_results.json`

## 3. Câu trả lời chốt Giai đoạn 1

| Câu hỏi | Trả lời ngắn |
|---------|--------------|
| Mô hình nào đang chạy? | YOLOv12 classroom `best.pt` + YOLO11n-pose + InsightFace buffalo_l + gallery 33 HS — xem `experiment_config_stage1.json` |
| Ngưỡng nào đang dùng? | ArcFace từ `meta.json` / UI; hành vi từ `BehaviorConfig` + `behavior_settings.json` — đã khóa trong JSON |
| HS được xác nhận thế nào? | ArcFace đạt tương thích + (`lock_sim` hoặc `min_margin`); ByteTrack đổi ID chỉ đề cử, khóa khi ArcFace xác minh lại |
| Hành vi xác định thế nào? | Hình học COCO-17 + ngưỡng thời gian quan sát hợp lệ (`1/fps_eff`, trần 1s/bước) |
| Excel lấy dữ liệu từ đâu? | `BehaviorRecorder`: sự kiện gắn `student_id` tại thời điểm; thời lượng = `(last−start)/FPS` video |

## 4. Việc còn làm trên video thật (checklist)

Không thể tự động hóa hết trong môi trường không có video lớp — điền:

- `evidence/tests/checklist_R01_R09.md`
- Chạy đánh giá ArcFace trên ảnh test (không dùng ảnh enroll):
  ```bash
  python -m tools.eval_arcface --dataset classroom-faces-lop07
  ```
  → so sánh đúng / nhầm / từ chối trước–sau trong `margin_rule_compare`
- Giữ cố định ngưỡng hành vi khi đo kết quả cuối; chỉ chỉnh trên tập phát triển

## 5. Lệnh tái lập nhanh

```bash
python -m tools.run_stage1_tests
python -m tools.save_experiment_config
python -m tools.eval_arcface --dataset classroom-faces-lop07   # cần InsightFace
```

## 6. Ghi chú luận văn

- `HARD_STICKY_TTL = 900` tính theo **số lần cập nhật track**, không mặc định = 15 phút.
- Điểm tương đồng quy đổi (%) ≠ xác suất nhận diện đúng.
- Thời gian quan sát hợp lệ có thể ngắn hơn thời gian video khi bỏ khung (`fps_eff` sàn 1 Hz).
