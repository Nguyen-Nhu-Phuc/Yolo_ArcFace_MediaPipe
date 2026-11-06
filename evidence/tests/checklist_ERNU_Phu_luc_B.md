# Phiếu kiểm tra trước khi chạy kiểm thử cuối (Phụ lục B — bộ tiêu chí ERNU)

Đánh dấu khi hoàn thành. Khóa nhãn đối chứng và quy tắc **trước** khi xem kết quả kiểm thử cuối.

| ☐ | Hạng mục |
|---|----------|
| ☐ | Đã xác định đủ hoạt động A1–A5 trong video hoặc đánh dấu rõ vùng chưa biết nhiệm vụ (`ACTIVITY_CONTEXT`, mã `CHUA_XAC_MINH` nếu chưa xác nhận). |
| ☐ | Đã lưu bộ trọng số YOLO, YOLO11-Pose, gallery ArcFace, ngưỡng và phiên bản mã nguồn (`experiment_config_*.json`, `ernu_rules_v1.json`). |
| ☐ | Đã sửa/kiểm tra chuyển danh tính khi ByteTrack đổi ID và cách ghi sự kiện theo `student_id` (không gộp theo `track_id`). |
| ☐ | Đã kiểm tra thời gian video khác với thời gian xử lý; không cộng khoảng thiếu bằng chứng vào T_E/T_R/T_N. |
| ☐ | Đã thống nhất cách gán nhãn B1–B5, ON/OFF/UNCERTAIN và cách xử lý che khuất (`visibility`). |
| ☐ | Đã khóa nhãn đối chứng và quy tắc trước khi xem kết quả kiểm thử cuối (`rules_version`, split `kiem_thu_cuoi`; file `ernu_rules_v1.json` được `ErnuConfig.load_rules` nạp khi chạy app). |
| ☐ | Đã có báo cáo độ bao phủ C, tỷ lệ N/U, proxy ON/PENDING (và OFF nếu có xác nhận) theo từng hoạt động (`eval_ernu` → block `proxy`, sheet `TONG_HOP_ERNU` / `QUALITY_REVIEW`). |
| ☐ | Đã hiểu: proxy `PENDING` (từ R) ≠ OFF; OFF chỉ sau `review-confirm` / xác nhận giáo viên. |
| ☐ | Đã bảo vệ dữ liệu cá nhân; chỉ giáo viên/người nghiên cứu được phép truy cập video, khuôn mặt và nhãn học sinh chưa thành niên. |

## Nhắc nguyên tắc (mục 1)

- Không đồng nhất tư thế bên ngoài với trạng thái chú ý / cảm xúc / năng lực.
- Không tự động xem quay đầu, cúi đầu, ngồi yên hoặc không giơ tay là mất tập trung.
- Không dùng E/R/N/U của chương trình làm nhãn đối chứng để tự đánh giá chính chương trình.
- Không công bố kết quả định lượng khi dữ liệu quá ít hoặc danh tính chưa đủ tin cậy (`publishable=0`).
- Không dùng chỉ số để xếp hạng, kỷ luật hoặc đánh giá bất lợi tự động đối với học sinh.
- Proxy ON/OFF/PENDING chỉ diễn giải theo nhiệm vụ quan sát được — không gọi là điểm tập trung nhận thức.

## Lệnh đối chiếu sau khi khóa GT

```bash
python -m tools.eval_ernu \
  --gt docs/Mau_Gan_Nhan_Hanh_Vi_Lop_Hoc.xlsx \
  --pred reports/bao_cao_hanh_vi_....xlsx \
  --out evidence/reports/eval_ernu_final.json \
  --quality-out evidence/reports/QUALITY_REVIEW_eval.xlsx
```
