# Mẫu đoạn mô tả phương pháp (Phụ lục C — có thể đưa vào luận văn)

Nghiên cứu đề xuất phương pháp phân tích biểu hiện tham gia hoạt động học tập dựa trên dữ liệu tư thế và hành vi quan sát được từ camera lớp học. Các hành vi được trích xuất bằng mô hình phát hiện, theo dõi đối tượng và ước lượng tư thế, sau đó được diễn giải theo bối cảnh hoạt động A1–A5 do người quan sát xác nhận. Mỗi phiên học được chia thành các cửa sổ thời gian và phân loại thành bốn trạng thái: E (có bằng chứng thực hiện nhiệm vụ), R (có tín hiệu cần xem lại), N (quan sát được nhưng chưa đủ bằng chứng) và U (không đủ dữ liệu quan sát). Kết quả được tổng hợp thành thời lượng, tỷ lệ từng trạng thái và độ bao phủ dữ liệu. Phương pháp không đồng nhất hành vi quan sát được với trạng thái tập trung nhận thức; chất lượng suy luận được đánh giá bằng dữ liệu đối chứng độc lập do người quan sát gán nhãn. Chỉ số có trọng số, nếu được sử dụng, chỉ đóng vai trò mô tả bổ sung và không thay thế cho các kết quả kiểm chứng theo từng trạng thái.

## Bổ sung kỹ thuật (triển khai trong dự án)

- Phát hiện / theo dõi: YOLOv12 + ByteTrack; tư thế: YOLO11-Pose; danh tính: ArcFace.
- Hành vi thô B1–B5 tách khỏi nhãn cửa sổ E/R/N/U (`core/behavior.py` → `core/ernu.py`).
- Cửa sổ mặc định 10 giây không chồng; tách tại mốc đổi A1–A5.
- Chỉ số chính: \(T_E, T_R, T_N, T_U\), \(C = T_{valid}/T\), \(P_E, P_R, P_N\) trên \(T_{valid}\), \(P_U\) trên \(T\).
- BEEI là chỉ số phụ; báo cáo kèm độ nhạy \(w_N \in \{0, 0{,}5, 1\}\).
- GT độc lập: tầng 1 `BEHAVIOR_GT`, tầng 2 `ENGAGEMENT_GT` (ON/OFF/UNCERTAIN).
