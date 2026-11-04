# Checklist hồi quy R01–R09 (CHINHCODE §9)

Điền khi chạy trên video cố định + cùng cấu hình đã khóa (`experiment_config_stage1.json`).

| Mã | Nội dung | Điều kiện đạt | Kết quả (Đạt/Chưa) | Ghi chú / minh chứng |
|----|----------|---------------|--------------------|----------------------|
| R01 | Khởi động hệ thống và nạp mô hình | Không lỗi khi mở app / nạp YOLO+Pose+ArcFace | | |
| R02 | Phát hiện người trên video | Lớp người = 0 (person), box hợp lệ | | |
| R03 | Nhận diện học sinh | Không lỗi so khớp; đối chiếu thủ công đúng/sai/từ chối | | Bảng ArcFace |
| R04 | Đổi ID ByteTrack | Không tráo tên (ngồi yên, che, cắt ngang, đổi chỗ) | | Video + log ID |
| R05 | Mất keypoint | Hiện/ghi `Không xác định` / `pose_status=no_keypoints` | | `logs/keypoints` |
| R06 | Giơ tay, viết bài, quay đầu, cúi đầu | Cảnh báo khớp tình huống đã gán nhãn | | |
| R07 | Bỏ khung (1 / 5 / 25) | Thời gian video vs quan sát hợp lệ đúng định nghĩa đã chốt | | Bảng thời gian |
| R08 | Xuất Excel | Đúng mã, tên, sự kiện, thời lượng; không gộp nhầm lịch sử | | `bao_cao_hanh_vi.xlsx` |
| R09 | Chạy lại cùng video + cấu hình | Kết quả nhất quán | | |

## Bảng tình huống identity (R04)

| Tình huống | Mong muốn | Track trước/sau | HS trước/sau | Kết quả |
|------------|-----------|-----------------|--------------|---------|
| Ngồi yên, ID không đổi | Giữ tên | | | |
| Che 2–3 giây rồi hiện lại | Khôi phục nếu ArcFace xác minh | | | |
| Hai HS đi ngang | Không tráo tên | | | |
| HS08 rời, HS09 ngồi vào | Không chuyển tên HS08 sang HS09 | | | |
| Mất tín hiệu camera rồi nối lại | Không gán danh tính cũ thiếu căn cứ | | | |

## Bảng thời gian bỏ khung (R07)

Video: ________  FPS: ________

| Chế độ | Cảnh báo bắt đầu | Kết thúc | Số lần nhận diện | Tổng thời lượng báo cáo |
|--------|------------------|----------|------------------|-------------------------|
| Mọi khung | | | | |
| Cách 5 khung | | | | |
| Cách 25 khung | | | | |

Nhắc: thời gian video ≠ thời gian quan sát hợp lệ (`fps_eff`) ≠ thời gian xử lý (ms).
