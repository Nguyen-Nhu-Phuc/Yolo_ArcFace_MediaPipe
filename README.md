# YOLOv12 + ArcFace — theo dõi tập trung trong lớp

Web app (FastAPI + WebSocket), luồng giống `classroom_attention`:

```
Video tải lên / webcam / camera RTSP
  → server đọc khung (đúng tốc độ thật, chậm thì bỏ khung)
  → YOLOv12 phát hiện + ByteTrack theo dõi từng người
  → YOLO11-pose lấy 17 keypoint trên crop từng người
  → chấm điểm tập trung (cúi đầu, nhắm mắt, thẫn thờ, quay đầu, nghiêng người,
     tay bất thường, giơ tay, viết bài)
  → ArcFace (InsightFace buffalo_l) gắn tên học sinh — chạy ở luồng riêng
  → vẽ box + nhãn → đẩy ảnh JPEG + thống kê qua WebSocket
```

YOLOv12 không có trọng số pose chính thức, nên phần keypoint dùng YOLO11-pose
(`yolo11n-pose.pt`, tự tải lần đầu).

## Chạy

```bash
pip install -r requirements.txt
python app.py
```

Mở http://127.0.0.1:5000, rồi:
- bấm **Tải video lên** (tự phân tích ngay), hoặc
- nhập `0` (webcam), `rtsp://...` (camera IP) hay đường dẫn file rồi bấm **Kết nối**.

Video tải lên được lưu trong `static/uploads/` (không commit).

### Camera qua OBS (OBS Virtual Camera)

1. Mở OBS, dựng cảnh (camera / màn hình / RTSP…), bấm **Start Virtual Camera**.
2. Trên dashboard bấm **Làm mới camera** — danh sách tự chọn "OBS Virtual Camera".
3. Bấm **Kết nối**. Có thể gõ thẳng tên vào ô nguồn (`OBS Virtual Camera` hoặc `obs`).

Lấy hình từ EZVIZ Studio qua OBS tự động: bật OBS WebSocket (Tools → WebSocket Server Settings),
mở EZVIZ Studio ở Live View rồi chạy nền `python obs_ezviz.py`. Script tạo nguồn Window Capture,
cắt đúng vùng video, đặt OBS 1280x720 @ 15fps, bật Virtual Camera và tự căn lại khi cửa sổ EZVIZ
đổi kích thước. Cửa sổ EZVIZ phải mở (phóng to cho nét), không thu nhỏ xuống taskbar.

Camera theo số được mở qua DirectShow (OBS Virtual Camera chỉ có trên DirectShow),
số trong danh sách = số dùng ở ô nguồn.
Trang `/enroll`: nạp học sinh mới bằng webcam + test nhận diện/hành vi.

## Báo cáo Excel hành vi

Bấm **Xuất Excel** (lúc đang chạy hoặc sau khi xong). Khi video phát hết / bấm Dừng,
báo cáo cũng tự lưu vào `reports/bao_cao_hanh_vi_<ngày giờ>_<phiên>.xlsx`.

- **Chi tiết hành vi**: mỗi dòng 1 đợt hành vi của 1 học sinh — ngày, giờ bắt đầu,
  giờ kết thúc, thời lượng, điểm thấp nhất, vị trí trong video
- **Tổng hợp theo HS**: điểm TB, % thời gian mất tập trung, số lần + tổng giây từng hành vi
- **Điểm theo phút**: điểm TB của từng học sinh theo từng phút
- **Thông tin**: nguồn video, nguồn thời gian, đoạn đã phân tích
- **Chú thích**: nhãn viết tắt trên video (Cúi lâu, Ngủ, Thẫn, Quay, Cúi, Nghiêng, Tay, Nói, Viết, Giơ tay),
  điều kiện nhận biết, điểm trừ, ý nghĩa màu khung

Bật/tắt từng hành vi: tab **Hành vi** ở thanh bên (công tắc từng dòng, nút Bật tất cả / Tắt tất cả).
Áp dụng ngay cho video đang chạy, lưu vào `behavior_settings.json` cho lần chạy sau
(mặc định tắt "Nói chuyện"). Hành vi tắt thì không trừ điểm, không gắn nhãn, không ghi vào Excel.

Trên dashboard: tab **Hành vi** ở thanh bên cũng là bảng chú thích, nút **Chú thích** (phím `L`) hiện bảng chú thích đè lên video
(dùng được cả khi toàn màn hình); rê chuột vào nhãn của học sinh để xem tên hành vi đầy đủ.

Ngày giờ lấy từ dòng ngày giờ camera in trên hình (đọc bằng EasyOCR, chỉnh lại mỗi 60 giây video),
nên khớp với giờ hiển thị trên video. Ngày ghi đúng định dạng như trên video.
Không đọc được: camera trực tiếp dùng đồng hồ máy, file video dùng vị trí trong video (00:01:23).
Ngày kiểu `10-07-2026` không phân biệt được ngày/tháng — mặc định ngày-tháng-năm,
đổi bằng `set OSD_DATE_ORDER=MDY` (chỉ ảnh hưởng khi video qua nửa đêm / sang tháng).

## Màu box

- Xanh lá: tập trung (điểm ≥ 50)
- Đỏ: mất tập trung (đỏ dày = cúi đầu lâu)
- Xanh dương nhạt: chưa lấy được tư thế (pose)

Nhãn: `HS12 85 · Cúi` = mã học sinh, điểm, lý do chính. Chưa nhận diện được thì hiện `ID<track>`.

## Model / biến môi trường

| Biến | Ý nghĩa | Mặc định |
|---|---|---|
| `YOLO_WEIGHTS` | `classroom` = YOLOv12n train cho lớp; `coco` = `yolo12n.pt` | `classroom` |
| `YOLO_MODEL_PATH` | Đường dẫn `.pt` YOLO bất kỳ (ghi đè) | — |
| `POSE_MODEL` | Model pose | `yolo11n-pose.pt` |
| `CLASSROOM_FACES_DIR` / `ARCFACE_MODEL_DIR` | Dataset / gallery ArcFace | `2_dataset_arcface` |

Ngưỡng hành vi và điểm nằm trong `config.py` (`BehaviorConfig`, `attention_threshold`).

## Dữ liệu

- Gallery mặc định: `models/arcface_2_dataset_arcface/gallery.npz`
- Bản cũ: `classroom-faces-lop07/` + `models/arcface_classroom-faces-lop07/`
- Weights YOLOv12 lớp học: `runs/detect/train/weights/best.pt`

Rebuild gallery:

```bash
python export_arcface_model.py --dataset 2_dataset_arcface
```

Đổi về gallery cũ:

```bash
set CLASSROOM_FACES_DIR=classroom-faces-lop07
set ARCFACE_MODEL_DIR=models\arcface_classroom-faces-lop07
python app.py
```
