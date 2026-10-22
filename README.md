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

## Dashboard giáo viên

Kết quả được tổng hợp và hiển thị trên giao diện Dashboard phục vụ giáo viên theo dõi sĩ số
và mức độ chú ý. Tab **Lớp học** ở thanh bên cập nhật mỗi giây, tính trên cả phiên (không chỉ khung hiện tại):

- **Sĩ số**: có mặt / tổng số học sinh trong gallery ArcFace. Một học sinh được tính có mặt khi
  ArcFace nhận ra mã HS đó ở ít nhất 3 lần phân tích. Học sinh chưa nhận ra được nằm ở mục
  **Vắng / chưa nhận diện**; số người trên khung chưa nhận diện được mặt hiện riêng bên dưới.
- **Mức chú ý cả lớp**: điểm chú ý trung bình cả phiên, % thời gian tập trung (điểm ≥ ngưỡng),
  biểu đồ điểm trung bình cả lớp theo từng phút (cột đỏ = dưới ngưỡng).
- **Từng học sinh có mặt** (chú ý thấp xếp trước): điểm TB, điểm thấp nhất, % thời gian tập trung,
  giờ vào khung, 2 hành vi mất tập trung kéo dài nhất, chấm xanh = đang trên khung.

Số liệu này cùng nguồn với báo cáo Excel bên dưới.

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

- **Điểm danh theo HS** (sheet đầu): đầu trang là thông tin buổi học (lớp, video, ngày, sĩ số,
  có mặt, vắng, tỉ lệ chuyên cần, điểm chú ý cả lớp); bên dưới mỗi học sinh trong danh sách lớp
  là 1 hàng — STT, mã HS, họ tên, lớp, trạng thái Có mặt/Vắng, giờ điểm danh (lần đầu nhận diện),
  lần cuối thấy, số phút có mặt, số lần nhận diện, độ khớp ArcFace TB / cao nhất, mức tin cậy
  (Cao / Trung bình / Thấp — cần kiểm tra), ảnh khuôn mặt chụp trong lớp, điểm chú ý TB / thấp nhất,
  % thời gian tập trung / mất tập trung, hành vi mất tập trung nhiều nhất, số lần + số giây từng
  hành vi, cột ghi chú để giáo viên điền. Lớp lấy từ `students.csv` (HS không có trong file lấy lớp
  phổ biến nhất).
- **Chi tiết hành vi**: mỗi dòng 1 đợt hành vi của 1 học sinh — ngày, giờ bắt đầu,
  giờ kết thúc, thời lượng, điểm thấp nhất, vị trí trong video
- **Track chưa nhận diện**: người trên khung chưa nhận ra được mặt (ID theo dõi tạm) và hành vi của họ
- **Điểm theo phút**: điểm TB của từng học sinh theo từng phút
- **Thông tin**: nguồn video, nguồn thời gian, đoạn đã phân tích
- **Chú thích**: nhãn viết tắt trên video (Cúi lâu, Ngủ, Thẫn, Quay, Cúi, Nghiêng, Tay, Nói, Viết, Giơ tay),
  điều kiện nhận biết, điểm trừ, ý nghĩa màu khung

Bật/tắt từng hành vi: tab **Hành vi** ở thanh bên (công tắc từng dòng, nút Bật tất cả / Tắt tất cả).
Áp dụng ngay cho video đang chạy, lưu vào `behavior_settings.json` cho lần chạy sau
(mặc định tắt "Nói chuyện"). Hành vi tắt thì không trừ điểm, không gắn nhãn, không ghi vào Excel.

Ngưỡng ArcFace: tab **Nhận diện** ở thanh bên có ba thanh trượt — ngưỡng nhận diện (% giống tối thiểu để gắn tên),
ngưỡng khóa cứng và chênh lệch tối thiểu so với người khớp nhì. Bấm **Áp dụng** thì có hiệu lực ngay cho video đang chạy
(nâng ngưỡng sẽ bỏ tên các em đang khớp dưới mức mới) và được lưu vào `meta.json` của gallery. Dựng lại gallery hay nạp học sinh mới
vẫn giữ nguyên ngưỡng đã chỉnh.

Trên dashboard: tab **Hành vi** ở thanh bên cũng là bảng chú thích, nút **Chú thích** (phím `L`) hiện bảng chú thích đè lên video
(dùng được cả khi toàn màn hình); rê chuột vào nhãn của học sinh để xem tên hành vi đầy đủ.

Ngày giờ lấy từ dòng ngày giờ camera in trên hình (đọc bằng EasyOCR, chỉnh lại mỗi 60 giây video),
nên khớp với giờ hiển thị trên video. Ngày ghi đúng định dạng như trên video.
Không đọc được: camera trực tiếp dùng đồng hồ máy, file video dùng vị trí trong video (00:01:23).
Ngày kiểu `10-07-2026` không phân biệt được ngày/tháng — mặc định ngày-tháng-năm,
đổi bằng `set OSD_DATE_ORDER=MDY` (chỉ ảnh hưởng khi video qua nửa đêm / sang tháng).

## Luật hành vi

- Mọi khoảng cách chia cho **bề rộng vai**, nên học sinh ngồi xa hay gần camera dùng chung ngưỡng.
- **Cúi đầu** so với tư thế ngồi thẳng thường ngày của chính học sinh đó (p75 độ cao mũi trên vai).
  Khi chưa đủ khoảng 20 lần phân tích cho em đó thì dùng ngưỡng chung `head_down_ratio`.
  Nếu InsightFace thấy mặt, góc cúi (pitch) cũng được tính.
- **Quay đầu** dùng góc yaw của InsightFace (không thấy mặt thì dùng mũi và hai tai), đo so với hướng nhìn
  thường ngày của em đó. Lý do: camera đặt lệch nên học sinh ngồi hai bên lúc nhìn bảng đã lệch khoảng ±30°.
- **Nói chuyện** (thử nghiệm, mặc định tắt): hai em ngồi cạnh nhau cùng hàng, em bên trái quay sang phải và em bên phải
  quay sang trái, giữ ≥ `chat_duration` giây. Nếu thấy được miệng mà miệng không động thì không tính.
- Mỗi hành vi kéo dài đều có ngưỡng thời gian, và điều kiện sai ngắn hơn `miss_tolerance_sec` (do pose chập chờn) không làm mất đợt.
  Điểm được làm mượt theo thời gian (`score_smoothing`).

## Đo và chỉnh ngưỡng hành vi

Mỗi phiên web ghi keypoint + tín hiệu mặt vào `logs/keypoints/<ngày giờ>_<phiên>.jsonl`
(tắt bằng `set RECORD_KEYPOINTS=0`). Chấm điểm chỉ dựa vào các tín hiệu này nên chạy lại offline được y hệt,
thử hàng trăm bộ ngưỡng trong vài giây mà không phải chạy lại YOLO.

1. Tạo log cho video cần gán nhãn (không cần mở web). ArcFace chạy mỗi bước nên chậm, khoảng 5 s/bước trên CPU:

   ```bash
   python extract_keypoints.py --video static/uploads/x.mp4 --start 0 --duration 600 --step 0.5
   ```

2. Gán nhãn vào `labels/<tên log>.csv` theo mẫu `labels/mau_nhan.csv`, gồm các cột `student_id,behavior,start,end,note`.
   Thời gian là vị trí trong video (giây hoặc `mm:ss` / `hh:mm:ss`); mã HS là `HSxxx`, track chưa nhận diện thì ghi `ID<track>`.
   Cần gán **đủ mọi hành vi của mọi học sinh** trong đoạn đã chọn, vì đợt dự đoán không khớp nhãn nào sẽ bị tính là báo nhầm.
   Để có bản nháp, chạy `--dump`, sau đó sửa và xóa những dòng sai:

   ```bash
   python eval_behavior.py --log logs/keypoints/x.jsonl --dump labels/x.csv
   ```

3. Đo, rồi dò ngưỡng (ưu tiên precision qua F0.5, đặc biệt với Ngủ gục / Cúi đầu lâu):

   ```bash
   python eval_behavior.py --log logs/keypoints/x.jsonl --labels labels/x.csv
   python eval_behavior.py --log logs/keypoints/x.jsonl --labels labels/x.csv \
       --grid head_down_drop=0.3,0.4,0.5 --grid head_down_duration=3,4,5 --behaviors "Cúi đầu lâu"
   ```

   Một đợt dự đoán được tính là khớp nhãn khi cùng HS, cùng hành vi và phần chồng lên nhau ≥ 30% đợt ngắn hơn.
   Bộ tham số tốt nhất thì ghi vào `BehaviorConfig` trong `config.py`.

## Màu box

- Xanh lá: tập trung (điểm ≥ 50)
- Đỏ: mất tập trung (đỏ dày = cúi đầu lâu)
- Xanh dương nhạt: chưa lấy được tư thế (pose)

Nhãn: `HS012 85 · Cúi đầu lâu` = mã học sinh, điểm, hành vi chính (tên đầy đủ). Chưa nhận diện được thì hiện `ID<track>`.

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

Đánh giá nhận diện (gallery dựng từ ảnh `enroll`, kiểm tra trên ảnh `test`, kết quả ở `reports/arcface_eval_<dataset>.json`):

```bash
python eval_arcface.py --dataset classroom-faces-lop07
```

Đổi về gallery cũ:

```bash
set CLASSROOM_FACES_DIR=classroom-faces-lop07
set ARCFACE_MODEL_DIR=models\arcface_classroom-faces-lop07
python app.py
```
