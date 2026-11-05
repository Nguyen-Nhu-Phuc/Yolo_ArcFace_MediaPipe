# YOLOv12 + ArcFace — theo dõi tập trung trong lớp

Web app (FastAPI + WebSocket), luồng giống `classroom_attention`:

```
Video tải lên / camera RTSP
  → server đọc khung (đúng tốc độ thật, chậm thì bỏ khung)
  → YOLOv12 phát hiện + ByteTrack theo dõi từng người
  → YOLO11-pose lấy 17 keypoint trên crop từng người
  → nhận biết hành vi (giơ tay, viết bài, quay đầu, cúi đầu lâu, gục xuống bàn;
     nghiêng người chỉ ghi tư thế; thẫn thờ không nhận biết tự động)
  → ArcFace (InsightFace buffalo_l) gắn tên học sinh — chạy ở luồng riêng
  → vẽ box + nhãn → đẩy ảnh JPEG + thống kê qua WebSocket
```

YOLOv12 không có trọng số pose chính thức, nên phần keypoint dùng YOLO11-pose
(`models/yolo11n-pose.pt`).

## Cấu trúc thư mục

```
1.Documents/        báo cáo, bài báo, luận văn mẫu
2.Data/             dataset khuôn mặt, nhãn hành vi (labels/), behavior_settings.json, cameras.json
3.Thesis/           phần mềm đang chạy
    app.py          máy chủ web — python app.py
    core/           cấu hình, phân tích khung, nhận diện, luật hành vi, báo cáo
    tools/          dựng gallery, đánh giá ArcFace, trích keypoint, đo hành vi
    models/         yolo12_classroom/, yolo11n-pose.pt, mediapipe/face_landmarker.task, arcface_*/
    templates/, static/
    logs/, reports/
    bao_cao/        script viết báo cáo tiến độ
```

Script trong `tools/` chạy từ thư mục `3.Thesis`, ví dụ `python tools/eval_behavior.py ...`.

## Chạy

```bash
cd 3.Thesis
pip install -r requirements.txt
python app.py
```

Trên máy khác (nhận dự án bằng tệp zip, cần Python 3.12): nhấp đúp `3.Thesis/cai_dat.bat` một lần, sau đó `3.Thesis/chay.bat` mỗi lần dùng.

Mở http://127.0.0.1:5000, rồi:
- bấm **Tải video lên** (tự phân tích ngay), hoặc
- dán `rtsp://...` (camera IP hoặc DDNS) hay đường dẫn file rồi bấm **Kết nối**.

Ứng dụng không mở webcam của máy tính.

### Camera từ xa (Port Forward + DDNS)

Tab **Camera** nhận hostname DDNS, cổng ngoài, tài khoản và đường dẫn stream, rồi ghép thành `rtsp://user:pass@your-ddns.ddns.net:15554/...`. Router phía camera phải chuyển cổng ngoài đó vào cổng 554 của camera; hostname DDNS trỏ tới IP public của router. Phần mềm không cấu hình router. OpenCV mở URL qua Internet và YOLO chạy trên máy này. Mật khẩu được lưu trong `2.Data/cameras.json` và bị che khi hiện trên giao diện.

Video tải lên chỉ được lưu tạm trong `static/uploads/` trong lúc phân tích và bị xóa khi phiên kết thúc
(video phát hết, bấm Dừng hoặc đóng trang); bản tạm sót lại được dọn khi khởi động server.
Muốn không chép video lên máy chủ thì dán đường dẫn file vào ô nguồn rồi bấm **Kết nối**.

## Dashboard giáo viên

Kết quả được tổng hợp cho giáo viên theo dõi sĩ số và những gì quan sát được.
Tab **Lớp học** cập nhật mỗi giây, tính trên cả phiên:

- **Sĩ số**: có mặt / tổng số học sinh trong gallery ArcFace. Một học sinh được tính có mặt khi
  ArcFace nhận ra mã HS đó ở ít nhất 3 lần phân tích. Học sinh chưa nhận ra được nằm ở mục
  **Vắng / chưa nhận diện**.
- **Chỉ số ERNU (cửa sổ 10s)**: độ bao phủ C; P_E / P_R / P_N trên thời gian quan sát hợp lệ; P_U trên toàn phiên; BEEI (phụ).
- **Tín hiệu thô B1–B5**: giơ tay, viết bài, quay đầu (trung tính → N), cúi đầu lâu / gục bàn (xem lại tùy A), thiếu pose.
- Biểu đồ theo phút là tỷ lệ khung có tín hiệu B4/B5.
- **Từng học sinh có mặt** (P_R / tín hiệu cần xem xếp trước): thời lượng và chỉ số ERNU.
- Không đồng nhất tư thế với tập trung nhận thức; không xếp hạng học sinh bằng chỉ số.

## Báo cáo Excel hành vi

Bấm **Xuất Excel** (lúc đang chạy hoặc sau khi xong). Khi video phát hết / bấm Dừng,
báo cáo cũng tự lưu vào `reports/bao_cao_hanh_vi_<ngày giờ>_<phiên>.xlsx`.

- **Điểm danh theo HS** (sheet đầu): đầu trang là thông tin buổi học (lớp, video, ngày, sĩ số,
  có mặt, vắng, tỉ lệ chuyên cần, hoạt động quan sát được, tư thế cần xem lại, thời gian không xác định);
  bên dưới mỗi học sinh trong danh sách lớp là 1 hàng — STT, mã HS, họ tên lấy từ cột `full_name` của `2.Data/classroom-faces-lop07/danh_sach_hoc_sinh.csv` (không có file này thì dùng `students.csv`; đối chiếu theo mã HS, ghi đè tên trong gallery), lớp, trạng thái Có mặt/Vắng,
  giờ điểm danh, lần cuối thấy, số phút có mặt, số lần nhận diện, độ khớp ArcFace, mức tin cậy,
  ảnh khuôn mặt, ba nhóm quan sát, số lần + số giây từng hành vi, cột ghi chú để giáo viên điền.
- **Chi tiết hành vi**: mỗi dòng 1 đợt — ngày, giờ bắt đầu, giờ kết thúc, thời lượng, nhóm
  (hoạt động quan sát được / cần giáo viên xem / không xác định / tư thế ghi nhận)
- **Track chưa nhận diện**: người trên khung chưa nhận ra được mặt và các nhóm ở trên
- **Thông tin** và **Chú thích**

Bật/tắt từng hành vi: tab **Hành vi**. "Không xác định" luôn bật. Mặc định tắt các hành vi đã ẩn
(nhắm mắt, nói chuyện, tay bất thường, cúi đầu thoáng qua). Thẫn thờ không còn được nhận biết tự động.
Nghiêng người vẫn ghi nhận nhưng không nằm trong nhóm cần xem.

Ngưỡng ArcFace: tab **Nhận diện** ở thanh bên có ba thanh trượt — ngưỡng nhận diện (% giống tối thiểu để gắn tên),
ngưỡng khóa cứng và chênh lệch tối thiểu so với người khớp nhì. Bấm **Áp dụng** thì có hiệu lực ngay cho video đang chạy
(nâng ngưỡng sẽ bỏ tên các em đang khớp dưới mức mới) và được lưu vào `meta.json` của gallery. Dựng lại gallery
vẫn giữ nguyên ngưỡng đã chỉnh.

Trên dashboard: tab **Hành vi** ở thanh bên cũng là bảng chú thích, nút **Chú thích** (phím `L`) hiện bảng chú thích đè lên video
(dùng được cả khi toàn màn hình); rê chuột vào nhãn của học sinh để xem tên hành vi đầy đủ.

Ngày giờ lấy từ dòng ngày giờ camera in trên hình (đọc bằng EasyOCR, chỉnh lại mỗi 60 giây video),
nên khớp với giờ hiển thị trên video. Ngày ghi đúng định dạng như trên video.
Không đọc được: camera trực tiếp dùng đồng hồ máy, file video dùng vị trí trong video (00:01:23).
Ngày kiểu `10-07-2026` không phân biệt được ngày/tháng — mặc định ngày-tháng-năm,
đổi bằng `set OSD_DATE_ORDER=MDY` (chỉ ảnh hưởng khi video qua nửa đêm / sang tháng).

## Luật hành vi

Mô hình pose giữ nguyên `yolo11n-pose.pt`, kích thước đầu vào 320. Phần dưới là hậu xử lý.
Các ngưỡng là điểm khởi đầu, cần đo lại trên video đã gán nhãn trước khi coi là cấu hình cuối.

- Điểm khớp có độ tin cậy dưới 0,5 không được dùng. Tọa độ đủ tin cậy được làm mượt bằng EMA
  (hệ số 0,6 cho giá trị mới). Mất một điểm quá 1 giây thì bỏ điểm cũ, không kéo dài hành vi.
- Mọi khoảng cách chia cho **bề rộng vai**. Tư thế nền của từng học sinh chỉ cập nhật từ khung
  đủ tin cậy và tương đối ổn định; tư thế đang cúi hoặc đang quay không được đưa vào mức tham chiếu.
- Thiếu vai hoặc không thấy điểm đầu (che khuất, quay lưng) → **Không xác định**.
- **Giơ tay**: cổ tay và khuỷu tay cao hơn vai, giữ ≥ 1 giây.
- **Viết bài**: cúi đầu, cổ tay ở khu vực bàn và tay đang chuyển động. Chỉ cúi đầu thì chưa phải viết.
- **Quay đầu**: so với hướng nhìn thường ngày của chính học sinh, giữ ≥ 2 giây.
- **Cúi đầu lâu**: ≥ 5 giây và không có bằng chứng viết bài.
- **Gục xuống bàn**: đầu thấp hoặc lệch sang vai, đồng thời thân nghiêng, giữ ≥ 5 giây.
  Không kết luận học sinh đang ngủ. Khi đang viết bài thì không gắn nhãn này và không gắn cúi đầu lâu.
- **Nghiêng người**: chỉ ghi tư thế. **Thẫn thờ** không nhận biết tự động.

## Đo và chỉnh ngưỡng hành vi

Mỗi phiên web ghi keypoint + tín hiệu mặt vào `logs/keypoints/<ngày giờ>_<phiên>.jsonl`
(tắt bằng `set RECORD_KEYPOINTS=0`). Luật hành vi chỉ dựa vào các tín hiệu này nên chạy lại offline được y hệt,
thử hàng trăm bộ ngưỡng trong vài giây mà không phải chạy lại YOLO.

1. Tạo log cho video cần gán nhãn (không cần mở web). ArcFace chạy mỗi bước nên chậm, khoảng 5 s/bước trên CPU:

   ```bash
   python tools/extract_keypoints.py --video D:\video\tiet3.mp4 --start 0 --duration 600 --step 0.5
   ```

2. Gán nhãn vào `2.Data/labels/<tên log>.csv` theo mẫu `2.Data/labels/mau_nhan.csv`, gồm các cột `student_id,behavior,start,end,note`.
   Thời gian là vị trí trong video (giây hoặc `mm:ss` / `hh:mm:ss`); mã HS là `HSxxx`, track chưa nhận diện thì ghi `ID<track>`.
   Cần gán **đủ mọi hành vi của mọi học sinh** trong đoạn đã chọn, vì đợt dự đoán không khớp nhãn nào sẽ bị tính là báo nhầm.
   Để có bản nháp, chạy `--dump`, sau đó sửa và xóa những dòng sai:

   ```bash
   python tools/eval_behavior.py --log logs/keypoints/x.jsonl --dump ../2.Data/labels/x.csv
   ```

3. Đo precision, recall, F0.5 và số lần báo nhầm. Với ít video, kết quả chỉ là kiểm chứng bước đầu:

   ```bash
   python tools/eval_behavior.py --log logs/keypoints/x.jsonl --labels ../2.Data/labels/x.csv
   python tools/eval_behavior.py --log logs/keypoints/x.jsonl --labels ../2.Data/labels/x.csv \
       --grid head_down_drop=0.3,0.4,0.5 --grid head_down_duration=5,6,7 --behaviors "Cúi đầu lâu"
   ```

   Một đợt dự đoán được tính là khớp nhãn khi cùng HS, cùng hành vi và phần chồng lên nhau ≥ 30% đợt ngắn hơn.
   Bộ tham số tốt nhất thì ghi vào `BehaviorConfig` trong `core/config.py`.

## Bộ tiêu chí ERNU (E/R/N/U) + proxy nhiệm vụ

Diễn giải hành vi thô theo ngữ cảnh A1–A5 → nhãn cửa sổ E/R/N/U (`core/ernu.py`).
Quy tắc nạp lúc chạy từ `03_EVIDENCE_STAGE1/config/ernu_rules_v1.json` (hoặc `ERNU_RULES_PATH`).

**Proxy vận hành** (không phải tập trung nhận thức): `E→ON`, `R→PENDING`, `N→UNCERTAIN`, `U→NO_DATA`.
`OFF` chỉ sau xác nhận giáo viên (`POST /api/ernu/review-confirm` hoặc form tab **Ngữ cảnh A**). Không tự gán OFF từ R.

```bash
# Nạp bảng ACTIVITY_CONTEXT (A1–A5 do giáo viên xác nhận)
set ERNU_ACTIVITY_CONTEXT=..\Mau_Gan_Nhan_Hanh_Vi_Lop_Hoc.xlsx
# Tuỳ chọn: đường dẫn rules JSON
set ERNU_RULES_PATH=03_EVIDENCE_STAGE1\config\ernu_rules_v1.json

# Tạo / cập nhật workbook gán nhãn đối chứng (Phụ lục A)
python -m tools.make_annotation_workbook

# Đối chiếu GT độc lập (không dùng pred làm GT) — có block proxy trong JSON
python -m tools.eval_ernu --gt ../Mau_Gan_Nhan_Hanh_Vi_Lop_Hoc.xlsx --pred reports/bao_cao_....xlsx
```

UI: tab **Ngữ cảnh A** để thêm đoạn A1–A5 / `CHUA_XAC_MINH`; tab **Lớp học** hiện P_ON / P_PENDING / P_OFF.
Excel: `PREDICTED_WINDOWS` có cột `proxy_label` / `confirmed_off`; `TONG_HOP_ERNU` có T_*/P_* proxy.
Checklist trước kiểm thử cuối: `03_EVIDENCE_STAGE1/tests/checklist_ERNU_Phu_luc_B.md`.

## Màu box

- Đỏ, viền dày: tín hiệu cần xem (B4/B5 — cúi đầu lâu, gục xuống bàn); diễn giải R tùy A
- Xanh dương: tín hiệu hoạt động (B1/B2 — giơ tay, viết bài); diễn giải E tùy A
- Xám: có tư thế, không gắn cảnh báo (ngồi bình thường → N)
- Xanh lá nhạt trên chú thích: quay đầu (B3) — trung tính → N theo ERNU
- Vàng: nghiêng người (chỉ ghi tư thế)
- Xanh nhạt: chưa đủ tư thế hoặc không xác định (→ U khi coverage thấp)

Nhãn trên khung: `1 · Cúi đầu lâu` = số thứ tự trong danh sách học sinh (HS7_01 là 1, đến HS7_33 là 33) và hành vi chính. File Excel vẫn ghi mã HS7_01. Chưa nhận diện được thì hiện `ID<track>`.

## Model / biến môi trường

| Biến | Ý nghĩa | Mặc định |
|---|---|---|
| `YOLO_WEIGHTS` | `classroom` = YOLOv12n train cho lớp; `coco` = `yolo12n.pt` | `classroom` |
| `YOLO_MODEL_PATH` | Đường dẫn `.pt` YOLO bất kỳ (ghi đè) | — |
| `POSE_MODEL` | Model pose | `models/yolo11n-pose.pt` |
| `ERNU_ACTIVITY_CONTEXT` | File `.xlsx`/`.json` bảng A1–A5 | (trống → mặc định A1) |
| `ERNU_RULES_PATH` | File quy tắc ERNU JSON | `03_EVIDENCE_STAGE1/config/ernu_rules_v1.json` |
| `CLASSROOM_FACES_DIR` / `ARCFACE_MODEL_DIR` | Dataset / gallery ArcFace | `2.Data/2_dataset_arcface` |

Ngưỡng hành vi nằm trong `core/config.py` (`BehaviorConfig`).

## Dữ liệu

- Gallery mặc định: `models/arcface_2_dataset_arcface/gallery.npz`
- Bản cũ: `2.Data/classroom-faces-lop07/` + `models/arcface_classroom-faces-lop07/`
- Weights YOLOv12 lớp học: `models/yolo12_classroom/best.pt` (kèm `results.csv`, `args.yaml` của lần huấn luyện)

Rebuild gallery:

```bash
python tools/export_arcface_model.py --dataset 2_dataset_arcface
```

Đánh giá nhận diện (gallery dựng từ ảnh `enroll`, kiểm tra trên ảnh `test`, kết quả ở `reports/arcface_eval_<dataset>.json`):

```bash
python tools/eval_arcface.py --dataset classroom-faces-lop07
```

Đổi về gallery cũ:

```bash
set CLASSROOM_FACES_DIR=2.Data\classroom-faces-lop07
set ARCFACE_MODEL_DIR=models\arcface_classroom-faces-lop07
python app.py
```
