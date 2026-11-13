# CVAT Job Runner

CVAT Job Runner là công cụ Windows giúp đề xuất nhãn cho ảnh 2D trong một job CVAT. Model chạy trên máy của bạn. Bạn xem kết quả trước khi quyết định lưu lên CVAT.

## Chuẩn bị lần đầu

1. Tải repository về máy và mở PowerShell trong thư mục chứa `Mo_CVAT_Job_Runner.cmd`.
2. Cài Python 3.10 trở lên có Tkinter. Cài thư viện bằng lệnh sau.

   ```powershell
   python -m pip install torch "transformers>=5,<6" ultralytics opencv-python numpy Pillow
   ```

3. Chuẩn bị model theo chế độ bạn dùng.

   | Nhu cầu | Model local |
   |---|---|
   | Rectangle, Polygon, Polyline hoặc Mask theo mô tả | SAM 3 trong `models/sam3/` |
   | Skeleton người | `models/yolo11n-pose.pt`. Job cần có nhãn Skeleton với các điểm con theo tên COCO như `nose` và `left_shoulder`. |
   | Chỉ Rectangle theo lớp COCO | `models/yolo11n.pt` |

Nếu cần SAM 3, xin quyền truy cập model bằng tài khoản Hugging Face rồi tải checkpoint khoảng 3,44 GB vào thư mục tool. [Trang model SAM 3](https://huggingface.co/facebook/sam3)

```powershell
python -m pip install huggingface_hub
hf auth login
hf download facebook/sam3 --exclude sam3.pt --local-dir models/sam3
```

## Lấy guideline cho job

- **Job đã có guideline:** Xin bản mới nhất từ người giao job hoặc người phụ trách review. Đối chiếu tên nhãn và kiểu shape với mục **Labels** của task hoặc project trên CVAT. Nếu file đã có khối `cvat-guideline` V2, có thể nạp trực tiếp.
- **Job chưa có guideline cho tool:** Tải hoặc sao chép [mẫu Markdown V2](data/Mau_Guideline_Khach_Hang_CVAT.md), lưu thành một file `.md` riêng cho job rồi điền tên nhãn, kiểu shape, mô tả và quy tắc của dự án. Nếu đã có tài liệu Word, lấy nội dung nghiệp vụ từ bản đó để điền vào mẫu. Tool cũng đọc được bảng Word có cột **CVAT shape** và **Class áp dụng**.
- **Muốn xem ví dụ đã điền:** Dùng ví dụ để tham khảo cách viết rồi đối chiếu với nhãn thật của job. [BBox và vạch đường giao thông](data/Guideline_Gan_Nhan_Giao_Thong_CVAT_V2.md) · [Semantic Segmentation](data/Guideline_Semantic_Segmentation_CVAT_V2.md)

## Cấu trúc guideline tổng quát

| Phần | Cần ghi gì |
|---|---|
| Phạm vi | Loại ảnh, đối tượng hoặc vùng cần gán, trường hợp loại trừ. |
| Nhãn và cách vẽ | Tên chính xác trên CVAT, kiểu shape, khi nào gán, cách đặt biên và ví dụ. |
| Ca dễ nhầm và checklist | Dấu hiệu để phân biệt, cách xử lý phần khuất hoặc thiếu bằng chứng, người chốt ca khó. |
| Khối `cvat-guideline` | JSON V2 gồm danh sách nhãn và thông tin model để tool điền sẵn. |

Giữ một khối `cvat-guideline` trong file. Trong `labels`, mỗi khóa là **tên nhãn chính xác trên CVAT** và mỗi nhãn có một `shape`. Các kiểu hợp lệ là `rectangle`, `polygon`, `polyline`, `mask` và `skeleton`. `sam3_prompt` là cụm mô tả đối tượng, thường viết ngắn bằng tiếng Anh. `yolo_class` là tên lớp COCO và chỉ dùng với `rectangle`. Nhãn `skeleton` dùng cấu trúc điểm trên CVAT. Bạn có thể bỏ các ngưỡng để dùng mặc định của tool.

```cvat-guideline
{
  "version": 2,
  "labels": {
    "chai_nhua": {"shape": "rectangle", "yolo_class": "bottle", "sam3_prompt": "plastic bottle"},
    "vung_nuoc_do": {"shape": "polygon", "sam3_prompt": "liquid spill on the floor"}
  }
}
```

Tool đọc cấu hình trong khối JSON. Các mô tả nghiệp vụ và checklist dùng để đối chiếu khi review. Xem cách điền từng phần và ý nghĩa các ngưỡng trong [cấu trúc guideline chi tiết](data/Cau_Truc_Guideline_Gan_Nhan.md).

## Chạy một nhóm ảnh

1. Mở [Mo_CVAT_Job_Runner.cmd](Mo_CVAT_Job_Runner.cmd). Dán link job đầy đủ dạng `https://cvat.example.com/tasks/123/jobs/456`. Nhập access token của cùng máy chủ. Với workspace cá nhân, để trống **Organization**. Với job của tổ chức, nhập tên ngắn của tổ chức trên CVAT.
2. Chọn **SAM 3 và Pose** hoặc **Job BBox (YOLO)**. Bấm **Chọn guideline** nếu có file, rồi bấm **Tải nhãn từ job**.
3. Tick nhãn cần chạy hoặc dùng **Chọn tất cả** để chọn các nhãn đủ cấu hình. Kiểm tra cột **Kiểu xuất** và mô tả SAM 3 hoặc lớp YOLO. Nếu đổi lựa chọn, hãy tạo lại bản xem trước.
4. Đặt **Từ ảnh** và **Đến ảnh**, tính từ 1 theo thứ tự ảnh trong job. Có thể thử `1` đến `1` trước. Mỗi lượt tối đa 10 ảnh. Bấm **Chạy xem trước** và theo dõi nhật ký. Dùng **Mở bản review** nếu trình duyệt chưa tự mở.
5. Kiểm tra nhãn, biên và các mục cần rà. Nếu đạt, bấm **Lưu lên CVAT** rồi **Mở job CVAT** để kiểm tra kết quả. Bản xem trước chỉ nằm trên máy cho đến khi bạn bấm lưu.

Kết quả xem trước nằm trong `runs/`. Giữ token, guideline nội bộ và kết quả chứa ảnh job trên máy hoặc trong nơi lưu trữ mà dự án cho phép.

## Khi gặp lỗi

| Hiện tượng | Cách kiểm tra |
|---|---|
| `401` hoặc `403` | Kiểm tra máy chủ, token, Organization và quyền trên job. Khi lưu, token cần thuộc tài khoản được giao job. |
| Không thấy nhãn hoặc sai kiểu shape | Đối chiếu tên chính xác và kiểu nhãn trong CVAT với `labels` trong guideline. |
| Chạy xem trước bị mờ | Tick ít nhất một nhãn. Điền mô tả hoặc lớp model nếu ô còn trống. |
| Bản xem trước có 0 shape | Xem nhật ký, thử một ảnh có đối tượng rõ và kiểm tra kết quả đã có trên CVAT. |

Nếu dùng biến môi trường, CSV token hoặc dòng lệnh, xem [hướng dẫn CLI](docs/README_CVAT_BBOX_AGENT.md).
