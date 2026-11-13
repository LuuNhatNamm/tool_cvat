# CVAT Job Runner

Tool Windows gán nhãn ảnh 2D trên CVAT bằng model chạy trên máy. **SAM 3 và Pose** tạo Rectangle, Polygon, Polyline, Mask và Skeleton trong cùng lượt. **Job BBox (YOLO)** dùng cho BBox theo lớp COCO.

## Chạy một nhóm ảnh

1. Mở `Mo_CVAT_Job_Runner.cmd`. Nhập link đầy đủ của job, access token của máy chủ trong link và slug Organization nếu job thuộc tổ chức. Với workspace cá nhân, để trống Organization. Nhập token trực tiếp hoặc đặt `CVAT_SERVER` trùng máy chủ trước khi dùng token từ biến môi trường hoặc `config/access_tokens.csv`.
2. Chọn chế độ rồi bấm **Tải nhãn từ job**. Tick các nhãn cần gán. Cột **Kiểu xuất** giới hạn lựa chọn theo kiểu nhãn trên CVAT và guideline đã nạp.
3. Với **SAM 3**, chọn **Kiểu xuất** và nhập mô tả ngắn như `utility pole` hoặc `road surface`. Kiểu nhãn trên CVAT và guideline quyết định các lựa chọn trong bảng. Nếu nhãn `Any` cho phép cả hai kiểu, bạn có thể chọn **Rectangle và Polygon**. Hai nhãn khác nhau có thể dùng chung mô tả nếu một nhãn xuất Rectangle và nhãn còn lại xuất Polygon. Tool chạy mỗi mô tả một lần trên mỗi ảnh. Với **Job BBox (YOLO)**, chọn lớp tương ứng trong 80 lớp COCO.
4. Chọn **Từ ảnh** và **Đến ảnh**, tính từ 1 theo thứ tự ảnh của job. Mỗi lượt tối đa 10 ảnh. Bấm **Chạy xem trước**. Tool tạo báo cáo HTML cục bộ và tự mở để bạn kiểm tra. Bạn có thể mở lại bằng **Mở bản review**.
5. Khi kết quả đạt yêu cầu, bấm **Lưu lên CVAT** trong cửa sổ tool. Tool gửi các shape từ nhóm ảnh vừa xem trước. Nếu bạn đổi job, chế độ, nhãn, mô tả hoặc nhóm ảnh, hãy chạy xem trước lại. Sau khi lưu, dùng **Mở job CVAT** để kiểm tra và chỉnh tọa độ hoặc đỉnh Polygon.

- **Polyline** đi theo đường tim của vùng SAM 3 dự đoán. Rà đầu mút, các đoạn đứt và các sọc ở vạch qua đường trước khi lưu. Vùng bị model nhận liền vẫn có thể cần tách bằng tay.
- **Mask** giữ vùng pixel và lỗ bên trong, được lưu dưới kiểu Mask gốc của CVAT.
- **Skeleton** dùng `models/yolo11n-pose.pt` cho pose người. Job phải có nhãn `skeleton` với các điểm con đặt theo tên COCO như `nose`, `left_shoulder`, `right_shoulder`. Tool dùng cấu trúc cạnh của CVAT và đánh dấu điểm dưới ngưỡng tin cậy là `outside`. Skeleton có điểm ngoài bộ COCO cần model riêng.

## Chuẩn bị model

- Cài Python 3.10 trở lên có Tkinter. Nếu thiếu thư viện, chạy `python -m pip install torch transformers ultralytics opencv-python numpy Pillow`.
- Giữ trọng số YOLO ở `models/yolo11n.pt` và checkpoint SAM 3 ở `models/sam3/`. SAM 3 dùng `Sam3Model` và `Sam3Processor` của Transformers 5.x. Checkpoint khoảng 3,44 GB. Lượt thử một ảnh giúp ước lượng thời gian chạy trên CPU.
- Nếu thiếu checkpoint, xin quyền truy cập [model SAM 3](https://huggingface.co/facebook/sam3) rồi tải vào thư mục tool.

```powershell
hf auth login
hf download facebook/sam3 --exclude sam3.pt --local-dir models/sam3
```

## Dùng guideline của job

Bấm **Chọn guideline** và dùng [mẫu Markdown chuẩn](data/Mau_Guideline_Khach_Hang_CVAT.md) cho job mới. Khối `cvat-guideline` phiên bản 2 khai báo tên CVAT, kiểu shape, lớp YOLO, mô tả SAM 3 và các ngưỡng tùy chọn. Xem [ví dụ đã điền cho ảnh giao thông](data/Guideline_Mau_Gan_Nhan_Anh_Giao_Thong.md). Tool lọc kiểu shape, điền gợi ý và báo trong nhật ký nếu tên hoặc kiểu nhãn không khớp CVAT. Bạn tick những nhãn muốn chạy. Phần chữ hiện trong bản xem trước để đối chiếu.

Tool vẫn đọc Markdown phiên bản 1 và file Word có bảng **CVAT shape** và **Class áp dụng**. Dùng **Bỏ guideline** để quay lại chọn trực tiếp theo nhãn CVAT.

Trong [guideline giao thông](data/Guideline_Gan_Nhan_Giao_Thong_CVAT_V2.md), `area/direct` và `area/alternative` có kiểu **Polygon**, còn `lane/*` có kiểu **Polyline**. Kiểm tra hướng di chuyển, ranh giới và tim vạch trước khi lưu. Nếu job có thêm Mask hoặc Skeleton, khai báo đúng tên và kiểu của chúng trong guideline hoặc chọn nhãn trực tiếp khi không dùng guideline.

## Đọc kết quả

Mỗi lượt có một thư mục trong `runs/` với `index.html` để xem trước và `review.json` để đối chiếu tọa độ. Bản xem trước hiển thị các shape có thể thêm tại thời điểm chạy. Tool kiểm tra lại nhãn, ảnh và annotation trước khi lưu. Nếu dữ liệu đã đổi và ảnh hưởng đến shape đã duyệt, hãy tạo bản xem trước mới.

| Hiện tượng | Cách xử lý |
|---|---|
| CVAT báo `401` hoặc `403` | Kiểm tra token, Organization và quyền trên job. Khi lưu, token phải thuộc tài khoản assignee theo quy tắc của runner. |
| Bản xem trước có 0 shape | Rà lại lớp YOLO hoặc mô tả SAM 3, thử một ảnh có đối tượng rõ và xem nhật ký các shape đã có hoặc bị trùng. |
| SAM 3 mất nhiều thời gian | Chạy thử một ảnh với một nhãn. Mỗi mô tả bổ sung cần thêm một lượt dự đoán trên ảnh. |
| Frame đã có segmentation khác | Đối chiếu polygon hoặc mask hiện có trên CVAT với nhật ký. |

Xem [hướng dẫn dòng lệnh](docs/README_CVAT_BBOX_AGENT.md) cho quy trình CLI hiện có.
