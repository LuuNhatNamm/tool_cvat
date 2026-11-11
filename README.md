# CVAT Job Runner

Tool Windows dùng YOLO để đề xuất bounding box và SegFormer để tạo mask segmentation cho job ảnh 2D trên CVAT. Model chạy trên máy. Ảnh được tải từ CVAT và kết quả chỉ được ghi lên job khi bật tùy chọn lưu.

## Chuẩn bị

- Python 3.10 trở lên có Tkinter. Nếu thiếu thư viện, chạy `python -m pip install torch transformers ultralytics opencv-python numpy Pillow` trong môi trường Python dùng để mở tool.
- Trọng số YOLO nằm ở `models/yolo11n.pt`. Checkpoint SegFormer nằm trong `models/segformer-b0-cityscapes/`. Giữ nguyên cấu trúc khi chuyển thư mục tool.
- Link đầy đủ của job, ví dụ `https://cvat.edufun.ai/tasks/75/jobs/222`, cùng token của tài khoản có quyền truy cập job. Để lưu kết quả, dùng token có quyền ghi của người được giao job.

## Chạy tool

1. Bấm đúp `Mo_CVAT_Job_Runner.cmd`. Bạn cũng có thể mở PowerShell trong thư mục tool và chạy:

   ```powershell
   python .\src\cvat_job_runner.py
   ```

2. Nhập **Link job**, **Access token CVAT** và slug **Organization**. Với workspace cá nhân, để trống Organization. Với CVAT Edufun, có thể để trống token nếu đã có `config/access_tokens.csv`. Nếu dùng CSV, đặt giá trị token vào cột `token` hoặc `access_token`. Token nhập trong cửa sổ được ẩn và dùng cho lượt chạy đó.
3. Chọn file **Guideline khách hàng** nếu job dùng cấu hình riêng. Để trống ô này để dùng bộ nhãn giao thông mặc định.
4. Chọn **Phạm vi lưu** gồm **BBox và segmentation**, **Chỉ Bounding Box** hoặc **Chỉ segmentation**. Bỏ chọn **Lưu annotation lên CVAT** nếu muốn xem trước. Bấm **Chạy** và theo dõi trạng thái trong cửa sổ.
5. Bấm **Mở bản review** để xem ảnh và các ca cần rà. Nếu đã bật lưu, bấm **Mở job CVAT** để kiểm tra kết quả trên web trước khi chọn **Finish the job**.

## Dùng guideline khách hàng

Gửi khách hàng [mẫu guideline Markdown](data/Mau_Guideline_Khach_Hang_CVAT.md). Điền bảng mô tả nhãn và ví dụ, rồi khai báo phần tool áp dụng tự động trong khối `cvat-guideline` của file. Ví dụ:

````markdown
```cvat-guideline
{
  "version": 1,
  "bbox": {"car": "xe_con"},
  "segmentation": {"road": "mat_duong"},
  "bbox_min_score": 0.40,
  "bbox_min_area_px": 250,
  "mask_min_score": 0.60
}
```
````

Khóa là tên lớp model nhận ra. Giá trị là tên nhãn chính xác trên CVAT. Ví dụ trên yêu cầu nhãn `xe_con` nhận Rectangle và `mat_duong` nhận Mask. Tool kiểm tra lớp nguồn, nhãn đích và loại shape trước khi chạy. Phần mô tả và ví dụ dành cho reviewer. Lớp ngoài khả năng của model hoặc quy tắc nghiệp vụ riêng cần bổ sung model hay logic tương ứng.

## Đọc kết quả

Mỗi lượt chạy có một thư mục trong `runs/` chứa `index.html`, `review.json`, mask PNG và ảnh overlay. Ô log cho biết frame đã xử lý, shape CVAT xác nhận và các mục cần rà. Hộp vàng nét đứt là gợi ý review. Pixel có ID `255` nằm ngoài vùng xuất mask lên CVAT.

| Hiện tượng | Cách xử lý |
|---|---|
| Link job không hợp lệ | Dùng link HTTPS có cả `/tasks/<id>/jobs/<id>`. |
| CVAT báo `401` hoặc `403` | Kiểm tra token, Organization và quyền trên job. |
| Thiếu nhãn hoặc sai kiểu shape | Đối chiếu nhãn CVAT với phạm vi lưu và ánh xạ trong guideline. |
| BBox bị bỏ qua vì `truncated` | Rà hộp chạm biên trong bản review. Job cần thuộc tính `truncated` kiểu checkbox để tool ghi cờ này. |
| Frame đã có segmentation khác | Mở frame trên CVAT để đối chiếu mask hoặc polygon hiện có với bản review. |

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `src/` | Mã giao diện, kết nối CVAT và xử lý ảnh. |
| `config/` | Tệp token dùng trên máy. |
| `models/` | Trọng số YOLO và SegFormer. |
| `data/` | Tài liệu nguồn và mẫu guideline khách hàng. |
| `docs/` | Hướng dẫn gán nhãn và hướng dẫn dòng lệnh. |
| `runs/` | Báo cáo theo từng lượt chạy từ giao diện. Tool tạo thư mục này khi có kết quả. |

Xem [hướng dẫn dòng lệnh và cấu hình chi tiết](docs/README_CVAT_BBOX_AGENT.md) nếu cần chạy từng frame hoặc gửi một bản review đã tạo.
