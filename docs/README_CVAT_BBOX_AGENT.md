# Chạy agent BBox cho job 222

## Chạy bằng cửa sổ Windows

Mở `Mo_CVAT_Job_Runner.cmd` hoặc chạy `python .\src\cvat_job_runner.py` trong PowerShell. Nhập link HTTPS đầy đủ dạng `/tasks/75/jobs/222`, token và slug Organization. Xóa ô Organization nếu job thuộc workspace cá nhân. Cửa sổ điền sẵn job 222. Token nhập trực tiếp được ẩn và dùng trong phiên chạy. Với CVAT Edufun, để trống token để dùng `config/access_tokens.csv` hiện có.

Chọn phạm vi lưu **BBox và segmentation**, **Chỉ Bounding Box** hoặc **Chỉ segmentation**, rồi bấm **Chạy**. Bỏ chọn **Lưu annotation lên CVAT** để chỉ tạo bản xem trước. Tiến độ và các mục bị bỏ qua xuất hiện trong cửa sổ. Sau khi chạy, dùng **Mở bản review** hoặc **Mở job CVAT**. Báo cáo và mask được lưu theo từng lượt trong thư mục `runs`.

Tool yêu cầu job ảnh 2D, nhãn phù hợp với phạm vi đã chọn và token có quyền ghi nếu bật lưu. Các quy tắc hiệu chỉnh camera đã rà được áp dụng khi chọn job 222 trên CVAT Edufun. Các job khác dùng quy tắc chung. Kết quả được lưu để bạn review và tự nộp trong CVAT.

### Guideline khách hàng

Gửi khách hàng `data/Mau_Guideline_Khach_Hang_CVAT.md`. Họ điền bảng mô tả nhãn và phần `cvat-guideline` với ánh xạ lớp model → nhãn CVAT, ngưỡng điểm BBox, diện tích BBox và ngưỡng mask. Trong cửa sổ, chọn file `.md` ở ô **Guideline khách hàng** rồi chọn phạm vi lưu tương ứng. Tool kiểm tra model có hỗ trợ lớp nguồn, nhãn đích có trên job và đúng kiểu shape. Có thể chạy thử bằng cách bỏ chọn **Lưu annotation lên CVAT**. `review.json` lưu tên file và cấu hình đã áp dụng.

Khối `cvat-guideline` điều khiển luồng tự động. Phần mô tả và ví dụ do reviewer đối chiếu. Chế độ guideline dùng ngưỡng mask trên kết quả SegFormer kết hợp các lượt ánh sáng. Các hiệu chỉnh mask riêng của job 222 dành cho chế độ mặc định. Để trống ô guideline để chạy theo cấu hình giao thông sẵn có.

Tool Python đọc ảnh từ [job 222 của task 75](https://cvat.edufun.ai/tasks/75/jobs/222), dùng YOLO11n trên CPU và áp dụng các quy tắc có thể kiểm tra của `BBOX_ANNOTATION_GUIDELINE.md`. Đích CVAT được cố định trong code. `run` chạy nhận diện và thêm BBox vào job. `preview` tạo bản xem trước. `apply` thêm hộp từ một file JSON đã có.

## Chuẩn bị trên Windows

1. Dùng Python 3.10 trở lên với `ultralytics` và `Pillow`.
2. Đặt trọng số ở `models/yolo11n.pt`. Tải từ [Ultralytics Assets](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt).
3. Lấy Personal Access Token trong phần Security của tài khoản CVAT được giao job 222. `preview` cần quyền đọc và `apply` cần token có quyền ghi. Đặt tệp token ở `config/access_tokens.csv`, với cột `token` hoặc `access_token` chứa đúng một giá trị PAT. File CVAT xuất có các cột như `name`, `readOnly` và `expiryDate` chỉ chứa thông tin quản lý, vì vậy cần thêm cột `token` và dán giá trị được hiển thị khi tạo PAT. Nếu dùng biến môi trường như bên dưới, script ưu tiên token từ biến đó. Nếu chạy trực tiếp mà chưa đặt biến môi trường và chưa có CSV, script sẽ hỏi token và ẩn ký tự lúc nhập. Trên CVAT dùng token API cũ, đặt `CVAT_LEGACY_TOKEN` thay cho `CVAT_ACCESS_TOKEN`.

```powershell
Set-Location 'E:\voc_vach\tool_cvat'
$cvatSecret = Read-Host 'CVAT Personal Access Token' -AsSecureString
$env:CVAT_ACCESS_TOKEN = [System.Net.NetworkCredential]::new('', $cvatSecret).Password
Remove-Variable cvatSecret
```

## Thử trên một frame

```powershell
python .\src\cvat_bbox_agent.py preview
```

Mở `bbox-job-222.html` để đối chiếu ảnh, khung và các ca cần review. Kết quả đề xuất nằm trong `bbox-job-222.json`. Các hộp dự đoán có thể lệch hoặc bỏ sót đối tượng nhỏ. Kiểm tra bằng ảnh trước khi lưu vào job. Nếu cần sửa file JSON, `bbox_2d` là bốn số nguyên `[x1, y1, x2, y2]` trên thang 0–1000 theo từng trục.

```powershell
python .\src\cvat_bbox_agent.py apply --plan .\bbox-job-222.json
```

Sau khi ghi, mở lại job 222 để kiểm tra. Tool chỉ thêm Rectangle và báo số hộp CVAT xác nhận. Bạn tạo Issue trên CVAT từ các ca được ghi trong kết quả, rà đủ hai lượt theo guideline rồi tự quyết định khi nào **Finish the job**.

## Chọn phạm vi tiếp theo trong job 222

```powershell
python .\src\cvat_bbox_agent.py preview --limit 5 --output .\bbox-job-222-5frames.json
python .\src\cvat_bbox_agent.py preview --all --output .\bbox-job-222-all.json
python .\src\cvat_bbox_agent.py run --output .\bbox-job-222-yolo.json
```

Dùng `--frame` để chọn số frame bắt đầu theo CVAT. `run` mặc định xử lý toàn job. Dùng tên file mới mỗi lần chạy. HTML chứa bản sao ảnh để kiểm tra trên máy. YOLO xử lý ảnh cục bộ. Nếu đã cài Qwen qua Ollama, vẫn có thể chọn `preview --model qwen3-vl:4b`.

YOLO COCO nhận dạng người và phương tiện theo bộ nhãn huấn luyện sẵn. Agent báo ca người chồng lên xe hai bánh để phân biệt `rider`, hộp chồng lấn để rà `occluded` và hộp chạm biên để rà `truncated`. `stop sign` được ánh xạ sang `traffic sign`. Khi review cần rà cả các biển báo khác và đối tượng nhỏ YOLO có thể bỏ sót.

Khi ghi, tool đối chiếu lại máy chủ, task, job, assignee, nhãn và kích thước frame. Nó loại hộp gần trùng cùng class trên cùng frame theo IoU từ 0,85. Các hộp của những class khác nhau vẫn được phép chồng lên nhau. Nếu job thiếu checkbox `truncated`, hộp cần thuộc tính đó được đưa vào báo cáo `ATTRIBUTE_CHECK`. Chạy một phiên ghi vào job tại một thời điểm.

Tham chiếu kỹ thuật [CVAT Jobs API](https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/jobs-api/) và [Ollama Chat API](https://docs.ollama.com/api/chat).

## Xem trước BBox và segmentation cục bộ

```powershell
python .\src\local_annotation_review.py --source .\bbox-job-222-yolo.html --output .\review-job-222-seg-refined
```

Nguồn có thể là một ảnh, thư mục ảnh hoặc HTML preview của agent. Lệnh đọc ảnh cục bộ và lưu `index.html`, `review.json`, mask PNG theo ID lớp và overlay theo từng frame. Dùng `--limit 1` để thử một ảnh, hoặc chọn thư mục đầu ra mới khi chạy lại.

YOLO chạy trên ảnh gốc và các vùng chồng nhau để tìm vật nhỏ. SegFormer B0 đã tinh chỉnh trên Cityscapes tạo mask theo 19 nhãn và palette trong `data/Semantic_Segmentation_Annotation_Guideline_V1.docx`. Mỗi pixel được gán một ID. Pixel thiếu chắc chắn có ID 255 để review. Hộp vàng nét đứt gợi ý vị trí `pedestrian`, `rider`, `traffic light` hoặc `traffic sign` từ các vùng mask. Kiểm tra từng đối tượng vì một vùng mask có thể chứa nhiều instance.

Pipeline review tự nhận ảnh từ HTML gốc của job 222 và áp dụng vùng loại trừ đã hiệu chỉnh cho từng frame. Có thể chọn `--camera-profile none` cho ảnh khác góc máy. Vùng loại trừ dùng ID 255 và được thống kê riêng với pixel thiếu chắc chắn. Hộp dưới 150 px², tâm nằm trong vùng loại trừ, xe hạng nặng quá lớn hoặc nghi bao nhiều xe được ghi trong `excluded_boxes`. Lượt SegFormer phóng to các vùng chồng nhau để tìm thêm gợi ý biển báo, đèn và người. Ảnh tối có thêm gợi ý từ cụm sáng xanh, cần xác minh đầu đèn thật. Các chỉnh sửa đã đối chiếu ảnh ở frame 0, 14 và 18 chỉ chạy trên đúng nội dung ảnh đã rà. Frame 2, 4, 6 và 11 được đưa lên đầu trang review.

Mask kết hợp ba lượt SegFormer trên ảnh gốc, ảnh tăng tương phản cục bộ và ảnh đổi độ sáng. Nếu dưới hai lượt đồng ý về lớp dự đoán, pixel đó được đánh dấu ID 255. `building` và `pole` dùng thêm bằng chứng từ các vùng ảnh phóng to, với lớp phải có tín hiệu hỗ trợ từ lượt toàn ảnh. `review.json` lưu số pixel được bổ sung cho hai lớp và số pixel bất đồng khi đổi sáng. Hai lớp này thuộc taxonomy segmentation.

Lượt giảm bóng cân bằng độ sáng cục bộ và kiểm tra lại các khoảng hở giữa hai vùng `road`. Chỉ bù pixel khi model dự đoán `road` đủ mạnh trên ảnh đã cân bằng và có bằng chứng từ ảnh gốc hoặc pixel đang cần review. Bộ lọc bảo vệ vật thể có BBox, ranh giới rào chắn chắc chắn và vùng ID 255 của xe ghi hình. Frame 2 có thêm biên đường và rào bê tông đã rà trên đúng ảnh mẫu. Báo cáo đếm riêng phần bù tự động và phần hiệu chỉnh mẫu. Dùng `--frame 2` cùng một thư mục đầu ra mới để thử riêng ảnh này.

Ảnh tối có thêm một lượt giảm nhiễu độ sáng và nâng bóng nhẹ, hạn chế khuếch đại đèn pha. Chỉ bù lỗ mask nhỏ nằm hoàn toàn trong thân xe cùng lớp và trong BBox. Các hiệu chỉnh theo biên đã rà ở frame 11, 18 và 19 được áp dụng theo dấu vân tay của đúng ảnh. Frame 19 có thêm lượt cân góc ảnh, sau đó chiếu mask về kích thước và góc của ảnh gốc. Báo cáo ghi riêng số pixel từ lượt cân góc, lấp lỗ và hiệu chỉnh mẫu.

## Gửi bản đã review lên job 222

```powershell
python .\src\publish_cvat_review.py
python .\src\publish_cvat_review.py --apply
```

Lệnh đầu đọc job và in số shape dự kiến. Lệnh thứ hai thêm BBox còn thiếu và mask theo lớp từ `review-job-222-seg-refined`. Có thể dùng `--review` để chọn bản review khác có đúng metadata của cả job. Tool bỏ qua pixel 255, gợi ý hộp vàng, các BBox xung đột và các frame đã có mask hoặc polygon segmentation khác với bản review. Bản đã rà có một sửa nhãn xe van `truck` → `car` trên BBox do tool tạo ở frame 0. Tool đọc lại các shape đã gửi để xác nhận. Bạn có thể rà trực tiếp trong CVAT trước khi nộp job.

Checkpoint cục bộ nằm ở `models/segformer-b0-cityscapes` và gồm `config.json`, `preprocessor_config.json`, `pytorch_model.bin`. Môi trường cần `torch`, `transformers`, `ultralytics`, `opencv-python`, `numpy` và `Pillow`. [Checkpoint SegFormer Cityscapes](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024/tree/21b3847)
