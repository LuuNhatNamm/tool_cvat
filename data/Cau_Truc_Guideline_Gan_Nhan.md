# Cấu trúc guideline gán nhãn cho một job CVAT

Sao chép [mẫu Markdown V2](Mau_Guideline_Khach_Hang_CVAT.md) để tạo file cho job mới. Xin quy tắc từ người giao job và lấy tên nhãn từ **Labels** của task hoặc project. Dùng cấu trúc này cho bất kỳ bộ ảnh 2D nào mà tool hỗ trợ.

## Phạm vi

Ghi loại ảnh, đối tượng hoặc vùng cần gán và các trường hợp loại trừ. Nêu yêu cầu về kích thước, độ nhìn thấy hoặc số lượng annotation nếu dự án có quy định.

## Danh sách nhãn và cách vẽ

| Tên chính xác trên CVAT | Kiểu shape | Khi nào gán | Cách vẽ và ví dụ |
|---|---|---|---|
| `[tên nhãn]` | `[kiểu shape]` | [Dấu hiệu đủ để gán và trường hợp loại trừ] | [Cách đặt biên, tách đối tượng hoặc xử lý phần bị che. Thêm một ví dụ rõ.] |

Thêm một dòng cho mỗi nhãn. Nếu có thuộc tính như `occluded` hoặc `truncated`, ghi điều kiện bật và ví dụ.

## Tình huống dễ nhầm và kiểm tra kết quả

| Tình huống | Cách xử lý |
|---|---|
| Hai nhãn có hình dạng gần giống | Ghi dấu hiệu để phân biệt và ví dụ cho từng nhãn. |
| Đối tượng bị khuất hoặc cắt ở mép | Ghi cách đặt biên và thuộc tính cần bật. |
| Ảnh thiếu sáng, chói hoặc mờ | Ghi khi nào có thể gán và khi nào chuyển người review. |

Khi gửi ca khó, ghi số frame, vị trí, dấu hiệu đã thấy và câu hỏi cần chốt. Trước khi lưu, rà tên nhãn, hình học, phần bỏ sót và annotation trùng.

## Cấu hình cho Job Runner

Đặt đúng một khối `cvat-guideline` ở cuối file. Giữ `version` là `2`. Trong `labels`, khóa là tên nhãn trên CVAT, kể cả chữ hoa, dấu cách và dấu gạch dưới. Thay các tên ví dụ trong mẫu bằng tên thật của job.

| Trường | Cách điền |
|---|---|
| `shape` | Bắt buộc cho mỗi nhãn. Dùng `rectangle`, `polygon`, `polyline`, `mask` hoặc `skeleton`. |
| `sam3_prompt` | Tùy chọn. Dùng cụm ngắn như `plastic bottle` để điền sẵn trong SAM 3. Áp dụng cho Rectangle, Polygon, Polyline và Mask. |
| `yolo_class` | Tùy chọn cho Rectangle. Dùng tên lớp COCO chính xác, ví dụ `bottle`. |
| `bbox_min_score` | Ngưỡng YOLO tùy chọn, từ 0,30 đến 1. Mặc định 0,30. |
| `bbox_min_area_px` | Diện tích Rectangle tối thiểu, số nguyên từ 150. Mặc định 150 px². |
| `mask_min_score` | Ngưỡng nhận diện SAM 3, lớn hơn 0 và tối đa 1. Mặc định 0,50. |

Với Skeleton, chỉ khai báo `shape` và cấu hình các điểm con theo COCO Pose trong CVAT. Nếu để trống mô tả SAM 3 hoặc lớp YOLO trong file, bạn có thể điền trong bảng nhãn của tool.

Nạp file bằng **Chọn guideline**, tải nhãn từ job và xem nhật ký để phát hiện tên hoặc kiểu không khớp. Các quy tắc viết bằng chữ được đưa vào bản xem trước để người dùng rà kết quả.
