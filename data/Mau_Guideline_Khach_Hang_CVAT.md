# Guideline khách hàng cho job CVAT

Điền loại dữ liệu, điều kiện gán nhãn và các trường hợp cần reviewer quyết định. Bảng dưới là ví dụ cho ảnh giao thông 2D. Thay tên và nội dung theo job của bạn.

## Danh sách nhãn

| Tên nhãn trong CVAT | Loại shape | Ý nghĩa và điều kiện gán | Ví dụ |
|---|---|---|---|
| `car` | Rectangle | Vẽ một hộp sát mỗi ô tô con nhìn thấy rõ. | Xe con màu trắng đang đi trên đường. |
| `road` | Mask | Gán phần mặt đường nhìn thấy. | Mặt đường liên tục qua vùng bóng đổ. |
| `building` | Mask | Gán bề mặt công trình. | Mặt tiền tòa nhà nhìn từ đường. |

## Trường hợp cần review

| Tình huống | Dấu hiệu | Cách xử lý |
|---|---|---|
| Không rõ một vùng thuộc `road` hay `building` | Bóng đậm hoặc mép vật thể không nhìn rõ | Ghi frame, vị trí, dấu hiệu còn thiếu và câu hỏi cần reviewer chốt. |

## Cấu hình dùng trong tool

Trong mỗi ánh xạ, khóa là tên lớp mà model nhận ra và giá trị là tên nhãn chính xác đã cấu hình trên CVAT. Xóa dòng của những lớp ngoài phạm vi. Khi chỉ chạy một loại shape, có thể bỏ bảng ánh xạ còn lại.

```cvat-guideline
{
  "version": 1,
  "bbox": {"car": "car"},
  "segmentation": {"road": "road", "building": "building"},
  "bbox_min_score": 0.40,
  "bbox_min_area_px": 250,
  "mask_min_score": 0.60
}
```

Nguồn BBox hỗ trợ `pedestrian`, `bicycle`, `car`, `motorcycle`, `bus`, `train`, `truck`, `traffic light` và `traffic sign` (YOLO COCO chỉ phát hiện biển STOP). Nguồn segmentation hỗ trợ `road`, `sidewalk`, `building`, `wall`, `fence`, `pole`, `traffic_light`, `traffic_sign`, `vegetation`, `terrain`, `sky`, `person`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle` và `bicycle`.

Các ví dụ và quy tắc bằng văn bản dành cho người review. Chuyển yêu cầu máy có thể kiểm tra thành ánh xạ hoặc ngưỡng trong khối cấu hình. Với lớp mới hoặc quy tắc nghiệp vụ khác, người vận hành cần chọn model hoặc bổ sung logic trước khi chạy tự động.
