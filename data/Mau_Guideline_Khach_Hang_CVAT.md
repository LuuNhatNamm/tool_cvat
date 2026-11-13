# Guideline gán nhãn CVAT cho [tên job]

Sao chép file này cho từng job. Thay các nhãn minh họa bằng tên chính xác trong **Labels** trên CVAT, kể cả dấu cách và chữ hoa. Điền mô tả nghiệp vụ bằng tiếng Việt nếu thuận tiện. Với SAM 3, nên dùng cụm từ tiếng Anh mô tả đối tượng nhìn thấy trong ảnh.

## Phạm vi

Ghi loại ảnh, các đối tượng hoặc vùng cần gán, và các điều kiện loại trừ áp dụng cho cả job.

## Ý nghĩa nhãn và cách vẽ

| Tên nhãn CVAT | Khi nào gán | Cách vẽ và ví dụ |
|---|---|---|
| `[Tên chính xác]` | [Dấu hiệu đủ để gán] | [Cách xác định biên, tách các đối tượng, xử lý phần khuất] |

## Ca dễ nhầm và checklist review

| Tình huống | Cách xử lý |
|---|---|
| [Hai nhãn dễ nhầm] | [Dấu hiệu để chọn nhãn đúng] |
| [Biên khuất, bóng đổ hoặc phản chiếu] | [Ghi vùng cần đối chiếu, thuộc tính cần kiểm tra] |
| [Thiếu bằng chứng] | [Ghi số frame, vị trí và điểm cần người review chốt] |

## Cấu hình cho Job Runner

Giữ đúng một khối `cvat-guideline` và dùng JSON hợp lệ. Khóa trong `labels` là tên nhãn CVAT. `yolo_class` dùng cho `rectangle`. `sam3_prompt` dùng cho `rectangle`, `polygon`, `polyline` và `mask`. Có thể điền mô tả trong tool. Nhãn `skeleton` dùng model pose người và lấy cấu trúc điểm từ CVAT.

```cvat-guideline
{
  "version": 2,
  "labels": {
    "nhan_vat_the": {"shape": "rectangle", "yolo_class": "bottle", "sam3_prompt": "plastic bottle"},
    "nhan_vung": {"shape": "polygon", "sam3_prompt": "liquid spill on the floor"},
    "nhan_duong": {"shape": "polyline", "sam3_prompt": "painted line on the floor"},
    "nhan_mask": {"shape": "mask", "sam3_prompt": "liquid spill on the floor"},
    "nhan_pose_nguoi": {"shape": "skeleton"}
  },
  "bbox_min_score": 0.30,
  "bbox_min_area_px": 150,
  "mask_min_score": 0.50
}
```

Các ngưỡng là tùy chọn. `bbox_min_score` áp dụng cho YOLO. `bbox_min_area_px` lọc Rectangle từ cả YOLO và SAM 3. `mask_min_score` là điểm nhận diện tối thiểu của SAM 3 cho cả Rectangle và Polygon.

Trong Job Runner, chọn file, tải nhãn từ job và tick các nhãn cần chạy. Tool dùng cấu hình trên để kiểm tra kiểu shape và điền gợi ý cho model. Phần quy tắc bằng chữ dùng khi review. Bấm **Chạy xem trước**, kiểm tra kết quả rồi mới bấm **Lưu lên CVAT**.
