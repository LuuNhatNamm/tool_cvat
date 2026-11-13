# Guideline mẫu cho ảnh giao thông

Ví dụ theo danh sách nhãn trong guideline BBox, Polygon và Polyline. Đối chiếu tên nhãn với job trước khi dùng. Tick những nhãn cần chạy cho từng chế độ.

## Phạm vi và quy tắc review

- Rectangle bao sát từng vật thể. Kiểm tra xe chở người và xe chở hàng khi phân biệt `car` và `truck`. Rà lại `occluded` và `truncated`. Loại phản chiếu, bóng và vật thể chỉ xuất hiện trên màn hình.
- Polygon `area/direct` là vùng xe có thể đi thẳng theo hướng hiện tại. `area/alternative` là vùng cùng chiều có thể đi tới sau khi chuyển làn. Kiểm tra ranh giới và hướng chạy trên ảnh khi review.
- Polyline `lane/*` đi theo tim vạch hoặc đường biên bó vỉa theo nhãn. Rà các đề xuất SAM 3 và dừng khi không còn đủ dấu hiệu thị giác.
- Với ca khó, ghi số frame, vị trí và điểm chưa rõ để người review chốt.

## Cấu hình cho Job Runner

`traffic sign` dùng SAM 3 vì lớp `stop sign` của YOLO chỉ bao phủ biển STOP. Hai nhãn `area/*` để trống mô tả để người vận hành tự điền theo cảnh. SAM 3 chưa tự suy ra làn đi thẳng và làn thay thế từ quy tắc bằng chữ.

```cvat-guideline
{
  "version": 2,
  "labels": {
    "pedestrian": {"shape": "rectangle", "yolo_class": "person", "sam3_prompt": "pedestrian"},
    "rider": {"shape": "rectangle", "sam3_prompt": "person riding a bicycle or motorcycle"},
    "car": {"shape": "rectangle", "yolo_class": "car", "sam3_prompt": "car"},
    "truck": {"shape": "rectangle", "yolo_class": "truck", "sam3_prompt": "truck"},
    "bus": {"shape": "rectangle", "yolo_class": "bus", "sam3_prompt": "bus"},
    "train": {"shape": "rectangle", "yolo_class": "train", "sam3_prompt": "train"},
    "motorcycle": {"shape": "rectangle", "yolo_class": "motorcycle", "sam3_prompt": "motorcycle"},
    "bicycle": {"shape": "rectangle", "yolo_class": "bicycle", "sam3_prompt": "bicycle"},
    "traffic light": {"shape": "rectangle", "yolo_class": "traffic light", "sam3_prompt": "traffic light"},
    "traffic sign": {"shape": "rectangle", "sam3_prompt": "traffic sign"},
    "area/direct": {"shape": "polygon"},
    "area/alternative": {"shape": "polygon"},
    "lane/crosswalk": {"shape": "polyline", "sam3_prompt": "white pedestrian crossing stripe"},
    "lane/double white": {"shape": "polyline"},
    "lane/double yellow": {"shape": "polyline"},
    "lane/double other": {"shape": "polyline"},
    "lane/road curb": {"shape": "polyline"},
    "lane/single white": {"shape": "polyline"},
    "lane/single yellow": {"shape": "polyline", "sam3_prompt": "yellow lane marking"},
    "lane/single other": {"shape": "polyline"}
  }
}
```
