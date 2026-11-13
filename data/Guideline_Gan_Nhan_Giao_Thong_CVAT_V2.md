# Guideline gán nhãn ảnh giao thông trên CVAT

Áp dụng cho ảnh đường phố trong bộ dữ liệu gốc (`images`, ảnh JPG 1280 × 720). Giữ nguyên tên nhãn và kiểu shape trong danh sách bên dưới. Dùng bản xem trước để rà đề xuất theo quy định của job. Tài liệu gốc yêu cầu học viên không dùng annotation hoặc ground truth có sẵn khi làm bài.

## Rectangle cho vật thể

| Nhãn CVAT | Khi nào gán |
|---|---|
| `pedestrian` | Người đi bộ hoặc người đang đứng, chạy, ngồi. Bỏ các vật dụng cá nhân như túi, ba lô và ô khỏi hộp. |
| `rider` | Người điều khiển hoặc ngồi trên phương tiện hai bánh. Gán người và phương tiện bằng các nhãn riêng khi cả hai nhận ra được. |
| `car` | Ô tô con, gồm xe van dùng để chở người. |
| `truck` | Xe tải, gồm xe van dùng để chở hàng. Nếu chưa phân biệt được công dụng của xe van, đưa vào review. |
| `bus` | Xe buýt. |
| `train` | Tàu hỏa. |
| `motorcycle` | Xe máy. |
| `bicycle` | Xe đạp. |
| `traffic light` | Đèn tín hiệu giao thông. |
| `traffic sign` | Biển báo giao thông. |

- Vẽ một hộp cho mỗi vật thể. Bám sát bốn điểm ngoài cùng theo phần nhìn thấy. Các hộp có thể chồng lấn khi vật thể ở gần hoặc che nhau.
- Gán vật thể nhận diện được khi phần bị che hoặc nằm ngoài ảnh chiếm tối đa 80% kích thước ước tính. Nếu vượt 80%, vẫn có thể gán khi còn đủ căn cứ xác định lớp và vị trí.
- Bật `occluded` khi vật thể bị vật thể khác che một phần. Bật `truncated` khi vật thể bị cắt bởi biên ảnh. Có thể bật cả hai.
- Bỏ bóng đổ, phản chiếu và hình vật thể trên biển quảng cáo hoặc màn hình, trừ trường hợp guideline riêng của batch quy định khác.

## Polygon cho vùng xe chạy

| Nhãn CVAT | Khi nào gán |
|---|---|
| `area/direct` | Phần mặt đường xe đang đi hoặc có thể tiếp tục đi theo hướng hiện tại mà không phải chuyển làn hay đổi hướng. |
| `area/alternative` | Làn liền kề cùng chiều mà xe gắn camera có thể chuyển vào. Loại làn ngược chiều, lề đường và vùng kẻ sọc cách ly. |

Vẽ đường bao khép kín sát ranh giới vùng nhìn thấy. Đặt ít đỉnh ở đoạn thẳng và nhiều đỉnh ở khúc cong hoặc biên phức tạp. Kiểm tra các cạnh không tự giao nhau.

## Polyline cho vạch đường và bó vỉa

| Nhãn CVAT | Khi nào gán |
|---|---|
| `lane/crosswalk` | Vạch dành cho người đi bộ sang đường. |
| `lane/double white` | Vạch đôi màu trắng, nét liền hoặc nét đứt. |
| `lane/double yellow` | Vạch đôi màu vàng, nét liền hoặc nét đứt. |
| `lane/double other` | Vạch đôi có màu khác trắng và vàng, nét liền hoặc nét đứt. |
| `lane/road curb` | Đường biên bó vỉa ngăn mặt đường với vỉa hè hoặc vùng liền kề. |
| `lane/single white` | Vạch đơn màu trắng, nét liền hoặc nét đứt. |
| `lane/single yellow` | Vạch đơn màu vàng, nét liền hoặc nét đứt. |
| `lane/single other` | Vạch đơn có màu khác trắng và vàng, nét liền hoặc nét đứt. |

Vẽ đường mở theo tim vạch hoặc theo đường biên bó vỉa. Dừng tại điểm không còn đủ dấu hiệu để xác định đường. Giữ hướng vẽ thống nhất nếu batch có yêu cầu. Khi dùng đề xuất SAM 3, kiểm tra riêng các sọc ở vạch qua đường và khoảng trống giữa các vạch đứt.

## Rà soát trước khi lưu

- Lượt 1: Rà đúng lớp, đúng kiểu shape, vật thể bị bỏ sót và annotation bị trùng. Kiểm tra riêng `rider` so với `pedestrian` và xe van chở người so với xe van chở hàng.
- Lượt 2: Rà biên hộp và thuộc tính, kiểm tra hướng của vùng xe chạy và các đoạn Polyline nối qua vùng thiếu dấu hiệu.
- Nếu cần review, ghi frame, vị trí, loại vấn đề và câu hỏi cần chốt. Dùng `UNCERTAIN_CLASS`, `UNCERTAIN_BOUNDARY`, `UNCERTAIN_SCOPE`, `ATTRIBUTE_CHECK` hoặc `LOW_VISIBILITY`. Nhóm đối chiếu cách gán trước và chuyển ca chưa thống nhất cho mentor.

## Cấu hình Job Runner

Chọn file này trong **Chọn guideline**, tải nhãn từ job, rồi tick các nhãn muốn chạy. Chế độ **SAM 3 và Pose** có thể đề xuất Rectangle, Polygon và Polyline trong cùng lượt. Các mô tả tiếng Anh là gợi ý cho model. Khi xem trước, kiểm tra số vạch đơn hoặc đôi và màu vạch. Với `traffic sign`, dùng SAM 3 để đề xuất nhiều loại biển. YOLO có lớp `stop sign` dành riêng cho biển STOP.

Với `area/direct`, Job Runner dùng vùng mặt đường và hai biên làn để đề xuất phần làn trước camera. Với `area/alternative`, tool có thể đề xuất làn kề bên trái khi thấy vạch trắng đứt ngăn hai làn và vạch vàng phía ngoài. Cách này áp dụng cho ảnh camera nhìn thẳng, trong bộ ảnh dùng vạch vàng phân cách chiều xe. Khi dấu hiệu không đủ, hoặc làn thay thế nằm ở bên phải, hãy xác nhận và bổ sung trên CVAT. Rà lại cả hai vùng trước khi lưu.

Tool đọc tên nhãn, kiểu shape và mô tả từ khối bên dưới. Các quy tắc như ngưỡng che khuất 80% và phân biệt xe van cần người review áp dụng. Xem trước rồi mới bấm **Lưu lên CVAT** nếu kết quả đạt.

```cvat-guideline
{
  "version": 2,
  "labels": {
    "pedestrian": {"shape": "rectangle", "yolo_class": "person", "sam3_prompt": "pedestrian"},
    "rider": {"shape": "rectangle", "sam3_prompt": "person riding a bicycle or motorcycle"},
    "car": {"shape": "rectangle", "yolo_class": "car", "sam3_prompt": "passenger car"},
    "truck": {"shape": "rectangle", "yolo_class": "truck", "sam3_prompt": "truck"},
    "bus": {"shape": "rectangle", "yolo_class": "bus", "sam3_prompt": "bus"},
    "train": {"shape": "rectangle", "yolo_class": "train", "sam3_prompt": "train"},
    "motorcycle": {"shape": "rectangle", "yolo_class": "motorcycle", "sam3_prompt": "motorcycle"},
    "bicycle": {"shape": "rectangle", "yolo_class": "bicycle", "sam3_prompt": "bicycle"},
    "traffic light": {"shape": "rectangle", "yolo_class": "traffic light", "sam3_prompt": "traffic light"},
    "traffic sign": {"shape": "rectangle", "sam3_prompt": "traffic sign"},
    "area/direct": {"shape": "polygon", "sam3_prompt": "current driving lane directly ahead of the camera vehicle"},
    "area/alternative": {"shape": "polygon", "sam3_prompt": "adjacent drivable lane with traffic moving in the same direction"},
    "lane/crosswalk": {"shape": "polyline", "sam3_prompt": "white pedestrian crossing stripe"},
    "lane/double white": {"shape": "polyline", "sam3_prompt": "double white lane marking"},
    "lane/double yellow": {"shape": "polyline", "sam3_prompt": "double yellow lane marking"},
    "lane/double other": {"shape": "polyline", "sam3_prompt": "double lane marking in a color other than white or yellow"},
    "lane/road curb": {"shape": "polyline", "sam3_prompt": "curb edge between the roadway and sidewalk"},
    "lane/single white": {"shape": "polyline", "sam3_prompt": "single white lane marking"},
    "lane/single yellow": {"shape": "polyline", "sam3_prompt": "single yellow lane marking"},
    "lane/single other": {"shape": "polyline", "sam3_prompt": "single lane marking in a color other than white or yellow"}
  }
}
```
