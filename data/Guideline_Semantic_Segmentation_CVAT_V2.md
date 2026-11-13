# Guideline Semantic Segmentation cho ảnh giao thông

Áp dụng cho ảnh JPG 1280 × 720 trong `segmentation/images`. Mỗi pixel thuộc tối đa một lớp ngữ nghĩa. Các vật thể cùng lớp dùng chung tên nhãn, không cần đánh số từng cá thể. Giữ nguyên 19 tên dưới đây. Màu dùng để đối chiếu khi xem lớp phủ.

## Danh sách nhãn và cách phân biệt

| Nhãn CVAT | Màu HEX | Vùng cần gán |
|---|---|---|
| `road` | `#804080` | Mặt đường dành cho phương tiện. Bóng đổ trên mặt đường vẫn thuộc `road`. |
| `sidewalk` | `#F423E8` | Vỉa hè hoặc lối đi bộ tách khỏi mặt đường. |
| `building` | `#464646` | Bề mặt thuộc công trình hoặc tòa nhà, gồm phần tường là một bộ phận của tòa nhà. |
| `wall` | `#66669C` | Tường đặc độc lập hoặc tường ranh giới. |
| `fence` | `#BE9999` | Hàng rào dạng thanh hoặc lưới. Kiểm tra các khe hở còn nhìn rõ. |
| `pole` | `#999999` | Thân cột hoặc phần giá đỡ của đèn và biển báo. Tách khỏi phần đèn và mặt biển. |
| `traffic_light` | `#FAAA1E` | Cụm đèn tín hiệu giao thông. |
| `traffic_sign` | `#DCDC00` | Phần biển báo giao thông nhìn thấy. |
| `vegetation` | `#6B8E23` | Cây, bụi cây và tán lá. |
| `terrain` | `#98FB98` | Bề mặt tự nhiên như đất và cỏ thấp. |
| `sky` | `#4682B4` | Phần bầu trời nhìn thấy. |
| `person` | `#DC143C` | Người đi bộ, đứng hoặc ngồi ngoài phương tiện đang điều khiển hay cưỡi. |
| `rider` | `#FF0000` | Phần người đang điều khiển hoặc ngồi trên phương tiện, tách khỏi phần xe. |
| `car` | `#00008E` | Ô tô con. Đưa xe van khó phân loại vào review. |
| `truck` | `#000046` | Xe tải. Phân biệt với ô tô con và xe buýt theo hình dáng và công dụng quan sát được. |
| `bus` | `#003C64` | Xe buýt. |
| `train` | `#005064` | Tàu hỏa. |
| `motorcycle` | `#0000E6` | Xe máy, tách khỏi người trên xe. |
| `bicycle` | `#770B20` | Xe đạp, tách khỏi người trên xe. Giữ các chi tiết mảnh còn nhìn rõ. |

## Vẽ Polygon và rà biên

- Vẽ sát phần vùng nhìn thấy. Kiểm tra kỹ cột, đèn, biển báo, người và xe hai bánh khi phóng to. Bỏ phần bị che khuất và phần nằm ngoài ảnh.
- Rà vùng giao nhau giữa các lớp. Mỗi pixel chỉ có một nhãn. Lấp khe hở giữa các vùng đã xác định rõ, để lại vùng chưa đủ căn cứ cho reviewer.
- Bỏ phản chiếu và hình vật thể trên màn hình hoặc biển quảng cáo khi gán lớp vật thể. Giữ nhãn của bề mặt dưới bóng đổ nếu vẫn nhận ra được bề mặt.
- Rà vùng lớn như đường, trời, tòa nhà và cây trước. Sau đó rà vật thể nhỏ và các lớp dễ nhầm. Có thể giảm độ đậm lớp phủ để đối chiếu ảnh gốc.

## Kiểm tra trước khi lưu

1. Rà lần đầu để kiểm tra đủ vùng rõ ràng, đúng tên nhãn và không trùng vùng.
2. Rà lần hai để kiểm tra biên, khe hở và lớp chồng lên nhau. SAM 3 có thể trả về các vùng chồng nhau từ nhiều mô tả, nên cần xử lý chỗ chồng trước khi dùng kết quả làm nhãn semantic.
3. Nếu thiếu căn cứ, ghi số frame, vị trí, điều chưa rõ và câu hỏi cần chốt. Dùng `UNCERTAIN_CLASS`, `UNCERTAIN_BOUNDARY`, `UNCERTAIN_SCOPE`, `ATTRIBUTE_CHECK` hoặc `LOW_VISIBILITY` tùy trường hợp. Chuyển ca chưa thống nhất cho mentor theo quy trình của job.

## Cấu hình cho CVAT Job Runner

Chọn file này bằng **Chọn guideline**, chọn chế độ **SAM 3 và Pose**, rồi tải nhãn của job segmentation. Cấu hình dưới đây dùng **Polygon** theo phạm vi của tài liệu gốc. Nhãn trên CVAT cần có cùng tên và kiểu Polygon hoặc Any. Tên có dấu gạch dưới như `traffic_light` và `traffic_sign` phải khớp chính xác. Nếu job cho phép Mask, xác nhận kiểu với người phụ trách trước khi đổi cấu hình.

Tick các nhãn cần chạy rồi chọn **Chạy xem trước**. Các mô tả tiếng Anh là gợi ý cho SAM 3. Bảng màu bên trên dùng để đối chiếu màu nhãn trên CVAT. Các quy tắc về lớp chồng nhau, khoảng trống và ca khó cần được kiểm tra trong bản xem trước trước khi bấm **Lưu lên CVAT**.

Với Polygon `vegetation`, Job Runner ưu tiên SegFormer Cityscapes khi có model local để nhận diện cây, bụi cây và tán lá. Nếu thiếu model này, tool dùng mô tả SAM 3 trong cấu hình. Rà lại vùng cây sát nền đất, cỏ thấp và công trình.

```cvat-guideline
{
  "version": 2,
  "labels": {
    "road": {"shape": "polygon", "sam3_prompt": "road surface for vehicles"},
    "sidewalk": {"shape": "polygon", "sam3_prompt": "pedestrian sidewalk"},
    "building": {"shape": "polygon", "sam3_prompt": "building facade"},
    "wall": {"shape": "polygon", "sam3_prompt": "freestanding solid wall"},
    "fence": {"shape": "polygon", "sam3_prompt": "fence with bars or mesh"},
    "pole": {"shape": "polygon", "sam3_prompt": "street pole or sign support pole"},
    "traffic_light": {"shape": "polygon", "sam3_prompt": "traffic signal head"},
    "traffic_sign": {"shape": "polygon", "sam3_prompt": "traffic sign face"},
    "vegetation": {"shape": "polygon", "sam3_prompt": "tree"},
    "terrain": {"shape": "polygon", "sam3_prompt": "natural ground with soil or short grass"},
    "sky": {"shape": "polygon", "sam3_prompt": "sky"},
    "person": {"shape": "polygon", "sam3_prompt": "pedestrian or person off a vehicle"},
    "rider": {"shape": "polygon", "sam3_prompt": "person riding a bicycle or motorcycle"},
    "car": {"shape": "polygon", "sam3_prompt": "passenger car"},
    "truck": {"shape": "polygon", "sam3_prompt": "truck"},
    "bus": {"shape": "polygon", "sam3_prompt": "bus"},
    "train": {"shape": "polygon", "sam3_prompt": "train"},
    "motorcycle": {"shape": "polygon", "sam3_prompt": "motorcycle"},
    "bicycle": {"shape": "polygon", "sam3_prompt": "bicycle"}
  }
}
```
