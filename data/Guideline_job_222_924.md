# Guideline mẫu gán nhãn ảnh giao thông

Tra theo loại annotation đang thực hiện để chọn đúng tên label và cách xử lý. Mỗi mục nêu label hợp lệ, dấu hiệu phân biệt và cách xử lý trường hợp khó. Dùng đúng tên trong bảng, kể cả dấu cách, dấu gạch dưới và tiền tố `area/`, `lane/`. Chỉ gán label có trong danh mục của bài.

## 1. Bounding Box cho đối tượng

| Label | Khi nào dùng và cách phân biệt |
| --- | --- |
| `pedestrian` | Người không ở trên phương tiện hai bánh, ví dụ đang đi, chạy, đứng hoặc ngồi. Hộp không bao gồm túi, ba lô hoặc ô. |
| `rider` | Người đang điều khiển hoặc ngồi trên phương tiện hai bánh. Gán hộp riêng cho người và cho phương tiện nếu phương tiện thuộc danh mục. |
| `car` | Ô tô con và van chở người. |
| `truck` | Xe tải và van chở hàng. Nếu không đủ dấu hiệu để phân biệt van chở người và van chở hàng, tạo Issue `UNCERTAIN_CLASS`. |
| `bus` | Xe buýt. |
| `train` | Tàu hỏa. |
| `motorcycle` | Xe máy. |
| `bicycle` | Xe đạp. |
| `traffic light` | Đèn tín hiệu giao thông. |
| `traffic sign` | Biển báo giao thông. |

| Trường hợp | Cách xử lý |
| --- | --- |
| Nhiều đối tượng đứng sát hoặc che nhau | Vẽ một hộp riêng cho từng đối tượng. Hộp được phép chồng lấn. Canh bốn cạnh sát phần cần gán nhãn và giữ hộp trong khung ảnh. |
| Bị vật khác che trong ảnh | Bật `occluded = true`. |
| Bị cắt ở mép ảnh | Bật `truncated = true`. Nếu vừa bị che vừa bị cắt, bật cả hai thuộc tính. |
| Phần bị che hoặc ngoài ảnh ước tính không quá 80% | Gán nhãn đối tượng. |
| Phần bị che hoặc ngoài ảnh ước tính trên 80% | Vẫn nên gán nhãn nếu đủ dấu hiệu xác định cả vị trí và loại đối tượng. Nếu thiếu dấu hiệu, tạo Issue phù hợp. |

## 2. Polygon cho vùng đường có thể đi

| Label | Khi nào dùng và cách phân biệt |
| --- | --- |
| `area/direct` | Vùng mặt đường xe đang đi hoặc có thể tiếp tục đi trực tiếp mà không cần chuyển làn hay đổi hướng. |
| `area/alternative` | Vùng mặt đường ở làn khác cùng chiều mà xe có thể chuyển sang. Loại trừ phần đường ngược chiều. |

- Vẽ đường bao khép kín sát ranh giới vùng và tránh để các cạnh tự cắt. Dùng ít điểm ở đoạn thẳng và thêm điểm ở đoạn cong hoặc biên phức tạp.
- Khi thiếu dấu hiệu xác định ranh giới vùng, tạo Issue `UNCERTAIN_BOUNDARY`. Khi chưa xác định được làn thuộc `area/direct` hay `area/alternative`, tạo Issue `UNCERTAIN_CLASS`.

## 3. Polyline cho vạch đường và bó vỉa

| Label | Khi nào dùng và cách phân biệt |
| --- | --- |
| `lane/crosswalk` | Vạch dành cho người đi bộ sang đường. |
| `lane/double white` | Vạch đôi màu trắng. |
| `lane/double yellow` | Vạch đôi màu vàng. |
| `lane/double other` | Vạch đôi có màu đã xác định, khác trắng và vàng. |
| `lane/road curb` | Biên bó vỉa phân cách mặt đường với vỉa hè hoặc khu vực liền kề. |
| `lane/single white` | Vạch đơn màu trắng. |
| `lane/single yellow` | Vạch đơn màu vàng. |
| `lane/single other` | Vạch đơn có màu đã xác định, khác trắng và vàng. |

- Nhóm `single` và `double` áp dụng cho cả nét liền và nét đứt. Nếu chưa xác định được màu hoặc số vạch, tạo Issue `UNCERTAIN_CLASS`.
- Vẽ đường mở theo tim vạch hoặc theo biên bó vỉa. Dừng ở nơi hết dấu hiệu thị giác và tránh nối tắt qua vùng không có vạch. Giữ hướng vẽ nhất quán nếu batch có quy định.

## 4. Semantic Segmentation theo pixel

| Label | Khi nào dùng và cách phân biệt |
| --- | --- |
| `road` | Mặt đường dành cho phương tiện. |
| `sidewalk` | Lối đi bộ hoặc vỉa hè tách khỏi mặt đường. |
| `building` | Bề mặt thuộc tòa nhà hoặc công trình. |
| `wall` | Tường độc lập hoặc tường ranh giới có kết cấu kín, đặc. |
| `fence` | Hàng rào dạng thanh, lưới hoặc kết cấu có khoảng hở. |
| `pole` | Cột hoặc giá đỡ. Tách phần đèn hoặc mặt biển khỏi phần cột đỡ. |
| `traffic_light` | Phần đèn tín hiệu giao thông nhìn thấy. |
| `traffic_sign` | Phần biển báo giao thông nhìn thấy. |
| `vegetation` | Cây, bụi cây hoặc tán lá. |
| `terrain` | Bề mặt địa hình tự nhiên, ví dụ đất hoặc cỏ thấp. |
| `sky` | Vùng bầu trời nhìn thấy. |
| `person` | Người đi bộ, đứng hoặc ngồi không thuộc `rider`. |
| `rider` | Người đang điều khiển hoặc ngồi trên phương tiện. Tách pixel của người và pixel của phương tiện theo label tương ứng. |
| `car` | Ô tô con hoặc van chở người. |
| `truck` | Xe tải hoặc van chở hàng. |
| `bus` | Xe buýt. Nếu chưa phân biệt được `car`, `truck` và `bus`, tạo Issue `UNCERTAIN_CLASS`. |
| `train` | Tàu hỏa. |
| `motorcycle` | Xe máy. |
| `bicycle` | Xe đạp. |

| Trường hợp | Cách xử lý |
| --- | --- |
| Hai lớp tiếp giáp hoặc đối tượng che nhau | Gán mỗi pixel cho lớp nhìn thấy tại vị trí đó. Bám biên, tránh chồng lớp và lấp các khe bỏ sót khi đã xác định được lớp. Chuyển vùng chưa rõ cho reviewer. |
| Đối tượng bị cắt ở biên ảnh | Chỉ gán phần pixel nhìn thấy trong khung ảnh. |
| Chi tiết nhỏ, mảnh hoặc có khoảng hở | Phóng to để bám hình dạng thật của người, xe hai bánh, cột, biển và đèn. Giữ đúng lớp nhìn thấy trong các khoảng hở. |
| Nhiều đối tượng cùng lớp | Dùng cùng label ngữ nghĩa cho các pixel tương ứng. Bài toán không yêu cầu mã định danh riêng cho từng đối tượng. |

## 5. Xử lý trường hợp chưa rõ

Bóng đổ, hình phản chiếu và hình trên biển quảng cáo hoặc màn hình không được gán thành đối tượng thật, trừ khi batch có quy định riêng.

| Loại Issue | Khi nào dùng |
| --- | --- |
| `UNCERTAIN_CLASS` | Thấy đối tượng hoặc vùng nhưng chưa phân biệt được label, ví dụ van chở người hay van chở hàng. |
| `UNCERTAIN_BOUNDARY` | Biết label nhưng chưa xác định được biên hoặc điểm kết thúc, ví dụ vạch mờ dần. |
| `UNCERTAIN_SCOPE` | Chưa rõ đối tượng hoặc vùng có thuộc phạm vi gán nhãn. |
| `ATTRIBUTE_CHECK` | Chưa rõ nên bật `occluded`, `truncated` hay cả hai cho Bounding Box. |
| `LOW_VISIBILITY` | Thiếu dấu hiệu do đối tượng quá nhỏ, thiếu sáng, ngược sáng, mưa, sương hoặc nhòe chuyển động. |

1. Tạo Issue tại đúng frame và vùng hoặc đối tượng cần xem. Ghi loại Issue, label đang cân nhắc và dấu hiệu còn thiếu.
2. Đối chiếu nhận định trong nhóm và ghi kết quả biểu quyết theo đa số vào Issue. Chuyển reviewer hoặc mentor nếu nhóm chưa thống nhất hoặc guideline chưa có quy tắc. Áp dụng kết luận đã chốt cho các ca tương tự trong batch.

**Ví dụ Issue** — `UNCERTAIN_CLASS` · Frame 042 · Xe van bên phải · `car` hoặc `truck` · Thân sau bị che nên chưa xác định xe chở người hay chở hàng.

## 6. Cấu hình cho CVAT Job Runner

Khóa trong mỗi bảng là tên lớp nguồn mà Job Runner chấp nhận. Giá trị là tên label đích trong CVAT. Khi chọn **BBox và segmentation**, job cần có các label đích ở cả hai bảng. Label dùng chung cho hộp và mask cần có kiểu `Any` trong CVAT. Nếu job chỉ dùng một bộ nhãn con, bỏ các ánh xạ ngoài phạm vi của job.

Tool tự động tạo Bounding Box và mask cho các lớp có ánh xạ. `rider` có ánh xạ cho mask. Bounding Box của `rider`, polygon `area/*` và polyline `lane/*` cần gán thủ công. Với Bounding Box `traffic sign`, nguồn YOLO nhận diện biển STOP.

```cvat-guideline
{
  "version": 1,
  "bbox": {
    "pedestrian": "pedestrian",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign"
  },
  "segmentation": {
    "road": "road",
    "sidewalk": "sidewalk",
    "building": "building",
    "wall": "wall",
    "fence": "fence",
    "pole": "pole",
    "traffic_light": "traffic_light",
    "traffic_sign": "traffic_sign",
    "vegetation": "vegetation",
    "terrain": "terrain",
    "sky": "sky",
    "person": "person",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle"
  },
  "bbox_min_score": 0.30,
  "bbox_min_area_px": 150,
  "mask_min_score": 0.55
}
```
