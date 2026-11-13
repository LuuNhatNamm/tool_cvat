# Hướng dẫn gán Bounding Box trên CVAT

Áp dụng cho agent gán BBox trên ảnh giao thông. Bộ ảnh được mô tả trong bài là `.jpg`, 1280 × 720 px. Phạm vi công việc là vẽ và kiểm tra BBox trên các frame của job được giao.

## 1. Kiểm tra job trước khi vẽ

1. Chọn Organization của job, vào **Jobs** và mở đúng job được giao. Nếu công cụ đã mở sẵn job, đối chiếu Organization và job trước khi sửa.
2. Kiểm tra công cụ `Rectangle`, bộ nhãn và các thuộc tính của job. Khi job hoặc cấu hình không khớp, báo rõ điểm khác biệt cho lead để xác nhận. Dùng cấu hình có sẵn của job, giữ nguyên project, task, bộ nhãn và phân công.
3. Gán nhãn từ ảnh gốc. Bỏ qua annotation hoặc ground truth có sẵn dưới dạng tài liệu tham chiếu. Giữ các hộp hợp lệ đã lưu trong job khi tiếp tục công việc và kiểm tra để tránh tạo hộp trùng.

## 2. Nhãn hợp lệ

Dùng đúng tên nhãn trong bảng. Mỗi đối tượng hợp lệ có một `Rectangle` riêng.

| Nhãn CVAT | Phạm vi |
|---|---|
| `pedestrian` | Người đi bộ, đứng, chạy, ngồi hoặc ở tư thế khác ngoài trường hợp `rider`. Bỏ phần nhô ra của vật dụng cá nhân như túi xách, ba lô và ô. |
| `rider` | Mỗi người điều khiển hoặc ngồi trên phương tiện hai bánh. Gán phương tiện bằng một hộp riêng theo class của xe. |
| `car` | Ô tô con và xe van đã xác định được kiểu thân xe, theo quy ước bản review này. |
| `truck` | Xe tải. |
| `bus` | Xe buýt. |
| `train` | Tàu hỏa. |
| `motorcycle` | Xe máy. |
| `bicycle` | Xe đạp. |
| `traffic light` | Cụm đèn tín hiệu giao thông. Bám theo cụm đèn, bỏ phần cột hoặc giá đỡ nhô ra. |
| `traffic sign` | Mặt biển báo giao thông. Bám theo biển, bỏ phần cột hoặc giá đỡ nhô ra. |

Bỏ qua đối tượng ngoài danh sách, bóng đổ, phản chiếu và ảnh của đối tượng trên bảng quảng cáo hoặc màn hình. Nếu chưa phân biệt được thân van và thân xe tải, ghi `UNCERTAIN_CLASS`.

Trong bản review cục bộ, bỏ hộp dưới **150 px²** tính theo độ phân giải ảnh gốc, và bỏ hộp có tâm nằm trong vùng capô, taplo hoặc phụ kiện của xe ghi hình. Vùng xe ghi hình có ID 255 trong mask. Hộp YOLO dưới 0,30 chỉ được ghi chú để review. Hộp vàng từ mask cần xác minh đối tượng riêng trước khi dùng làm BBox.

## 3. Quy tắc vẽ hộp

1. Xác định các điểm ngoài cùng bên trái, bên phải, phía trên và phía dưới của đối tượng. Đặt bốn cạnh của hộp sát các điểm đó, bao đủ đối tượng và hạn chế nền thừa.
2. Giữ toàn bộ hộp trong biên ảnh và bảo đảm hộp có chiều rộng, chiều cao dương. Nếu đối tượng bị cắt ở mép ảnh, đặt cạnh tương ứng tại mép ảnh ở phía bị cắt.
3. Với đối tượng bị che, mặc định lấy bốn điểm ngoài cùng từ phần quan sát được. Nếu thấy nhiều mảnh rời và có đủ căn cứ chúng thuộc cùng một đối tượng, dùng một hộp bao các mảnh đó. Áp dụng quy ước của batch nếu batch quy định khác về phạm vi hộp qua phần che khuất.
4. Vẽ từng đối tượng riêng. Hộp của các đối tượng khác nhau có thể chồng lấn. Ví dụ, một người ngồi trên xe máy có một hộp `rider` và xe có một hộp `motorcycle`.
5. Phóng to vùng khó quan sát, nhất là người ở xa, xe hai bánh, đèn và biển báo nhỏ. Rà cả vùng sát mép ảnh và vùng quanh các vật thể lớn có thể che đối tượng nhỏ.

## 4. Đối tượng bị che hoặc cắt biên

| Tình huống | Thuộc tính cần bật |
|---|---|
| Đối tượng bị vật khác trong cảnh che một phần | `occluded = true` |
| Một phần đối tượng nằm ngoài khung ảnh | `truncated = true` |
| Đối tượng đồng thời bị che và nằm một phần ngoài khung ảnh | Bật cả hai |

- Nếu phần bị che hoặc nằm ngoài ảnh **không quá 80% kích thước ước tính của đối tượng**, gán BBox. Nếu vị trí, class hoặc biên vẫn chưa đủ rõ để vẽ đáng tin cậy, ghi Issue để review.
- Nếu phần bị che hoặc cắt **trên 80%**, vẫn ưu tiên gán khi còn đủ dấu hiệu xác định được cả vị trí và class. Nếu thiếu căn cứ, ghi Issue thay cho việc tạo hộp theo suy đoán.
- Khi hai xe đỗ sát nhau, vẽ hộp riêng theo phần nhìn thấy của từng xe. Chỉ bật `occluded` khi ảnh cho thấy vật khác thật sự che một phần đối tượng. Hộp chồng nhau là dấu hiệu để review thuộc tính và biên.
- Ước tính tỷ lệ trên kích thước của đối tượng đầy đủ. Khi khó ước tính và việc đó ảnh hưởng đến quyết định gán nhãn, chuyển review.

## 5. Ghi Issue khi cần review

Ghi frame, vị trí đối tượng, vấn đề cần chốt và dấu hiệu quan sát được. Dùng **Open an issue** trên vùng liên quan nếu job và công cụ hỗ trợ. Nếu agent không tạo được Issue trực tiếp, ghi đầy đủ các thông tin đó trong báo cáo cuối để reviewer xử lý.

| Loại Issue | Khi dùng |
|---|---|
| `UNCERTAIN_CLASS` | Chưa phân biệt được nhãn, ví dụ `car` hay `truck` với xe van. |
| `UNCERTAIN_BOUNDARY` | Chưa xác định được cạnh hộp hoặc chưa chắc các phần nhìn thấy thuộc cùng một đối tượng. |
| `UNCERTAIN_SCOPE` | Chưa rõ đối tượng có thuộc phạm vi gán BBox. |
| `ATTRIBUTE_CHECK` | Chưa chắc cách gán `occluded` hoặc `truncated`. |
| `LOW_VISIBILITY` | Đối tượng quá nhỏ, mờ, tối hoặc bị thời tiết làm khó quan sát. |

Ví dụ nội dung Issue

```text
UNCERTAIN_CLASS | Frame 018 | Xe giữa ảnh | Chỉ thấy đầu xe, chưa đủ dấu hiệu phân biệt thân van và thân xe tải.
```

## 6. Lưu và kiểm tra kết quả

1. Khi bắt đầu gán nhãn, đặt state `in progress` trong luồng CVAT. Lưu định kỳ bằng **Save** hoặc thao tác lưu tương ứng của công cụ.
2. Rà toàn bộ job hai lượt trước khi nộp.
   - **Lượt 1:** Đối chiếu 10 nhãn, rà đối tượng bị bỏ sót, hộp thừa và hộp trùng.
   - **Lượt 2:** Kiểm tra bốn cạnh, biên ảnh, thuộc tính và các ca đã ghi Issue hoặc đã chuyển reviewer.
3. Với yêu cầu chạy hết job đến bước nộp, chọn **Finish the job → Continue** sau khi đã rà hai lượt. Nếu yêu cầu chỉ định một số frame hoặc chỉ yêu cầu lưu kết quả, giữ trạng thái phù hợp với phạm vi được giao. Khi job bị trả về `rejected`, sửa theo tab **Issues**, kiểm tra lại rồi nộp theo luồng của job.
4. Khi kết thúc, báo job đã xử lý, phạm vi frame đã kiểm tra, trạng thái lưu hoặc nộp quan sát được và các Issue còn cần review. Nếu công cụ báo lưu thất bại, kiểm tra lại trạng thái trước khi thử lại để tránh tạo hộp trùng.

## Tài liệu tham chiếu

Xem [guideline giao thông V2](../data/Guideline_Gan_Nhan_Giao_Thong_CVAT_V2.md).
