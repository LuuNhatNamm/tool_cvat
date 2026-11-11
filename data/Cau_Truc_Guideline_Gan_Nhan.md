# Guideline gán nhãn cho [tên job]

[Ghi loại dữ liệu, đơn vị được gán nhãn và số label có thể gán cho một đơn vị nếu điều này ảnh hưởng đến cách chọn nhãn. Ví dụ “Mỗi phản hồi được gán một label theo yêu cầu chính của khách hàng.”]

## Phần 1 Danh sách label và ý nghĩa

| Tên label | Ý nghĩa và điều kiện gán | Dễ nhầm với label nào | Ví dụ điển hình |
| --- | --- | --- | --- |
| `[tên chính xác trong hệ thống]` | [Đối tượng, vùng hoặc nội dung nào đủ điều kiện nhận label này] | [Label gần nhất và dấu hiệu phân biệt] | [Một trường hợp rõ ràng và label cần gán] |

Thêm một dòng cho mỗi label hợp lệ. Nếu một label bắt buộc có loại annotation hoặc thuộc tính cụ thể, thêm cột tương ứng. Nếu job có nhiều bộ label, ghi rõ mỗi bộ áp dụng cho loại dữ liệu hoặc đơn vị nào.

## Phần 2 Tình huống dễ nhầm và cách xử lý

| Label liên quan | Tình huống và dấu hiệu nhận biết | Cách xử lý | Ví dụ và kết quả mong đợi |
| --- | --- | --- | --- |
| `[label A]`, `[label B]` | [Cùng một dữ liệu thỏa điều kiện của hai label] | [Khi nào chọn A, khi nào chọn B hoặc khi nào gán cả hai nếu job cho phép] | [Dữ liệu cụ thể → label cần gán và lý do] |
| `[label A]` | [Dữ liệu thiếu dấu hiệu để kết luận] | [Nêu dấu hiệu tối thiểu cần có. Nếu thiếu, ghi hành động đã được job quy định, chẳng hạn chuyển review] | [Dữ liệu thiếu dấu hiệu nào → hành động tương ứng] |
| [Các label áp dụng] | [Trường hợp có vẻ liên quan nhưng nằm ngoài phạm vi] | [Chỉ rõ điều kiện loại trừ và cách ghi nhận theo quy định của job] | [Dữ liệu cụ thể → kết quả] |

Chỉ đưa vào các tình huống có thể làm thay đổi label, số lượng nhãn, phạm vi hoặc thuộc tính. Nếu quy tắc áp dụng cho nhiều label, ghi một dòng chung và chỉ rõ phạm vi áp dụng.

## Phần 3 Trường hợp guideline chưa có quy tắc

| Cần ghi trong Issue | Nội dung |
| --- | --- |
| Vị trí | [ID mẫu, frame, vùng ảnh, đoạn văn hoặc mốc thời gian cần xem] |
| Điểm chưa rõ | [Label, ranh giới, thuộc tính hoặc phạm vi đang cần quyết định] |
| Dấu hiệu | [Dữ liệu quan sát được và dấu hiệu còn thiếu] |
| Quyết định cần chốt | [Câu hỏi cụ thể gửi người phụ trách review] |

Ghi rõ ai chốt và annotator cần làm gì trong lúc chờ. Khi có kết luận áp dụng cho các mẫu tương tự, thêm một dòng vào Phần 2 để cả nhóm dùng chung.
