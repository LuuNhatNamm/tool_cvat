# Chạy công cụ CVAT bằng dòng lệnh

Với Job Runner, nhập link job, token của máy chủ đó và Organization nếu có. Để trống Organization cho workspace cá nhân. Xem [README](../README.md) để chạy xem trước rồi lưu trong giao diện.

Khi dùng CLI, đặt `CVAT_SERVER`, `CVAT_TASK_ID`, `CVAT_JOB_ID` và `CVAT_ORGANIZATION` nếu job thuộc nhóm. Các giá trị dưới đây là ví dụ. Đặt token bằng biến môi trường hoặc bằng `config/access_tokens.csv` trên máy. File CSV cần cột `token` hoặc `access_token`.

```powershell
$env:CVAT_SERVER = 'https://cvat.example.com'
$env:CVAT_TASK_ID = '123'
$env:CVAT_JOB_ID = '456'
python .\src\cvat_bbox_agent.py preview --limit 5 --output .\bbox-review.json
```

Kiểm tra ảnh và hộp trong bản HTML trước khi ghi. Khi sẵn sàng, chạy `python .\src\cvat_bbox_agent.py apply --plan .\bbox-review.json`. Dùng `python .\src\local_annotation_review.py --help` để xem cách tạo bản review semantic từ ảnh cục bộ. Kiểm tra quyền sử dụng nhãn tự động theo quy định của job.
