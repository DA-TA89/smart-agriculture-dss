# Metadata Schema — PlantVillage Dataset

Mô tả cấu trúc file `data/processed/metadata.csv` và `data/processed/metadata_clean.csv`,
sinh ra bởi `pipelines/etl_pipeline.py` (Phase 3 — Data Engineering).

`metadata.csv` chứa **toàn bộ** ảnh tìm thấy (kể cả ảnh lỗi/trùng lặp).
`metadata_clean.csv` chỉ chứa các dòng có `is_corrupt=False` và `is_duplicate=False`,
dùng làm đầu vào cho Phase 4 (EDA) và Phase 5 (Preprocessing).

| Column | Type | Description |
|---|---|---|
| `image_id` | int | Số thứ tự định danh duy nhất, sinh tuần tự khi quét dataset. |
| `filepath` | str | Đường dẫn tương đối từ project root đến file ảnh. |
| `class_name` | str | Tên thư mục gốc của ảnh (ví dụ `Tomato___Bacterial_spot`). |
| `plant_type` | str | Loại cây trồng, tách từ `class_name` trước dấu `___`. |
| `disease_name` | str | Tên bệnh (hoặc `healthy`), tách từ `class_name` sau dấu `___`. |
| `is_healthy` | bool | `True` nếu `disease_name` chứa từ khóa "healthy". |
| `width` | int / NaN | Chiều rộng ảnh (pixel). `NaN` nếu ảnh lỗi. |
| `height` | int / NaN | Chiều cao ảnh (pixel). `NaN` nếu ảnh lỗi. |
| `image_format` | str / NaN | Định dạng ảnh do Pillow xác định (JPEG, PNG...). |
| `file_size_bytes` | int | Kích thước file trên đĩa. |
| `is_corrupt` | bool | `True` nếu ảnh không mở được bằng Pillow hoặc nhỏ hơn `min_file_size_bytes`. |
| `is_duplicate` | bool | `True` nếu nội dung file trùng khớp (hash) với một ảnh khác đã quét trước đó. |

## Quy tắc suy ra nhãn

`class_name` được lấy từ **tên thư mục cha trực tiếp** của mỗi file ảnh —
đúng với cách PlantVillage tổ chức dữ liệu (mỗi thư mục lá = một class).

`plant_type` / `disease_name` được tách theo cấu hình `class_separator` và
`healthy_keyword` trong `configs/data.yaml`, không hard-code trong source code.

## Trùng lặp (Duplicate Detection)

Việc phát hiện trùng lặp dựa trên **hash nội dung file** (mặc định MD5, cấu
hình tại `configs/data.yaml -> dataset.duplicate_hash_algorithm`). Đây là
duplicate **byte-for-byte**, không phát hiện được ảnh trùng về mặt nội dung
nhưng khác định dạng nén (ví dụ bản gốc và bản resize). Nếu Kaggle archive
có nhiều biến thể ảnh (`color` / `grayscale` / `segmented`), cân nhắc dùng
`dataset_subfolder` trong `configs/data.yaml` để chỉ xử lý một biến thể.