# EDA Report — Smart Agriculture DSS

- Tổng số ảnh: **54,284**
- Tổng số class: **38**
- Tổng số loại cây trồng (plant_type): **14**
- Tỷ lệ mất cân bằng (max/min): **36.23x** (CẢNH BÁO — cần xử lý ở Phase 5/6)

## Phân bố Class

- Class nhiều ảnh nhất: `Orange___Haunglongbing_(Citrus_greening)` — 5,507 ảnh
- Class ít ảnh nhất: `Potato___healthy` — 152 ảnh

## Duplicate Analysis

- Tổng số dòng: 54,284
- Duplicate còn sót lại: 0 (kỳ vọng = 0, đã loại bỏ ở Phase 3)

## Missing Analysis

Không phát hiện giá trị thiếu (NaN) ở bất kỳ cột nào.

## Khuyến nghị cho Phase 5/6

- Xử lý mất cân bằng class bằng class weighting, oversampling hoặc augmentation có trọng số cho các class ít ảnh.
- Kiểm tra lại các class có số ảnh thấp trước khi chia train/val/test để đảm bảo mỗi tập vẫn có đại diện đầy đủ (dùng stratified split).