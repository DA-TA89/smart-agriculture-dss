# Software Design Specification 

## Smart Agriculture Decision Support System 

---

# 1. Purpose 

## Mục tiêu 
Software Design Specification (SDS) là tài liệu kỹ thuật tối cao định nghĩa toàn bộ quy tắc thiết kế hệ thống phần mềm trước khi bắt đầu giai đoạn lập trình cho dự án Hệ Hỗ trợ Quyết định trong Nông nghiệp Thông minh .

Tài liệu này không tập trung vào giải thích lý thuyết Machine Learning, thuật toán Deep Learning hay chi tiết triển khai Database .

Vai trò cốt lõi của SDS bao gồm:
- Chuẩn hóa cấu trúc thư mục của toàn bộ dự án .
- Chuẩn hóa cấu trúc source code và phân chia trách nhiệm module .
- Chuẩn hóa quy tắc lập trình (Coding Standards) và đặt tên (Naming Convention) .
- Chuẩn hóa hệ thống quản lý cấu hình và log .
- Trở thành tài liệu tham chiếu duy nhất cho toàn bộ kỹ sư trong quá trình phát triển để đảm bảo tính đồng nhất .

---

# 2. Design Principles 

Toàn bộ dự án phải tuân thủ nghiêm ngặt các nguyên tắc thiết kế phần mềm sau :

## Modular Design 
Mỗi module hoạt động hoàn toàn độc lập . Một module chỉ chịu trách nhiệm cho một nhóm tác vụ duy nhất . Sự thay đổi bên trong một module không được phép làm hỏng hoạt động của module khác.

## Separation of Concerns 
Tách biệt rõ ràng các mối quan tâm :
- Data (Xử lý dữ liệu) 
- Training (Huấn luyện AI) 
- Evaluation (Đánh giá) 
- Inference (Dự đoán) 
- API (Backend) 
- Dashboard (Giao diện) 

Các thành phần này không được phụ thuộc trực tiếp vào logic nội bộ của nhau .

## High Cohesion 
Tính gắn kết cao. Các chức năng liên quan phải được đặt cùng module . Tất cả các class, function phục vụ chung cho một mục đích phải được gom lại.

## Low Coupling 
Tính phụ thuộc thấp. Các module giao tiếp với nhau thông qua các Interface hoặc các hàm trung gian đã được định nghĩa . Tuyệt đối không gọi chéo trực tiếp .

## Reusability 
Các module, hàm tiện ích phải có khả năng tái sử dụng cho nhiều pipeline khác nhau .

## Maintainability 
Mã nguồn phải dễ bảo trì:
- Không Hard Code (ghi cứng giá trị vào code) .
- Không Duplicate Code (lặp lại mã nguồn) .

## Configurability 
Mọi tham số động bắt buộc phải được khai báo trong file cấu hình .

## Scalability 
Hệ thống được thiết kế để dễ dàng mở rộng :
Từ việc hỗ trợ 1 model lên n model mà không làm thay đổi kiến trúc cốt lõi hiện tại .

---

# 3. Architecture Style 

Dự án áp dụng mô hình Layered Modular Architecture  (Kiến trúc Module phân lớp). Luồng dữ liệu và lời gọi hàm di chuyển từ trên xuống dưới.

    Presentation
            ↓
    Application
            ↓
    Business
            ↓
    AI
            ↓
    Infrastructure

## Presentation Layer 
Dashboard . Không xử lý AI . Hiển thị giao diện người dùng, nhận ảnh upload, hiển thị biểu đồ và kết quả. 

## Application Layer 
FastAPI . Nhận Request . Trả Response . Tiếp nhận HTTP Request từ giao diện và xác thực dữ liệu.

## Business Layer 
Điều phối luồng xử lý . Không huấn luyện mô hình . Chuyển đổi kết quả thô từ AI thành các khuyến nghị kinh doanh.

## AI Layer 
Training, Inference, Evaluation . Tiền xử lý tensor, huấn luyện mô hình và đưa ra dự đoán.

## Infrastructure Layer 
Logging, Configuration, Docker, GitHub . Cung cấp hạ tầng lưu trữ, cấu hình và giám sát hệ thống.

---

# 4. Project Directory Structure 

Cấu trúc thư mục cấp cao nhất của dự án:

    smart-agriculture-dss/
    ├── api/             # Mã nguồn FastAPI phục vụ REST API
    ├── configs/         # Lưu trữ toàn bộ file cấu hình YAML (.yaml)
    ├── dashboard/       # Mã nguồn giao diện Streamlit
    ├── data/            # Thư mục chứa dữ liệu vật lý (được Git ignore, DVC quản lý)
    ├── deployment/      # File cấu hình hạ tầng (Docker, Nginx, CI/CD)
    ├── docs/            # Tài liệu kỹ thuật, kiến trúc, sơ đồ
    ├── logs/            # Nơi lưu trữ log file sinh ra trong quá trình chạy
    ├── models/          # Nơi lưu trữ các file weights/checkpoints (.pth, .onnx)
    ├── notebooks/       # Các file Jupyter Notebook phục vụ EDA và thử nghiệm
    ├── pipelines/       # Các script chạy pipeline tự động (Train, ETL)
    ├── scripts/         # Shell scripts tiện ích (setup môi trường, run docker)
    ├── src/             # Mã nguồn AI và Business Logic cốt lõi
    ├── tests/           # Kịch bản kiểm thử tự động (Unit, Integration Test)
    ├── README.md        # Tài liệu tổng quan dự án
    └── requirements.txt # Danh sách thư viện phụ thuộc Python

---

# 5. Source Code Structure 

Bên trong thư mục cốt lõi src/, mã nguồn được tổ chức thành các package dựa trên trách nhiệm:

    src/
    ├── config/          # Module hỗ trợ load và parse cấu hình từ thư mục /configs
    ├── data/            # Logic kết nối dataset, tạo DataLoader
    ├── features/        # Tiền xử lý, transform ảnh, augmentation
    ├── models/          # Khai báo kiến trúc Mạng Nơ-ron (ViT, CNN)
    ├── training/        # Vòng lặp huấn luyện, optimizer, loss function, scheduler
    ├── evaluation/      # Tính toán metrics, sinh confusion matrix
    ├── inference/       # Load model checkpoint và dự đoán trên ảnh mới
    ├── decision_engine/ # Hệ chuyên gia sinh khuyến nghị từ dự đoán AI
    ├── services/        # Các service nghiệp vụ trung gian
    └── utils/           # Các hàm tiện ích dùng chung (logger, timer, formatters)

---

# 6. Module Responsibility 

Sự phân chia ranh giới rõ ràng giữa các module trong src/:

## data 
Đọc dữ liệu, kiểm tra dữ liệu . Không train .

## features 
Tiền xử lý, transform, augmentation .

## models 
Chỉ chứa kiến trúc mô hình .

## training 
Quản lý luồng huấn luyện , lưu trữ Checkpoints trong quá trình train.

## evaluation 
Đánh giá  và sinh báo cáo phân loại.

## inference 
Dự đoán . Load mô hình đã train để dự đoán thời gian thực.

## decision_engine 
Nhận Prediction, sinh Decision . Truy xuất Knowledge Base để tìm mức độ nghiêm trọng và giải pháp.

## services 
Business Service . Bọc các logic nhỏ lại thành dịch vụ cung cấp cho API.

## utils 
Hàm dùng chung  như loggers, hàm sinh định dạng.

---

# 7. Naming Convention 

Hệ thống tuân thủ nghiêm ngặt quy tắc đặt tên sau:

- Python file / Folder: Chữ thường, phân cách bằng dấu gạch dưới (`snake_case`).
  Ví dụ: `image_processor.py`, `decision_engine/`
- Function / Method: Chữ thường, diễn tả hành động bằng động từ (`snake_case`).
  Ví dụ: `load_dataset()`, `predict_disease()`
- Variable: Chữ thường, danh từ có ý nghĩa (`snake_case`).
  Ví dụ: `confidence_score`, `learning_rate`
- Constant: Chữ hoa toàn bộ, phân cách bằng dấu gạch dưới (`UPPER_SNAKE_CASE`).
  Ví dụ: `MAX_IMAGE_SIZE = 224`
- Class: Viết hoa chữ cái đầu mỗi từ, không gạch dưới (`PascalCase`).
  Ví dụ: `DecisionEngine`, `DataValidator`
- Model Checkpoint: Cú pháp `{architecture}_{dataset}_{metric}.pth`.
  Ví dụ: `vit_base_plantvillage_f1_0.98.pth`
- Notebook: Cú pháp `{số_thứ_tự}_{nội_dung_ngắn}.ipynb`.
  Ví dụ: `01_eda_plantvillage.ipynb`
- CSV / Image (Output): Chữ thường, dấu gạch ngang (`kebab-case`).
  Ví dụ: `confusion-matrix.png`, `validation-report.csv`
- Log file: Cú pháp `{YYYY-MM-DD}_{module}.log`.
  Ví dụ: `2026-06-29_api.log`

---

# 8. Configuration Strategy 

Tuyệt đối Không Hard Code . Toàn bộ cấu hình đặt trong configs/  dưới định dạng YAML.

- `base.yaml` : Đường dẫn thư mục toàn cục.
- `data.yaml` : Cấu hình bộ dataset.
- `model.yaml` : Cấu hình kiến trúc mạng.
- `training.yaml` : Siêu tham số huấn luyện.
- `logging.yaml` : Cấu hình định dạng Log, Log Level.
- `api.yaml`: Cấu hình Backend (Host IP, Port, API Version).

## Phân tách cấu hình và secrets

- **YAML files** (`configs/`): Chứa tham số kỹ thuật, không nhạy cảm. Được commit lên Git.
- **`.env` file** (root): Chứa thông tin nhạy cảm — database password, secret keys, external API keys. **Không được commit lên Git** (đã thêm vào `.gitignore`).
- **`.env.example`** (root): File mẫu liệt kê tất cả biến môi trường cần thiết nhưng không chứa giá trị thật. **Được commit lên Git** làm tài liệu hướng dẫn.

## Thư mục sinh tự động (không commit)

- `mlruns/` : Sinh tự động bởi MLflow. Phải thêm vào `.gitignore`.
- `logs/` : Sinh tự động trong quá trình chạy. Phải thêm vào `.gitignore`.
- `data/` : Quản lý bởi DVC, không commit trực tiếp lên Git.

---

# 9. Coding Standards 

Mọi dòng code viết ra phải đáp ứng các tiêu chuẩn:

- Mỗi module chỉ một trách nhiệm .
- Function ngắn : Tối đa không quá 50 dòng code để đảm bảo dễ đọc và dễ test.
- Type Hint : Bắt buộc sử dụng Type Hint của Python 3.
- Docstring : Bắt buộc viết Docstring cho tất cả các class và các hàm public.
- Logging thay vì print() : Không sử dụng lệnh `print()`.
- Không Magic Number : Phải gán các con số vào các hằng số.
- Không Code Duplication : Bất kỳ logic nào xuất hiện từ 2 lần trở lên đều phải tách hàm.

---

# 10. Final Project Tree 

Cấu trúc chính thức của toàn bộ dự án :

    smart-agriculture-dss/
    ├── api/
    │   ├── routers/
    │   ├── schemas/
    │   ├── dependencies.py
    │   └── main.py
    ├── configs/
    │   ├── api.yaml
    │   ├── base.yaml
    │   ├── data.yaml
    │   ├── logging.yaml
    │   ├── model.yaml
    │   └── training.yaml
    ├── dashboard/
    │   ├── components/
    │   ├── pages/
    │   └── app.py
    ├── data/
    │   ├── external/
    │   ├── processed/
    │   └── raw/
    ├── deployment/
    │   ├── docker/
    │   │   ├── Dockerfile.api
    │   │   └── Dockerfile.dashboard
    │   └── nginx/
    ├── docs/
    │   ├── architecture/
    │   ├── data_dictionary/
    │   └── user_guide/
    ├── logs/
    │   ├── api/
    │   └── system/
    ├── models/
    ├── notebooks/
    ├── pipelines/
    │   ├── etl_pipeline.py
    │   └── train_pipeline.py
    ├── scripts/
    ├── src/
    │   ├── __init__.py
    │   ├── config/
    │   │   └── __init__.py
    │   ├── data/
    │   │   └── __init__.py
    │   ├── decision_engine/
    │   │   └── __init__.py
    │   ├── evaluation/
    │   │   └── __init__.py
    │   ├── features/
    │   │   └── __init__.py
    │   ├── inference/
    │   │   └── __init__.py
    │   ├── models/
    │   │   └── __init__.py
    │   ├── services/
    │   │   └── __init__.py
    │   ├── training/
    │   │   └── __init__.py
    │   └── utils/
    │       └── __init__.py
    ├── tests/
    │   ├── conftest.py
    │   ├── api/
    │   ├── integration/
    │   └── unit/
    ├── .env.example
    ├── .gitignore
    ├── docker-compose.yml
    ├── dvc.yaml
    ├── README.md
    └── requirements.txt

---

# 11. Design Rules 

Trong toàn bộ dự án :
- Không được tạo thư mục ngoài SDS .
- Không đổi Naming Convention .
- Không Hard Code .
- Không lưu cấu hình trong source .
- Không commit dữ liệu nhạy cảm .
- Không commit file `.env` — chỉ commit `.env.example` .
- Không commit thư mục `data/`, `logs/`, `mlruns/`, `models/` — quản lý bởi DVC hoặc Git-ignore .
- `Dockerfile` chỉ được đặt trong `deployment/docker/`, không đặt ở root .
- Mọi module mới phải tuân theo cấu trúc đã định nghĩa .

---

# 12. Definition of Done 

- Cấu trúc thư mục đã được chốt .
- Module đã được xác định .
- Quy tắc đặt tên đã thống nhất .
- Quy tắc lập trình đã thống nhất .
- Quy tắc cấu hình đã thống nhất .
- Toàn bộ giai đoạn triển khai sau phải tuân theo SDS .