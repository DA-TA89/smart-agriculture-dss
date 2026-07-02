# Smart Agriculture Decision Support System
## Using Deep Learning & Vision Transformer
---
# 1. Project Overview
## Project Name
**Smart Agriculture Decision Support System using Deep Learning and Vision Transformer**

---

# 2. Business Problem

Trong nông nghiệp, việc phát hiện bệnh trên cây trồng thường dựa vào kinh nghiệm của người nông dân.
Các khó khăn bao gồm:
- Khó nhận biết bệnh ở giai đoạn sớm.
- Dễ nhầm lẫn giữa nhiều loại bệnh.
- Thiếu chuyên gia hỗ trợ tại địa phương.
- Chi phí kiểm tra cao.
- Bệnh có thể lây lan nhanh nếu xử lý chậm.
- Sử dụng thuốc bảo vệ thực vật không đúng cách gây lãng phí và ảnh hưởng môi trường.

Hệ thống cần hỗ trợ người dùng:
- Phát hiện bệnh cây từ ảnh lá.
- Đánh giá mức độ tin cậy của dự đoán.
- Đưa ra khuyến nghị xử lý.
- Hỗ trợ ra quyết định nhằm giảm thiệt hại và tối ưu năng suất.

---

# 3. Project Objectives

Xây dựng một hệ thống Decision Support System (DSS) hoàn chỉnh bao gồm:

- Data Engineering Pipeline
- Data Validation
- Data Analysis (EDA)
- Image Preprocessing
- Deep Learning Training
- Vision Transformer
- Model Evaluation
- Decision Engine
- REST API
- Dashboard
- MLOps
- Deployment

Đây là dự án Portfolio phục vụ ứng tuyển:

- Data Engineer
- Data Analyst

theo tiêu chuẩn Production.

---

# 4. Dataset

Dataset:

PlantVillage Dataset

https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Bao gồm:

- nhiều loại cây trồng
- nhiều loại bệnh
- Healthy class
- Hàng chục nghìn ảnh RGB

Input:

Leaf Image

Output:

Disease Classification

---

# 5. Functional Requirements

Hệ thống phải hỗ trợ:

### Image Upload

- Upload ảnh lá cây.

### Disease Detection

- Phân loại bệnh.

### Confidence Score

- Hiển thị xác suất dự đoán.

### Decision Support

Sau khi dự đoán:

- mức độ nguy hiểm
- mức độ ưu tiên
- khả năng lây lan
- khuyến nghị xử lý
- biện pháp phòng ngừa

### Dashboard

Hiển thị:

- thống kê bệnh
- thống kê mô hình
- lịch sử dự đoán
- biểu đồ

### REST API

Cho phép các hệ thống khác sử dụng.

---

# 6. Non-functional Requirements

Hệ thống cần:

- dễ mở rộng
- dễ bảo trì
- module độc lập
- logging
- configuration
- versioning
- reproducible
- deployment bằng Docker

---

# 7. Success Metrics

## Machine Learning Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Inference Time
- Model Size

## Business Metrics

- thời gian phản hồi
- khả năng hỗ trợ quyết định
- độ ổn định API
- khả năng mở rộng

---

# 8. High-Level System Architecture

```
                        PlantVillage Dataset
                                 │
                                 ▼
                        Data Validation
                                 │
                                 ▼
                        Data Engineering
                                 │
                                 ▼
                               EDA
                                 │
                                 ▼
                     Image Preprocessing
                                 │
                                 ▼
                     Feature Engineering
                                 │
                                 ▼
                    Deep Learning Training
                                 │
                ┌────────────────┴───────────────┐
                ▼                                ▼
          EfficientNet                     Vision Transformer
                │                                │
                └────────────────┬───────────────┘
                                 ▼
                      Model Evaluation
                                 │
                                 ▼
                        Best Model Selection
                                 │
                                 ▼
                          Decision Engine
                                 │
                                 ▼
                            REST API
                                 │
                                 ▼
                          Web Dashboard
                                 │
                                 ▼
                               User
```

---

# 9. Data Flow

```
Raw Images

↓

Validation

↓

EDA

↓

Preprocessing

↓

Train / Validation / Test Split

↓

Model Training

↓

Model Evaluation

↓

Model Registry (MLflow)

↓

Prediction

↓

Decision Engine

↓

REST API

↓

Dashboard

↓

End User
```

---

# 10. Decision Support Flow

```
Leaf Image

↓

AI Prediction

↓

Disease

↓

Confidence Score

↓

Severity Assessment

↓

Priority Level

↓

Treatment Recommendation

↓

Prevention Recommendation

↓

Decision Report
```

---

# 11. Major Components

## 1. Data Engineering

Phụ trách:

- Data ingestion
- Validation
- Cleaning
- Transformation
- Versioning

---

## 2. Data Analysis

Bao gồm:

- EDA
- Data Quality
- Class Distribution
- Duplicate Detection
- Missing Image Detection

---

## 3. Deep Learning

Bao gồm:

- CNN
- Transfer Learning
- Vision Transformer
- Fine-tuning

---

## 4. Decision Engine

Chuyển kết quả AI thành:

- mức độ nguy hiểm
- khuyến nghị xử lý
- mức độ ưu tiên

---

## 5. API

FastAPI

RESTful API

Swagger

---

## 6. Dashboard

Streamlit

Hiển thị:

- prediction
- analytics
- charts

---

## 7. MLOps

Bao gồm:

- MLflow
- DVC
- Docker
- GitHub Actions
- Model Registry

---

# 12. Technology Stack

## Programming

- Python

## Data Engineering

- Pandas
- NumPy
- DVC

## Deep Learning

- PyTorch
- timm
- torchvision

## Visualization

- Matplotlib
- Plotly
- Streamlit

## API

- FastAPI

## Database

- PostgreSQL

## MLOps

- MLflow
- Docker
- GitHub Actions

## Version Control

- Git
- GitHub

---

# 13. High-Level Project Structure

```
smart-agriculture-dss/
│
├── api/
├── configs/
├── dashboard/
├── data/
├── deployment/
├── docs/
├── logs/
├── models/
├── notebooks/
├── pipelines/
├── scripts/
├── src/
├── tests/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
└── README.md
```

> **Lưu ý:** `Dockerfile` và cấu hình Docker nằm trong `deployment/docker/` theo SDS.
> `mlruns/` được Git-ignore, sinh tự động bởi MLflow khi chạy.
> `experiments/` không dùng — MLflow đảm nhiệm toàn bộ experiment tracking.

---

# 14. Risks

Có thể gặp:

- mất cân bằng dữ liệu
- overfitting
- ảnh chất lượng thấp
- nhiều bệnh có triệu chứng giống nhau
- thời gian train lâu
- model quá lớn

---

# 15. Final Deliverables

Sau khi hoàn thành, dự án phải có:

✓ Data Engineering Pipeline

✓ Data Validation

✓ EDA Report

✓ Deep Learning Models

✓ Vision Transformer

✓ Model Comparison

✓ Decision Support Engine

✓ REST API

✓ Dashboard

✓ Docker Deployment

✓ MLflow

✓ DVC

✓ GitHub Repository

✓ Documentation

✓ README chuyên nghiệp

Đây là một hệ thống Decision Support System hoàn chỉnh theo tiêu chuẩn Production, phù hợp làm Portfolio ứng tuyển Data Engineer hoặc Data Analyst.