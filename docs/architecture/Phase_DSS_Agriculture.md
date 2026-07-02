# Roadmap triển khai dự án
# Smart Agriculture Decision Support System
## Using Deep Learning & Vision Transformer

---

# Mục tiêu

Chia toàn bộ dự án thành các giai đoạn triển khai theo đúng quy trình của một dự án AI Production.

Mỗi Phase đều có:

- Mục tiêu
- Kiến thức cần học
- Công nghệ sử dụng
- Deliverables
- Khó khăn thường gặp
- Checklist hoàn thành

Lưu ý:

- Chỉ hoàn thành Phase hiện tại mới chuyển sang Phase tiếp theo.
- Không triển khai song song.
- Mỗi Phase đều phải có commit GitHub riêng.

---

# Phase 1 — Business Understanding & Project Planning

## Mục tiêu

Hiểu rõ bài toán nghiệp vụ trước khi bắt đầu phát triển hệ thống.

Xác định:

- Business Problem
- Scope
- Stakeholders
- Objectives
- Success Metrics
- Project Architecture

## Kiến thức cần học

- Decision Support System
- Smart Agriculture
- Machine Learning Workflow
- CRISP-DM

## Công nghệ

- Markdown
- Draw.io
- Mermaid

## Deliverables

- Architecture.md
- Requirement Specification
- Project Scope
- Use Case Diagram
- System Architecture

## Khó khăn

- Không xác định rõ phạm vi dự án
- Thiết kế kiến trúc chưa hợp lý

## Checklist

- [ ] Xác định Business Problem
- [ ] Xác định Success Metrics
- [ ] Hoàn thành Architecture
- [ ] Hoàn thành Project Blueprint

---

# Phase 2 — Project Initialization

## Mục tiêu

Khởi tạo toàn bộ môi trường phát triển.

## Kiến thức cần học

- Git
- GitHub
- Python Environment
- Project Structure

## Công nghệ

- Git
- GitHub
- Python
- Conda hoặc venv

## Deliverables

- Repository GitHub
- Folder Structure
- requirements.txt
- README.md

## Khó khăn

- Quản lý môi trường
- Cấu trúc thư mục

## Checklist

- [ ] Khởi tạo Git Repository
- [ ] Thiết lập Virtual Environment
- [ ] Tạo cấu trúc thư mục
- [ ] Push commit đầu tiên

---

# Phase 3 — Data Engineering

## Mục tiêu

Xây dựng Data Pipeline từ dữ liệu gốc đến dữ liệu sẵn sàng phân tích.

## Kiến thức cần học

- Data Engineering
- ETL
- Image Dataset Management
- Metadata

## Công nghệ

- Python
- Pandas
- NumPy
- DVC

## Deliverables

- Data Pipeline
- Metadata
- Data Validation Report

## Khó khăn

- Dữ liệu lỗi
- Ảnh trùng
- Thiếu nhãn

## Checklist

- [ ] Download Dataset
- [ ] Validate Dataset
- [ ] Clean Dataset
- [ ] Sinh Metadata
- [ ] Khởi tạo DVC & versioning data

---

# Phase 4 — Exploratory Data Analysis (EDA)

## Mục tiêu

Hiểu rõ bộ dữ liệu trước khi huấn luyện.

## Kiến thức cần học

- Data Analysis
- Data Visualization
- Image Dataset Analysis

## Công nghệ

- Pandas
- Matplotlib
- Plotly

## Deliverables

- EDA Report
- Dataset Statistics
- Visualization Report

## Khó khăn

- Dataset mất cân bằng
- Nhiều class

## Checklist

- [ ] Phân bố class
- [ ] Kích thước ảnh
- [ ] Duplicate Analysis
- [ ] Missing Analysis
- [ ] Visualization

---

# Phase 5 — Data Preprocessing

## Mục tiêu

Chuẩn hóa dữ liệu đầu vào cho Deep Learning.

## Kiến thức cần học

- Image Processing
- Data Augmentation
- Dataset Split

## Công nghệ

- OpenCV
- Pillow
- Torchvision

## Deliverables

- Processed Dataset
- Train Set
- Validation Set
- Test Set

## Khó khăn

- Overfitting
- Class imbalance

## Checklist

- [ ] Resize
- [ ] Normalize
- [ ] Dataset Split
- [ ] Data Augmentation
- [ ] DataLoader

---

# Phase 6 — Deep Learning Model Development

## Mục tiêu

Xây dựng và huấn luyện các mô hình AI.

## Kiến thức cần học

- CNN
- Transfer Learning
- Vision Transformer

## Công nghệ

- PyTorch
- timm

## Deliverables

- Trained Models
- Checkpoints

## Khó khăn

- GPU Memory
- Overfitting
- Hyperparameter Tuning

## Checklist

- [ ] EfficientNet
- [ ] Vision Transformer
- [ ] Training Pipeline
- [ ] Checkpoint Saving

---

# Phase 7 — Model Evaluation & Experiment Tracking

## Mục tiêu

Đánh giá, so sánh và lựa chọn mô hình tốt nhất.

> **Lưu ý phạm vi MLflow:**
> Phase 7 dùng MLflow để **Experiment Tracking** (log metrics, params, artifacts trong quá trình train và eval).
> Phase 11 dùng MLflow **Model Registry** (đăng ký, staging, production promotion cho deployment).

## Kiến thức cần học

- Classification Metrics
- Model Comparison
- Experiment Tracking

## Công nghệ

- Scikit-learn
- MLflow

## Deliverables

- Evaluation Report
- Confusion Matrix
- Best Model
- MLflow Experiments

## Khó khăn

- Chọn metric phù hợp
- So sánh nhiều mô hình

## Checklist

- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] F1-score
- [ ] ROC-AUC
- [ ] Confusion Matrix
- [ ] Inference Time
- [ ] Model Size
- [ ] Model Comparison Report
- [ ] Log experiments vào MLflow

---

# Phase 8 — Decision Support System

## Mục tiêu

Chuyển kết quả dự đoán thành khuyến nghị hỗ trợ quyết định.

## Kiến thức cần học

- Rule-based System
- Knowledge Base
- Business Logic

## Công nghệ

- Python
- JSON
- CSV

## Deliverables

- Knowledge Base
- Recommendation Engine
- Decision Report

## Khó khăn

- Thiết kế luật nghiệp vụ
- Chuẩn hóa khuyến nghị

## Checklist

- [ ] Disease Information
- [ ] Severity Level
- [ ] Treatment Recommendation
- [ ] Prevention Recommendation
- [ ] Decision Report

---

# Phase 9 — REST API & Database

## Mục tiêu

Xây dựng Backend phục vụ Dashboard.

## Kiến thức cần học

- REST API
- Database
- FastAPI

## Công nghệ

- FastAPI
- PostgreSQL
- SQLAlchemy

## Deliverables

- API
- Database
- Swagger

## Khó khăn

- Thiết kế API
- Quản lý Model

## Checklist

- [ ] Prediction API
- [ ] History API
- [ ] Database
- [ ] Swagger

---

# Phase 10 — Dashboard & Analytics

## Mục tiêu

Xây dựng Dashboard trực quan.

## Kiến thức cần học

- Dashboard Design
- Data Visualization

## Công nghệ

- Streamlit
- Plotly

## Deliverables

- Dashboard
- Charts
- Prediction Interface

## Khó khăn

- UX
- API Integration

## Checklist

- [ ] Upload Image
- [ ] Prediction
- [ ] Statistics
- [ ] Analytics
- [ ] History

---

# Phase 11 — MLOps & Deployment

## Mục tiêu

Đưa dự án lên chuẩn Production.

> **Lưu ý phạm vi MLflow:**
> Phase 11 dùng MLflow **Model Registry**: đăng ký model đã chọn từ Phase 7, quản lý vòng đời (Staging → Production), phục vụ deployment.

## Kiến thức cần học

- Docker
- MLflow Model Registry
- DVC Remote Storage
- CI/CD

## Công nghệ

- Docker
- Docker Compose
- GitHub Actions
- MLflow
- DVC

## Deliverables

- Docker Image
- CI/CD Pipeline
- Model Registry
- Deployment Guide

## Khó khăn

- Versioning
- Deployment

## Checklist

- [ ] Dockerfile cho từng service (API, Dashboard)
- [ ] Docker Compose toàn hệ thống
- [ ] DVC Remote Storage
- [ ] MLflow Model Registry (Staging → Production)
- [ ] GitHub Actions CI/CD Pipeline
- [ ] Cấu hình `.env` production
- [ ] Deployment Guide

---

# Kết quả cuối cùng

Sau khi hoàn thành toàn bộ roadmap, dự án sẽ bao gồm:

✅ Data Engineering Pipeline

✅ Data Validation

✅ Exploratory Data Analysis

✅ Image Preprocessing Pipeline

✅ Deep Learning Models

✅ Vision Transformer

✅ Model Evaluation

✅ Experiment Tracking

✅ Decision Support System

✅ Knowledge Base

✅ REST API

✅ PostgreSQL Database

✅ Dashboard

✅ Docker Deployment

✅ MLflow

✅ DVC

✅ GitHub Actions

✅ Documentation

✅ Production-ready Portfolio