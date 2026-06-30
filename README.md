# 🌱 Smart Agriculture Decision Support System

### Plant Disease Detection & Decision Support using Deep Learning and Vision Transformer

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

---

## 📋 Overview

A production-grade **Decision Support System (DSS)** that detects plant leaf diseases from images and translates AI predictions into actionable recommendations for farmers — severity assessment, treatment guidance, and prevention measures.

Built end-to-end across the full ML lifecycle: data engineering, exploratory data analysis, deep learning (CNN + Vision Transformer), model evaluation, a rule-based decision engine, a REST API, an interactive dashboard, and full MLOps deployment.

This project is developed as a **portfolio project** targeting **Data Engineer** and **Data Analyst** roles, following production-standard software design practices.

---

## 🎯 Business Problem

Farmers often rely on personal experience to detect crop diseases, leading to:

- Late detection at early disease stages
- Confusion between visually similar diseases
- Limited access to local agricultural experts
- High cost of manual inspection
- Rapid disease spread due to delayed response
- Misuse of pesticides, causing waste and environmental harm

This system addresses these challenges by providing automated disease detection from leaf images, prediction confidence scoring, and decision support recommendations to reduce crop loss and optimize yield.

---

## 🏗️ System Architecture

```
PlantVillage Dataset
        │
        ▼
Data Validation → Data Engineering → EDA → Image Preprocessing
        │
        ▼
   Deep Learning Training
        │
   ┌────┴────┐
   ▼         ▼
EfficientNet  Vision Transformer
   │         │
   └────┬────┘
        ▼
  Model Evaluation → Best Model Selection
        │
        ▼
   Decision Engine
        │
        ▼
    REST API (FastAPI)
        │
        ▼
  Web Dashboard (Streamlit)
        │
        ▼
      End User
```

The system follows a **Layered Modular Architecture**:

`Presentation → Application → Business → AI → Infrastructure`

For full architectural details, see [`docs/architecture/`](docs/architecture/).

---

## ✨ Features

- **Image Upload** — Upload leaf images for analysis
- **Disease Detection** — Multi-class plant disease classification
- **Confidence Score** — Prediction probability for transparency
- **Decision Support** — Severity level, priority level, spread risk, treatment and prevention recommendations
- **Dashboard** — Disease statistics, model analytics, prediction history, interactive charts
- **REST API** — Swagger-documented API for third-party integration

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.13 |
| **Deep Learning** | PyTorch, timm, torchvision |
| **Data Engineering** | Pandas, NumPy, DVC |
| **Visualization** | Matplotlib, Plotly |
| **API** | FastAPI |
| **Dashboard** | Streamlit |
| **Database** | PostgreSQL |
| **MLOps** | MLflow, Docker, GitHub Actions |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure

```
smart-agriculture-dss/
├── api/                # FastAPI REST API source code
├── configs/            # YAML configuration files
├── dashboard/          # Streamlit dashboard source code
├── data/                # Dataset (DVC-managed, Git-ignored)
├── deployment/          # Docker, Nginx, CI/CD configuration
├── docs/                # Technical documentation & architecture diagrams
├── logs/                # Runtime logs (Git-ignored)
├── models/              # Model weights/checkpoints (Git-ignored)
├── notebooks/            # Jupyter notebooks for EDA & experimentation
├── pipelines/            # Automated pipeline scripts (ETL, training)
├── scripts/              # Utility shell scripts
├── src/                  # Core AI and business logic
│   ├── config/            # Configuration loading
│   ├── data/               # Dataset & DataLoader logic
│   ├── features/           # Image preprocessing & augmentation
│   ├── models/              # Neural network architectures (ViT, CNN)
│   ├── training/             # Training loop, optimizer, loss, scheduler
│   ├── evaluation/            # Metrics & confusion matrix
│   ├── inference/              # Model loading & real-time prediction
│   ├── decision_engine/         # Knowledge base & recommendation logic
│   ├── services/                 # Business services layer
│   └── utils/                     # Shared utilities (logger, formatters)
├── tests/                  # Unit, integration, and API tests
├── .env.example             # Environment variable template
├── .gitignore
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
└── README.md
```

> Full structure specification: see [SDS (Software Design Specification)](docs/architecture/).

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- NVIDIA GPU with CUDA support (recommended for training)
- Git
- Docker (for deployment, later phases)

### 1. Clone the repository

```bash
git clone https://github.com/DA-TA89/smart-agriculture-dss.git
cd smart-agriculture-dss
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA support (GPU users)

PyTorch must be installed separately to match your CUDA version. Check your CUDA version first, then install via the official index, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to select the command matching your system.

### 5. Configure environment variables

```bash
# Copy the template and fill in real values
cp .env.example .env
```

### 6. Run the project

> Instructions for running the API, dashboard, and training pipelines will be added as each phase is completed.

---

## 📊 Dataset

**[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)**

Tens of thousands of RGB leaf images covering multiple crop species, multiple disease classes, and a healthy class.

- **Input:** Leaf image
- **Output:** Disease classification

---

## 🗺️ Project Roadmap

This project is developed in 11 sequential phases, following a production AI project workflow:

| Phase | Description | Status |
|---|---|---|
| 1 | Business Understanding & Project Planning | ✅ Done |
| 2 | Project Initialization | 🔄 In Progress |
| 3 | Data Engineering | ⬜ Pending |
| 4 | Exploratory Data Analysis (EDA) | ⬜ Pending |
| 5 | Data Preprocessing | ⬜ Pending |
| 6 | Deep Learning Model Development | ⬜ Pending |
| 7 | Model Evaluation & Experiment Tracking | ⬜ Pending |
| 8 | Decision Support System | ⬜ Pending |
| 9 | REST API & Database | ⬜ Pending |
| 10 | Dashboard & Analytics | ⬜ Pending |
| 11 | MLOps & Deployment | ⬜ Pending |

Full roadmap details available in [`docs/`](docs/).

---

## 📈 Success Metrics

**Machine Learning:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, Inference Time, Model Size

**Business:** Response time, decision support effectiveness, API stability, scalability

---

## 🤝 Contributing

This is a personal portfolio project and is not currently open for external contributions. Feedback and suggestions are welcome via Issues.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Phạm Tuấn Anh**

- 📧 Email: [anh.pt8925@gmail.com](mailto:anh.pt8925@gmail.com)
- 💻 GitHub: [@DA-TA89](https://github.com/DA-TA89)
