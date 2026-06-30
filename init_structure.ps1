# Script khoi tao cau truc thu muc du an - Smart Agriculture DSS
# Theo dung SDS_DSS_Agriculture.md - Section 10: Final Project Tree
# Chay trong PowerShell, tai thu muc goc smart-agriculture-dss/

$ErrorActionPreference = "Stop"

Write-Host "Dang tao cau truc thu muc..."

# Danh sach tat ca thu muc can tao
$folders = @(
    "api/routers",
    "api/schemas",
    "configs",
    "dashboard/components",
    "dashboard/pages",
    "data/external",
    "data/processed",
    "data/raw",
    "deployment/docker",
    "deployment/nginx",
    "docs/architecture",
    "docs/data_dictionary",
    "docs/user_guide",
    "logs/api",
    "logs/system",
    "models",
    "notebooks",
    "pipelines",
    "scripts",
    "src/config",
    "src/data",
    "src/decision_engine",
    "src/evaluation",
    "src/features",
    "src/inference",
    "src/models",
    "src/services",
    "src/training",
    "src/utils",
    "tests/api",
    "tests/integration",
    "tests/unit"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
}

Write-Host "Da tao xong cau truc thu muc cap cao."

# Tao file __init__.py cho src/ (bien thanh Python package)
$initFiles = @(
    "src/__init__.py",
    "src/config/__init__.py",
    "src/data/__init__.py",
    "src/decision_engine/__init__.py",
    "src/evaluation/__init__.py",
    "src/features/__init__.py",
    "src/inference/__init__.py",
    "src/models/__init__.py",
    "src/services/__init__.py",
    "src/training/__init__.py",
    "src/utils/__init__.py"
)

foreach ($file in $initFiles) {
    New-Item -ItemType File -Path $file -Force | Out-Null
}

# Tao file rong khac
$emptyFiles = @(
    "api/main.py",
    "api/dependencies.py",
    "dashboard/app.py",
    "pipelines/etl_pipeline.py",
    "pipelines/train_pipeline.py",
    "tests/conftest.py",
    "configs/api.yaml",
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/logging.yaml",
    "configs/model.yaml",
    "configs/training.yaml"
)

foreach ($file in $emptyFiles) {
    New-Item -ItemType File -Path $file -Force | Out-Null
}

# Tao .gitkeep cho cac thu muc se bi gitignore noi dung (giu thu muc rong tren git)
$gitkeepFiles = @(
    "data/external/.gitkeep",
    "data/processed/.gitkeep",
    "data/raw/.gitkeep",
    "logs/api/.gitkeep",
    "logs/system/.gitkeep",
    "models/.gitkeep"
)

foreach ($file in $gitkeepFiles) {
    New-Item -ItemType File -Path $file -Force | Out-Null
}

Write-Host "Da tao xong toan bo file khoi tao."
Write-Host "Hoan tat khoi tao cau truc thu muc du an."
