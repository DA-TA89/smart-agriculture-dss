# Deployment Guide — Smart Agriculture DSS

> Phase 11 — MLOps & Deployment. Phạm vi: demo local qua Docker Compose, KHÔNG deploy cloud thật (quyết định đã chốt khi bắt đầu Phase 11 — xem `PHASE11_LOG.md`).

Tài liệu này giả định bạn đã hoàn thành Phase 1–10 (checkpoint đã train ở `models/`, PostgreSQL local đã dùng ở Phase 9, Dashboard đã chạy được ở Phase 10). Các bước dưới đây là **việc mới của Phase 11**, thực hiện theo đúng thứ tự.

---

## 0. Chuẩn bị chung

Không đổi so với Phase 2–10: Windows, VS Code/PowerShell, venv, Python 3.13.2. Phase 11 cần thêm:

- **Docker Desktop** (bật WSL2 backend) — để chạy `docker-compose`.
- **Git** đã cấu hình `user.name`/`user.email` (đã có từ Phase 2).

---

## 1. Tạo GitHub repository thật (chưa có — bắt buộc để CI/CD chạy được)

1. Đăng nhập [github.com](https://github.com) bằng tài khoản `DA-TA89`.
2. Bấm **New repository**.
3. Điền:
   - Repository name: `smart-agriculture-dss`
   - Visibility: **Public**
   - **KHÔNG** tick "Add a README file" / ".gitignore" / "license" — repo local của bạn đã có sẵn tất cả, tick vào sẽ gây conflict khi push lần đầu.
4. Bấm **Create repository**. GitHub sẽ hiện trang hướng dẫn — bỏ qua, dùng lệnh dưới đây thay vì copy từ trang đó (vì repo bạn đã có commit sẵn từ Phase 2–10):

```powershell
cd C:\Project\smart-agriculture-dss
git remote add origin https://github.com/DA-TA89/smart-agriculture-dss.git
git branch -M master
git push -u origin master
```

5. Xác nhận: mở `https://github.com/DA-TA89/smart-agriculture-dss` trên trình duyệt, thấy đủ toàn bộ commit từ Phase 2 (`050936f`) tới commit Phase 10 (`8ce55a5`).

> ⚠️ Nếu lúc `git push` bị hỏi đăng nhập và bạn dùng mật khẩu tài khoản thay vì Personal Access Token — GitHub sẽ từ chối (không còn hỗ trợ password auth qua HTTPS từ 2021). Cần tạo Personal Access Token (Settings → Developer settings → Personal access tokens → Fine-grained tokens, quyền `Contents: Read and write`) và dùng token đó làm password khi Git hỏi.

---

## 2. Chuẩn bị GitHub Release cho model checkpoint (bắt buộc để CI build được image API)

**Vì sao cần bước này:** `models/*.pth` và `data/processed/class_to_idx.json` bị `.gitignore` (đúng thiết kế từ Phase 5/6 — không commit binary/generated file). Khi build Docker image trên máy bạn, `docker build` đọc thẳng từ ổ đĩa nên vẫn thấy các file này dù không có trong Git. Nhưng khi GitHub Actions build (`.github/workflows/ci-cd.yml`), runner chỉ có đúng những gì `git checkout` lấy về — **không có** các file này. Giải pháp: đăng file lên GitHub Release 1 lần, CI tải về trước khi build.

```powershell
cd C:\Project\smart-agriculture-dss

# Cài GitHub CLI nếu chưa có: winget install --id GitHub.cli
gh auth login

gh release create models-v1 `
  "models\efficientnet_b0_plantvillage_f1_0.97.pth" `
  "models\vit_tiny_plantvillage_f1_0.92.pth" `
  "data\processed\class_to_idx.json" `
  --title "Model checkpoints v1" `
  --notes "EfficientNet-B0 (macro-F1 0.9656) + ViT-Tiny (macro-F1 0.9216) - Phase 6/7. Dùng bởi CI/CD (Phase 11) để build Docker image API."
```

> Nếu tên file checkpoint thật trên máy bạn khác 2 tên trên (ví dụ đã train lại và tên đổi), báo lại — cần đồng bộ đúng tên trong `.github/workflows/ci-cd.yml` (bước "Download model checkpoints & class_to_idx.json").

---

## 3. DVC Remote Storage — Google Drive

```powershell
# Thêm gdrive extra vào dvc (chưa có trong requirements.txt hiện tại)
pip install "dvc[gdrive]==3.59.1"
```

Cập nhật `requirements.txt`, đổi dòng:
```
dvc==3.59.1
```
thành:
```
dvc[gdrive]==3.59.1
```

Tạo 1 thư mục trên Google Drive cá nhân (ví dụ "smart-agriculture-dss-dvc-storage"), mở thư mục đó, lấy **Folder ID** từ URL:
`https://drive.google.com/drive/folders/`**`1AbCdEfGhIjKlMnOpQrSt`** ← đoạn này

```powershell
cd C:\Project\smart-agriculture-dss
dvc remote add -d gdrive_remote gdrive://1AbCdEfGhIjKlMnOpQrSt
dvc remote modify gdrive_remote gdrive_acknowledge_abuse true
dvc push
```

Lệnh `dvc push` lần đầu sẽ mở trình duyệt yêu cầu đăng nhập Google + cấp quyền (dùng OAuth client mặc định của DVC, không cần tự tạo Google Cloud Project — phù hợp dự án cá nhân/portfolio). Sau khi cấp quyền, `data/raw.dvc` sẽ được đẩy lên Drive.

```powershell
git add requirements.txt .dvc/config
git commit -m "Phase 11: DVC remote storage - Google Drive"
git push
```

---

## 4. MLflow Model Registry — Promote lên Production

```powershell
python scripts\register_and_promote_model.py --model-key efficientnet
```

Kiểm tra kết quả:
```powershell
mlflow ui --backend-store-uri file:./mlruns
```
Mở `http://localhost:5000` → tab **Models** → `plantvillage-disease-classifier` → xác nhận version mới nhất có alias `production`.

---

## 5. Chạy hệ thống bằng Docker Compose

```powershell
cd C:\Project\smart-agriculture-dss
Copy-Item .env.docker.example .env.docker
notepad .env.docker   # điền DB_PASSWORD, API_SECRET_KEY thật

docker-compose up --build
```

Kiểm tra:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- `docker-compose ps` → cả 3 service (`sadss-postgres`, `sadss-api`, `sadss-dashboard`) đều `Up`/`healthy`.

Dừng hệ thống: `docker-compose down` (thêm `-v` nếu muốn xoá luôn volume Postgres, tức xoá sạch lịch sử prediction đã lưu trong lần demo).

> Dashboard trỏ tới API bên trong mạng Docker qua `configs/dashboard.docker.yaml` (COPY đè lên `configs/dashboard.yaml` lúc build `Dockerfile.dashboard`) — đã xác nhận đúng cơ chế thật của `dashboard/api_client.py -> get_api_base_url()` (đọc từ YAML, không phải biến môi trường). Không sửa `configs/dashboard.yaml` gốc dùng cho dev ngoài Docker.

---

## 6. CI/CD — GitHub Actions

Sau khi hoàn thành bước 1–2 và `git push`, vào tab **Actions** trên GitHub để xem pipeline `CI/CD Pipeline` tự chạy:
- Job `lint-and-test`: cài dependency, lint `tests/` (strict), lint `src/api/pipelines` (chỉ cảnh báo), chạy `pytest tests/unit`.
- Job `build-and-push`: chỉ chạy khi push vào `master`, tải checkpoint từ GitHub Release (bước 2), build + push 2 image lên `ghcr.io/DA-TA89/smart-agriculture-dss-api` và `...-dashboard`.

Xem image đã push tại: `https://github.com/DA-TA89?tab=packages`.

---

## 7. Việc còn treo / cần xác nhận thêm

- Nội dung `disease_kb.json` (Phase 8) vẫn chưa được chuyên gia nông nghiệp review — không liên quan tới việc deploy demo local, nhưng nhắc lại vì Phase 11 là bước gần "production" nhất tính tới nay.
- `deployment/nginx/` — để trống theo quyết định đã chốt (ngoài phạm vi demo local), có thể bổ sung sau nếu có nhu cầu deploy cloud thật.
- **Chưa verify thật trên máy Windows** — toàn bộ file Phase 11 (Dockerfile, docker-compose, CI/CD, script MLflow, test) mới được viết dựa trên source code đã xem qua chat (`requirements.txt`, `main.py`, `confidence_classifier.py`, `priority_calculator.py`, `knowledge_base_loader.py`, `api_client.py`) + nội dung mô tả trong các `PHASE*_LOG.md` — **chưa có lần `docker-compose up --build` thật nào chạy thành công** để xác nhận. Rất có thể phát sinh lỗi nhỏ (thiếu package hệ thống, sai đường dẫn COPY, sai tên biến env mà `src/database/session.py` thật sự dùng...) khi chạy lần đầu — đây là điều bình thường, báo lại log lỗi để sửa tiếp, đúng quy trình đã áp dụng xuyên suốt dự án từ Phase 3.
