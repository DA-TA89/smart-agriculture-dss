#!/usr/bin/env bash
# ==============================================================================
# entrypoint-api.sh - Smart Agriculture DSS API container entrypoint
# Phase 11 - MLOps & Deployment
# ==============================================================================
# Thứ tự thực hiện:
#   1. Chờ PostgreSQL sẵn sàng nhận kết nối (dù docker-compose.yml đã có
#      healthcheck "depends_on: condition: service_healthy", vẫn thêm 1 lớp
#      retry ở đây để an toàn - healthcheck của Postgres image xác nhận
#      server đã lên, nhưng không đảm bảo tuyệt đối request đầu tiên luôn
#      thành công ngay mili-giây đó).
#   2. Chạy Alembic migration (tạo/cập nhật bảng "predictions" - Phase 9)
#      để không cần chạy tay `alembic upgrade head` sau khi container start.
#   3. Start uvicorn.
#
# ⚠️ GIẢ ĐỊNH CẦN XÁC NHẬN: script này giả định
# src/database/session.py -> _build_database_url() build connection string
# theo pattern chuẩn SQLAlchemy:
#   postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}
# (khớp đúng tên 5 biến trong .env.example). Nếu thực tế
# src/database/session.py dùng driver/format khác, đoạn Python chờ kết nối
# dưới đây (dùng psycopg2 trực tiếp, KHÔNG qua SQLAlchemy) vẫn hoạt động độc
# lập vì chỉ cần đúng 5 biến DB_* - chỉ có bước "alembic upgrade head" mới
# phụ thuộc vào việc alembic/env.py build đúng URL, và đó là code Phase 9 đã
# có sẵn, không phải code mới của Phase 11.
# ==============================================================================
set -e

echo "[entrypoint] Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT} ..."
python - <<'PYWAIT'
import os
import sys
import time

import psycopg2

max_attempts = 30
delay_seconds = 2

for attempt in range(1, max_attempts + 1):
    try:
        conn = psycopg2.connect(
            host=os.environ["DB_HOST"],
            port=os.environ["DB_PORT"],
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            connect_timeout=3,
        )
        conn.close()
        print("[entrypoint] PostgreSQL is ready.")
        sys.exit(0)
    except psycopg2.OperationalError as exc:
        print(f"[entrypoint] Attempt {attempt}/{max_attempts} failed: {exc}")
        time.sleep(delay_seconds)

print("[entrypoint] PostgreSQL did not become ready in time.", file=sys.stderr)
sys.exit(1)
PYWAIT

echo "[entrypoint] Running Alembic migrations (alembic upgrade head) ..."
alembic upgrade head

echo "[entrypoint] Starting uvicorn ..."
exec uvicorn api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}"
