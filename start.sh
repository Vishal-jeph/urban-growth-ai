#!/usr/bin/env bash
set -e

uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &

exec streamlit run app/frontend/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port 8501
