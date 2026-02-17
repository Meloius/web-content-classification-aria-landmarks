#!/usr/bin/env bash
source .venv/bin/activate
python -m playwright install
streamlit run app/app.py
