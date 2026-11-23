#!/bin/sh
set -eu
python src/app.py &
exec python -m streamlit run src/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
