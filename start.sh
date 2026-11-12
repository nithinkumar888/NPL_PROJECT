#!/usr/bin/env bash
PORT="${PORT:-10000}"
echo "Starting Streamlit via start.sh on port $PORT..."
exec streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true

