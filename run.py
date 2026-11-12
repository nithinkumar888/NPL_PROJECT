import os
import sys
import streamlit.web.cli as stcli

if __name__ == "__main__":
    port = os.environ.get("PORT", "10000")
    print(f"Starting Streamlit via run.py on port {port}...")
    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
    ]
    sys.exit(stcli.main())
