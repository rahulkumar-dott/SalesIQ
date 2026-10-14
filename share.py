"""Run the API + UI and expose it through an ngrok tunnel.

    .venv\\Scripts\\python.exe share.py

Needs your ngrok authtoken (`ngrok config add-authtoken ...`, already in
%LOCALAPPDATA%\\ngrok\\ngrok.yml) and SALESIQ_PASSWORD in .env: the tunnel is
refused without a password because the app serves customer transcripts and can
start paid LLM jobs. Viewers log in with user SALESIQ_USER (default "salesiq").
Ctrl+C stops the tunnel and the server.
"""
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import config  # noqa: F401  (loads .env)

PORT = 8000


def main() -> None:
    if not os.environ.get("SALESIQ_PASSWORD") and os.environ.get("SALESIQ_PUBLIC") != "1":
        sys.exit("Set SALESIQ_PASSWORD in .env, or SALESIQ_PUBLIC=1 to deliberately allow open access.")
    if not (Path(__file__).parent / "ui" / "dist").exists():
        print("note: ui/dist missing, only the API will be served (run `npm run build` in ui/)")

    import uvicorn
    from pyngrok import conf, ngrok

    import api

    cfg = Path(os.environ.get("LOCALAPPDATA", "")) / "ngrok" / "ngrok.yml"
    exe = os.environ.get("NGROK_PATH") or shutil.which("ngrok")  # use an installed ngrok; else pyngrok downloads one
    if not exe:
        print("ngrok.exe not found on PATH; pyngrok will try to download it (set NGROK_PATH to skip)", flush=True)
    pc = conf.PyngrokConfig(
        config_path=str(cfg) if cfg.exists() else None,
        ngrok_path=exe,
    )

    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.2)

    tunnel = ngrok.connect(PORT, "http", pyngrok_config=pc)
    print(f"\nPublic URL : {tunnel.public_url}")
    if os.environ.get("SALESIQ_PASSWORD"):
        print(f"Login      : user '{os.environ.get('SALESIQ_USER', 'salesiq')}' / SALESIQ_PASSWORD from .env")
    else:
        print("Login      : none, anyone with the link can use it (LLM spend cap still applies)")
    print("API docs   : <URL>/docs      Ctrl+C to stop\n", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        server.should_exit = True


if __name__ == "__main__":
    main()
