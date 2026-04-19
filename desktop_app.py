from __future__ import annotations

import socket
import threading
import time

import uvicorn


HOST = "127.0.0.1"
PORT = 8000


def wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def run_server() -> None:
    uvicorn.run("app.server:app", host=HOST, port=PORT, reload=False)


def main() -> None:
    import webview

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    if not wait_for_port(HOST, PORT):
        raise RuntimeError("Timed out while starting the local API server.")

    window = webview.create_window(
        "Workspace",
        f"http://{HOST}:{PORT}",
        width=1320,
        height=460,
        min_size=(720, 180),
    )
    webview.start()


if __name__ == "__main__":
    main()
