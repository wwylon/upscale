"""Windows desktop shell: native window around the local Upscale server.

Frozen (PyInstaller) builds use %LOCALAPPDATA%\\Upscale for models and open
the bundled UI with no separate Python install required.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

HOST = "127.0.0.1"
PORT = int(os.environ.get("UPSCALE_PORT", "8787"))
URL = f"http://{HOST}:{PORT}/en/"


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _models_dir() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        p = Path(local) / "Upscale" / "models"
    else:
        p = _app_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _prepare_env() -> None:
    root = _app_dir()
    # Prefer bundled package data next to the exe
    os.environ.setdefault("UPSCALE_ROOT", str(root))
    os.environ["UPSCALE_MODELS_DIR"] = str(_models_dir())
    # Frozen desktop ships without Torch — always use ONNX Runtime
    if getattr(sys, "frozen", False):
        os.environ["UPSCALE_BACKEND"] = "onnx"
    else:
        os.environ.setdefault("UPSCALE_BACKEND", os.environ.get("UPSCALE_BACKEND", "auto"))
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # PyInstaller unpack dir
        meipass = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        os.environ.setdefault("UPSCALE_UI_DIR", str(meipass / "ui"))
        os.environ.setdefault("UPSCALE_DOCS_DIR", str(meipass / "docs"))
        os.environ.setdefault("UPSCALE_LOCALES_DIR", str(meipass / "locales"))


def _start_server() -> None:
    from upscaler.server import run_server

    run_server(host=HOST, port=PORT)


def main() -> None:
    _prepare_env()
    try:
        import webview
    except ImportError as exc:
        raise SystemExit(
            "Desktop extras required: pip install 'upscaler[desktop,server]'\n"
            f"({exc})"
        ) from exc

    thread = threading.Thread(target=_start_server, name="upscale-uvicorn", daemon=True)
    thread.start()

    deadline = time.time() + 60.0
    while time.time() < deadline:
        try:
            import urllib.request

            with urllib.request.urlopen(URL, timeout=1) as resp:
                if resp.status == 200:
                    break
        except OSError:
            time.sleep(0.25)
    else:
        raise SystemExit(f"Server did not start on {URL}")

    win_ref: dict[str, object] = {"w": None}

    def _js(script: str) -> None:
        w = win_ref["w"]
        if w is None:
            return
        try:
            w.evaluate_js(script)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _menu_about() -> None:
        _js("window.UpscaleUI&&window.UpscaleUI.showAbout&&window.UpscaleUI.showAbout()")

    def _menu_license() -> None:
        _js("window.UpscaleUI&&window.UpscaleUI.showLicense&&window.UpscaleUI.showLicense()")

    def _menu_updates() -> None:
        _js("window.UpscaleUI&&window.UpscaleUI.checkUpdates&&window.UpscaleUI.checkUpdates()")

    def _menu_models() -> None:
        _js(
            "(function(){var p=location.pathname||'/';var i=p.lastIndexOf('/');"
            "location.href=(i>=0?p.slice(0,i+1):'/')+'library.html';})()"
        )

    def _menu_open_models() -> None:
        import subprocess

        path = _models_dir()
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError:
            pass

    menu = None
    try:
        from webview.menu import Menu, MenuAction

        menu = [
            Menu(
                "Help",
                [
                    MenuAction("Check for updates…", _menu_updates),
                    MenuAction("License", _menu_license),
                    MenuAction("About Upscale", _menu_about),
                ],
            ),
            Menu(
                "Models",
                [
                    MenuAction("Model library…", _menu_models),
                    MenuAction("Open models folder", _menu_open_models),
                ],
            ),
        ]
    except Exception:
        menu = None

    window = webview.create_window(
        "Upscale",
        URL,
        width=1400,
        height=900,
        min_size=(1000, 700),
        menu=menu,
    )
    win_ref["w"] = window

    def _on_loaded() -> None:
        # Expose a tiny desktop notify hook for job completion toasts.
        _js(
            "window.UpscaleDesktop=window.UpscaleDesktop||{};"
            "window.UpscaleDesktop.notifyDone=function(name){"
            "try{if(window.external&&window.external.invoke)"
            "window.external.invoke('done:'+name);}catch(e){}"
            "document.title='Upscale — done: '+(name||'');"
            "setTimeout(function(){document.title='Upscale';},4000);"
            "};"
        )

    def _on_closed() -> None:
        # Process exit ends the daemon uvicorn thread
        pass

    try:
        window.events.loaded += _on_loaded  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        window.events.closed += _on_closed  # type: ignore[attr-defined]
    except Exception:
        pass

    webview.start()


if __name__ == "__main__":
    main()
