"""run_server must work from a worker thread (the desktop app starts it that way)."""

import threading


def test_run_server_in_worker_thread(monkeypatch):
    import uvicorn

    from upscale import server

    called = threading.Event()
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: called.set())
    errors: list[BaseException] = []

    def target() -> None:
        try:
            server.run_server(host="127.0.0.1", port=0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=target)
    t.start()
    t.join(10)
    assert not errors, errors
    assert called.is_set()
