from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from png_fixtures import TINY_PNG
from upscaler.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_system(client: TestClient):
    r = client.get("/api/system")
    assert r.status_code == 200
    data = r.json()
    assert "device" in data
    assert "cuda_available" in data
    assert "auto_tile" in data
    assert data["vram_fit"] in ("ok", "tight", "spill")


def test_api_system_vram_fit_query(client: TestClient):
    r = client.get("/api/system", params={"model": "RealESRGAN_x4plus", "tile": 512})
    assert r.status_code == 200
    data = r.json()
    assert data["vram_fit"] in ("ok", "tight", "spill")
    assert data["vram_fit_tile"] == 512


def test_api_help_filters(client: TestClient):
    r = client.get("/api/help/filters")
    assert r.status_code == 200
    body = r.json()
    assert body["topic"] == "filters"
    assert "Filters" in body["markdown"]


def test_api_models(client: TestClient):
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()
    assert isinstance(models, list)
    assert len(models) >= 1
    names = {m["name"] for m in models}
    assert "RealESRGAN_x4plus" in names
    assert all("status" in m for m in models)


def test_api_algorithms(client: TestClient):
    r = client.get("/api/algorithms")
    assert r.status_code == 200
    body = r.json()
    assert "nearest" in body["algorithms"]
    assert "lanczos3" in body["algorithms"]
    assert "texture_residual" in body["algorithms"]
    assert body["backend"] in ("opencv", "pillow")
    items = {it["name"]: it for it in body["items"]}
    assert items["lanczos3"]["group"] == "builtin"
    assert items["texture_residual"]["group"] == "paper"
    assert items["texture_residual"]["label"]
    assert "description" in items["texture_residual"]


def test_api_about_license_updates(client: TestClient):
    about = client.get("/api/about").json()
    assert about["name"] == "Upscale"
    assert "version" in about
    assert "models" in about
    lic = client.get("/api/license").json()
    assert lic["license"] == "MIT"
    assert "MIT" in lic["text"]
    upd = client.get("/api/updates").json()
    assert upd["status"] in ("ok", "unconfigured", "error")
    assert "current" in upd


def test_create_job_mock_execute(client: TestClient, tmp_path: Path):
    out = tmp_path / "out.png"
    out.write_bytes(TINY_PNG)

    def fake_execute(cfg, **kwargs):
        if kwargs.get("on_progress"):
            kwargs["on_progress"]({"type": "loading_model", "pct": 0.02, "message": "Loading model…"})
            kwargs["on_progress"]({"type": "tile", "pass": 1, "passes": 1, "tile": 1, "tiles": 1, "pct": 0.5})
            kwargs["on_progress"]({"type": "done", "pct": 1.0, "message": "done"})
        return out

    with patch("upscaler.server.execute", side_effect=fake_execute):
        r = client.post(
            "/api/jobs",
            files={"file": ("test.png", TINY_PNG, "image/png")},
            data={"settings": '{"scale": 2, "model": "RealESRGAN_x4plus", "device": "cpu"}'},
        )
    assert r.status_code == 200
    job_id = r.json()["id"]
    assert job_id

    prev = client.get(f"/api/jobs/{job_id}/preview")
    assert prev.status_code == 200

    result = None
    for _ in range(100):
        result = client.get(f"/api/jobs/{job_id}/result")
        if result.status_code == 200:
            break
        time.sleep(0.05)
    assert result is not None and result.status_code == 200
    disp = client.get(f"/api/jobs/{job_id}/display")
    assert disp.status_code == 200
    assert disp.headers.get("content-type", "").startswith("image/")


def test_job_with_crop_and_filters(client: TestClient, tmp_path: Path):
    out = tmp_path / "out.png"
    out.write_bytes(b"fake")
    seen = {}

    def fake_execute(cfg, **kwargs):
        seen["crop"] = cfg.crop
        seen["filters"] = cfg.filters
        if kwargs.get("on_progress"):
            kwargs["on_progress"]({"type": "done", "pct": 1.0, "message": "done"})
        return out

    settings = {
        "scale": 4,
        "model": "4x-spanx4-ch48",
        "device": "cpu",
        "crop": {"x": 0, "y": 0, "w": 32, "h": 32},
        "filters": {"pre_denoise": 0.4, "jpeg_clean": 0.0, "edge": 0.2, "grain": 0.0},
    }
    with patch("upscaler.server.execute", side_effect=fake_execute):
        r = client.post(
            "/api/jobs",
            files={"file": ("test.png", TINY_PNG, "image/png")},
            data={"settings": json.dumps(settings)},
        )
    assert r.status_code == 200
    for _ in range(100):
        if client.get(f"/api/jobs/{r.json()['id']}/result").status_code == 200:
            break
        time.sleep(0.05)
    assert seen.get("crop") == (0, 0, 32, 32)
    assert seen["filters"].pre_denoise == 0.4
    assert seen["filters"].edge == 0.2


def test_pause_cancel_endpoints(client: TestClient, tmp_path: Path):
    out = tmp_path / "out.png"
    out.write_bytes(b"fake")
    started = threading.Event()
    release = threading.Event()

    def fake_execute(cfg, **kwargs):
        started.set()
        release.wait(timeout=2)
        if kwargs.get("on_progress"):
            kwargs["on_progress"]({"type": "done", "pct": 1.0, "message": "done"})
        return out

    with patch("upscaler.server.execute", side_effect=fake_execute):
        r = client.post(
            "/api/jobs",
            files={"file": ("test.png", TINY_PNG, "image/png")},
            data={"settings": '{"scale": 2, "model": "RealESRGAN_x4plus", "device": "cpu"}'},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        assert started.wait(timeout=10), "worker did not start job"
        assert client.post(f"/api/jobs/{job_id}/pause").status_code == 200
        assert client.post(f"/api/jobs/{job_id}/resume").status_code == 200
        assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
        release.set()


def test_api_estimate(client: TestClient):
    r = client.post(
        "/api/estimate",
        json={"src_w": 256, "src_h": 256, "scale": 4, "model": "RealESRGAN_x4plus", "device": "cpu", "model_ready": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["seconds_high"] >= body["seconds_low"] > 0
    assert body["model_ready"] is False
    assert body["vram_fit"] in ("ok", "tight", "spill")


def test_api_decode_preview_tga(client: TestClient, tmp_path: Path):
    from PIL import Image

    import numpy as np

    tga = tmp_path / "sample.tga"
    Image.fromarray(np.full((20, 24, 3), 40, dtype=np.uint8), mode="RGB").save(tga)
    r = client.post(
        "/api/decode-preview",
        files={"file": ("sample.tga", tga.read_bytes(), "image/x-targa")},
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("image/png")
    assert int(r.headers.get("X-Image-Width", 0)) == 24
    assert int(r.headers.get("X-Image-Height", 0)) == 20
    assert r.headers.get("X-Has-Alpha") == "0"


def test_api_rejects_svg(client: TestClient):
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'
    r = client.post(
        "/api/decode-preview",
        files={"file": ("icon.svg", svg, "image/svg+xml")},
    )
    assert r.status_code == 400
    assert "vector" in r.text.lower() or "svg" in r.text.lower()


def test_api_models_add(client: TestClient, tmp_path: Path):
    fake = tmp_path / "custom.safetensors"
    fake.write_bytes(b"x" * 16)
    r = client.post(
        "/api/models",
        json={"name": "test-custom-model", "path": str(fake), "scale": 4},
    )
    assert r.status_code == 200
    names = {m["name"] for m in client.get("/api/models").json()}
    assert "test-custom-model" in names


def test_root_redirects_to_locale(client: TestClient):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/en/"


def test_locale_root_serves_simple(client: TestClient):
    r = client.get("/en/")
    assert r.status_code == 200
    assert b"Advanced view" in r.content or b"simple" in r.content.lower()


def test_locale_root_respects_view_cookie(client: TestClient):
    client.cookies.set("upscale_view", "advanced")
    r = client.get("/en/")
    assert r.status_code == 200
    assert b"MODELS" in r.content or b"Find best model" in r.content
    client.cookies.set("upscale_view", "simple")
    r2 = client.get("/en/")
    assert r2.status_code == 200
    assert b"Advanced view" in r2.content


def test_locale_index_page(client: TestClient):
    r = client.get("/en/index.html")
    assert r.status_code == 200
    assert b"UPSCALE" in r.content


def test_legacy_html_redirects_to_locale(client: TestClient):
    r = client.get("/index.html", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/en/index.html"


def test_locale_cookie_and_accept_language(client: TestClient):
    r = client.get("/", follow_redirects=False, headers={"Accept-Language": "pl,en;q=0.8"})
    assert r.status_code == 307
    assert r.headers["location"] == "/pl/"
    client.cookies.set("upscale_lang", "pl")
    r2 = client.get("/", follow_redirects=False, headers={"Accept-Language": "de"})
    assert r2.status_code == 307
    assert r2.headers["location"] == "/pl/"


def test_unknown_locale_404(client: TestClient):
    assert client.get("/xx/").status_code == 404
    assert client.get("/xx/index.html").status_code == 404
    assert client.get("/api/").status_code == 404


def test_api_presets_apply(client: TestClient):
    r = client.post("/api/presets/apply", json={"id": "photo_4x", "kind": "option"})
    assert r.status_code == 200
    body = r.json()
    assert body["settings"]["scale"] == 4.0
    r2 = client.post("/api/presets/apply", json={"id": "4k", "kind": "target"})
    assert r2.status_code == 200
    assert r2.json()["settings"]["target_size"] == 3840


def test_job_status_endpoint(client: TestClient, tmp_path: Path):
    out = tmp_path / "out.png"
    out.write_bytes(TINY_PNG)

    def fake_execute(cfg, **kwargs):
        if kwargs.get("on_progress"):
            kwargs["on_progress"]({"type": "done", "pct": 1.0, "message": "done"})
        return out

    with patch("upscaler.server.execute", side_effect=fake_execute):
        r = client.post(
            "/api/jobs",
            files={"file": ("test.png", TINY_PNG, "image/png")},
            data={"settings": '{"scale": 2, "algorithm": "nearest"}'},
        )
    job_id = r.json()["id"]
    for _ in range(100):
        st = client.get(f"/api/jobs/{job_id}")
        if st.status_code == 200 and st.json().get("state") == "done":
            break
        time.sleep(0.05)
    else:
        pytest.fail("job status never done")
    assert st.json()["id"] == job_id


def test_resume_emits_resumed_type(client: TestClient, tmp_path: Path):
    out = tmp_path / "out.png"
    out.write_bytes(b"x")
    started = threading.Event()
    release = threading.Event()

    def fake_execute(cfg, **kwargs):
        started.set()
        release.wait(timeout=2)
        return out

    with patch("upscaler.server.execute", side_effect=fake_execute):
        r = client.post(
            "/api/jobs",
            files={"file": ("test.png", TINY_PNG, "image/png")},
            data={"settings": '{"scale": 2, "algorithm": "nearest"}'},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        assert started.wait(timeout=10), "worker did not start job"
        client.post(f"/api/jobs/{job_id}/pause")
        res = client.post(f"/api/jobs/{job_id}/resume")
        assert res.status_code == 200
        release.set()


def test_display_serves_from_disk_without_memory(client: TestClient, tmp_path: Path):
    """After process restart, /display must still serve job_dir/display.png."""
    import upscaler.server as server

    job_id = "diskonlydisplay01"
    job_dir = server.JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "display.png").write_bytes(TINY_PNG)
    (job_dir / "result.png").write_bytes(TINY_PNG)
    with server._jobs_lock:
        server._jobs.pop(job_id, None)

    disp = client.get(f"/api/jobs/{job_id}/display")
    assert disp.status_code == 200, disp.text
    assert disp.content.startswith(b"\x89PNG")
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").json()["state"] == "done"


def test_sse_done_display_ready(client: TestClient, tmp_path: Path):
    """First SSE ``done`` must arrive only after /display is served (no race)."""
    out = tmp_path / "out.png"
    out.write_bytes(TINY_PNG)

    def fake_execute(cfg, **kwargs):
        # Premature engine done — must not reach the client before display exists
        if kwargs.get("on_progress"):
            kwargs["on_progress"]({"type": "done", "pct": 1.0, "message": "engine"})
        return out

    with patch("upscaler.server.execute", side_effect=fake_execute):
        r = client.post(
            "/api/jobs",
            files={"file": ("test.png", TINY_PNG, "image/png")},
            data={"settings": '{"scale": 2, "algorithm": "nearest"}'},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
            for line in stream.iter_lines():
                if not line or line.startswith(":") or not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev.get("type") == "done":
                    disp = client.get(f"/api/jobs/{job_id}/display")
                    assert disp.status_code == 200, disp.text
                    assert disp.headers.get("content-type", "").startswith("image/")
                    break
            else:
                pytest.fail("never received done SSE")


def test_sse_pause_resume_cancel_sequence(client: TestClient, tmp_path: Path):
    """SSE stream sees pause/resume control events then terminal cancelled."""
    out = tmp_path / "out.png"
    out.write_bytes(TINY_PNG)
    started = threading.Event()
    release = threading.Event()

    def fake_execute(cfg, **kwargs):
        on_progress = kwargs.get("on_progress")
        started.set()
        # Stay alive until cancel; emit tile so pause can mark paused state.
        if on_progress:
            on_progress({"type": "tile", "pct": 0.2, "message": "tile"})
        deadline = time.time() + 5
        while time.time() < deadline:
            if cfg.cancel_event is not None and cfg.cancel_event.is_set():
                from upscaler.tiling import JobCancelled

                raise JobCancelled()
            if cfg.pause_event is not None and cfg.pause_event.is_set() and on_progress:
                on_progress({"type": "tile", "pct": 0.25, "message": "paused-tick"})
            if release.is_set():
                break
            time.sleep(0.05)
        if on_progress:
            on_progress({"type": "done", "pct": 1.0, "message": "done"})
        return out

    with patch("upscaler.server.execute", side_effect=fake_execute):
        r = client.post(
            "/api/jobs",
            files={"file": ("test.png", TINY_PNG, "image/png")},
            data={"settings": '{"scale": 2, "algorithm": "nearest"}'},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        assert started.wait(timeout=10)

        types: list[str] = []
        with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
            assert client.post(f"/api/jobs/{job_id}/pause").status_code == 200
            assert client.post(f"/api/jobs/{job_id}/resume").status_code == 200
            assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
            for line in stream.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    types.append(ev.get("type", ""))
                    if ev.get("type") in ("done", "error", "cancelled"):
                        break

        assert "resumed" in types
        assert "cancelled" in types
        assert types.index("resumed") < types.index("cancelled")

