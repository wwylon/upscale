"""Model registry, download, hash verification, device resolution."""

from __future__ import annotations

import hashlib
import ipaddress
import math
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
# Hosts we allow for registry downloads (built-in + common mirrors).
_DOWNLOAD_HOST_ALLOW = frozenset(
    {
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
        "huggingface.co",
        "cdn-lfs.huggingface.co",
        "objectstorage.us-phoenix-1.oraclecloud.com",
    }
)

MODELS: dict[str, dict] = {
    # --- Real-ESRGAN / RealESRNet (xinntao) ---
    "RealESRGAN_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "scale": 4,
        "license": "BSD-3-Clause-like",
        "description": "General photorealistic 4x. RRDB-23, strong detail hallucination.",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "sha256": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        "scale": 4,
        "license": "BSD-3-Clause-like",
        "description": "Anime / illustration 4x. Lighter 6-block RRDB, sharp clean lines.",
    },
    "RealESRNet_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "sha256": "a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df",
        "scale": 4,
        "license": "BSD-3-Clause-like",
        "description": "Faster/smoother variant of x4plus; less aggressive sharpening.",
    },
    "realesr-general-x4v3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "sha256": "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        "scale": 4,
        "license": "BSD-3-Clause-like",
        "description": "Newer general-purpose model (2022). Lighter than x4plus, less over-sharpening.",
    },
    "realesr-general-wdn-x4v3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        "sha256": "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
        "scale": 4,
        "license": "BSD-3-Clause-like",
        "description": "Like general-x4v3 but trained with denoising; good for noisy/compressed input.",
    },
    "realesr-animevideov3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "sha256": "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
        "scale": 4,
        "license": "BSD-3-Clause-like",
        "description": "Anime video frames 4x. Temporally consistent, fewer halos than anime_6B.",
    },
    # --- HAT (Hybrid Attention Transformer, XPixelGroup 2023) ---
    "HAT-Real_GAN_sharper": {
        "url": "https://huggingface.co/anchuang/Real_HAT_GAN_SRx4_sharper/resolve/main/Real_HAT_GAN_sharper.pth",
        "sha256": "5800b67136006eb8cab3b4ed7c8d73b6a195bb18e6cc709b674f9aa069c00271",
        "scale": 4,
        "description": "HAT-L real-world 4x (sharper variant). Transformer SR, high fidelity + detail.",
    },
    # --- SPAN (Swift Parameter-free Attention Network, 2023) ---
    "4x-spanx4-ch48": {
        "url": "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-spanx4-ch48.pth",
        "sha256": "c79e716b8eb24182c1d7fcc74fa10ae074bdb34fee7c6e67c73053ff5498c667",
        "scale": 4,
        "description": "Official SPAN pretrain 4x (ch48). Very fast, 8.6 MB. Good general-purpose baseline.",
    },
    "4xNomosUni_span_multijpg": {
        "url": "https://huggingface.co/Nudimmud/4x_NomosUni_span_multijpg/resolve/main/4xNomosUni_span_multijpg.pth",
        # Pinned after trusted download; fail-closed on mismatch.
        "sha256": "PLACEHOLDER_PIN_ON_FIRST_TRUSTED_DOWNLOAD",
        "scale": 4,
        "description": "SPAN 4x fine-tuned on NomosUni with multi-JPEG degradation. Good for compressed photos.",
    },
    # --- Community models ---
    "4x-UltraSharp": {
        "url": "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
        "sha256": "a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01",
        "scale": 4,
        "license": "check upstream / possibly non-commercial",
        "description": "Popular community photo model (2022). Strong texture detail, can over-sharpen.",
    },
    "4x-UltraSharpV2": {
        "url": "https://huggingface.co/Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2.pth",
        "sha256": "0335cf48ad654ef79fc14aa97146f0c143c499b0602b2636e2be112f4cd403cf",
        "scale": 4,
        "license": "check upstream / possibly non-commercial",
        "description": "Updated UltraSharp (2023). Less ringing, better compression handling.",
    },
    "4x-ClearRealityV1": {
        "url": "https://huggingface.co/Kim2091/ClearRealityV1/resolve/main/4x-ClearRealityV1.pth",
        "sha256": "a4cd3a25b00e0be949d4302fc774eb4d7f2ed5f47cdb51551e2d75fa6562e51e",
        "scale": 4,
        "license": "check upstream / possibly non-commercial",
        "description": "Kim2091 2023 photo model. Cleaner than UltraSharp, less over-sharpening.",
    },
}

FAST_MODEL = "4x-spanx4-ch48"
DEFAULT_WEIGHTS_DIR = Path(__file__).parent.parent / "models"

_DOWNLOAD_TIMEOUT = 60
_DOWNLOAD_RETRIES = 3
_DOWNLOAD_CHUNK = 1 << 20


def registry_scale(name: str, default: int = 4) -> int:
    meta = all_models().get(name) or MODELS.get(name)
    if meta is None:
        return default
    return int(meta.get("scale", default))


def resolve_device(requested: str):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU.", stacklevel=2)
            return torch.device("cpu")
        return torch.device("cuda")
    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("MPS not available, falling back to CPU.", stacklevel=2)
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_user_registry(weights_dir: Path = DEFAULT_WEIGHTS_DIR) -> dict[str, dict]:
    """Load external models from models/registry.json."""
    path = weights_dir / "registry.json"
    if not path.is_file():
        return {}
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_user_registry(registry: dict[str, dict], weights_dir: Path = DEFAULT_WEIGHTS_DIR) -> Path:
    import json

    weights_dir.mkdir(parents=True, exist_ok=True)
    path = weights_dir / "registry.json"
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return path


def remove_external_model(
    name: str, weights_dir: Path = DEFAULT_WEIGHTS_DIR
) -> bool:
    """Drop an external registry entry. Returns True if something was removed."""
    reg = load_user_registry(weights_dir)
    if name not in reg:
        return False
    del reg[name]
    save_user_registry(reg, weights_dir)
    return True


def prune_stale_external_models(
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
) -> list[str]:
    """Remove path-only external entries whose files are gone (e.g. old pytest temps)."""
    reg = load_user_registry(weights_dir)
    removed: list[str] = []
    changed = False
    for name, meta in list(reg.items()):
        path = meta.get("path") or ""
        url = (meta.get("url") or "").strip()
        if path and not Path(path).is_file() and not url:
            del reg[name]
            removed.append(name)
            changed = True
    if changed:
        save_user_registry(reg, weights_dir)
    return removed


def all_models(weights_dir: Path = DEFAULT_WEIGHTS_DIR) -> dict[str, dict]:
    merged = dict(MODELS)
    for name, meta in load_user_registry(weights_dir).items():
        merged[name] = {**meta, "external": True}
    return merged


def validate_model_name(name: str) -> str:
    """Reject path separators / traversal in model registry names."""
    if not name or not _MODEL_NAME_RE.fullmatch(name):
        raise ValueError(
            "Model name must be alphanumeric with ._- only "
            f"(got {name!r})"
        )
    return name


def validate_download_url(url: str) -> str:
    """Allow only https URLs to known hosts; block SSRF to private networks."""
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Download URL is required")
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme.lower() != "https":
        raise ValueError("Only https:// model download URLs are allowed")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Download URL missing host")
    # Block literal IPs that are private / link-local / loopback
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise ValueError("Download URL must not target private/reserved IPs")
    except ValueError as exc:
        if "private/reserved" in str(exc):
            raise
        # host is a hostname, not an IP — resolve and check
        pass
    else:
        # literal public IP — still require allowlist host (IPs rarely in allowlist)
        if host not in _DOWNLOAD_HOST_ALLOW:
            raise ValueError(
                f"Download host not allowlisted: {host}. "
                f"Allowed: {', '.join(sorted(_DOWNLOAD_HOST_ALLOW))}"
            )
        return raw

    if host not in _DOWNLOAD_HOST_ALLOW and not any(
        host.endswith("." + h) for h in _DOWNLOAD_HOST_ALLOW
    ):
        raise ValueError(
            f"Download host not allowlisted: {host}. "
            f"Allowed: {', '.join(sorted(_DOWNLOAD_HOST_ALLOW))}"
        )
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve download host: {host}") from exc
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise ValueError(
                f"Download host {host} resolves to blocked address {addr}"
            )
    return raw


def add_external_model(
    name: str,
    *,
    url: str | None = None,
    path: str | None = None,
    scale: int = 4,
    sha256: str | None = None,
    description: str = "",
    weights_dir: Path | None = None,
    allow_overwrite_builtin: bool = False,
) -> dict:
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR
    name = validate_model_name(name)
    if name in MODELS and not allow_overwrite_builtin:
        raise ValueError(
            f"Refusing to shadow built-in model {name!r}. "
            "Choose a different name."
        )
    if url:
        url = validate_download_url(url)
    if path:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"External model path not found: {p}")
        wroot = Path(weights_dir).expanduser().resolve()
        if not p.is_relative_to(wroot):
            raise ValueError(
                f"External model path must stay under weights dir ({wroot})"
            )
        path = str(p)
    reg = load_user_registry(weights_dir)
    entry = {
        "url": url or "",
        "path": path,
        "scale": scale,
        "sha256": sha256,
        "description": description or f"External model {name}",
    }
    reg[name] = entry
    save_user_registry(reg, weights_dir)
    return entry


def model_status(name: str, weights_dir: Path = DEFAULT_WEIGHTS_DIR) -> str:
    """Return installed | not_downloaded | failed for a registry name."""
    catalog = all_models(weights_dir)
    if name not in catalog and not Path(name).exists():
        return "failed"
    if name not in catalog:
        return "installed" if Path(name).is_file() else "failed"
    meta = catalog[name]
    if meta.get("path"):
        if Path(meta["path"]).is_file():
            return "installed"
        # Path gone: downloadable if URL remains, otherwise a dead registry row.
        if (meta.get("url") or "").strip():
            return "not_downloaded"
        return "failed"
    # Slim/ONNX desktop: treat .onnx as installed for the same registry name
    onnx_dest = weights_dir / f"{name}.onnx"
    if onnx_dest.is_file():
        return "installed"
    url_path = (meta.get("url") or "").split("?")[0]
    ext = Path(url_path).suffix or ".pth"
    dest = weights_dir / f"{name}{ext}"
    if not dest.exists():
        return "not_downloaded"
    expected = meta.get("sha256")
    if expected and not str(expected).startswith("PLACEHOLDER_"):
        stored = _load_stored_hash(dest)
        if stored and stored != expected:
            return "failed"
    return "installed"


def _model_param_bytes(device) -> int:
    """Rough estimate of already-allocated CUDA memory (model + cache)."""
    import torch

    if getattr(device, "type", None) != "cuda":
        return 0
    try:
        return int(torch.cuda.memory_allocated(device))
    except Exception:
        return 0


def auto_tile_size(
    device,
    fp16: bool,
    model_bytes: int | None = None,
) -> int:
    import torch

    if getattr(device, "type", None) == "cpu":
        return 256
    bytes_per_pixel = 7000 if fp16 else 14000
    max_cap = 768 if fp16 else 512
    try:
        # mem_get_info free is already after current allocations — do not
        # subtract memory_allocated again (that under-sizes tiles).
        free_b, _total_b = torch.cuda.mem_get_info(device)
        headroom = 0.70
        if model_bytes is not None:
            # Optional hint for callers that know the next model size.
            free_b = max(0, free_b - int(model_bytes * 0.15))
        available = int(free_b * headroom)
    except Exception:
        available = int(4 * 1024**3 * 0.50)
    available = max(available, 64 * 1024 * 1024)
    tile = int(math.sqrt(available / bytes_per_pixel))
    tile = (tile // 64) * 64
    return max(128, min(tile, max_cap))


def ensure_model(name: str, weights_dir: Path = DEFAULT_WEIGHTS_DIR) -> Path:
    """Download model if missing or corrupt; return path to the weights file."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    catalog = all_models(weights_dir)

    if name not in catalog:
        p = Path(name)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {name}")
        return p

    meta = catalog[name]
    if meta.get("path"):
        p = Path(meta["path"])
        if p.is_file():
            return p
        url = (meta.get("url") or "").strip()
        if not url:
            remove_external_model(name, weights_dir)
            raise FileNotFoundError(
                f"External model {name!r} path is missing ({p}) and has no "
                "download URL; removed from the local registry."
            )
        # Fall through to URL download into weights_dir.
    else:
        url = (meta.get("url") or "").strip()
        if not url:
            raise FileNotFoundError(f"Model {name!r} has no path or download URL")
    url_path = (meta.get("url") or "").split("?")[0]
    if not url_path:
        raise FileNotFoundError(f"Model {name!r} has no download URL")
    ext = Path(url_path).suffix or ".pth"
    dest = weights_dir / f"{name}{ext}"
    expected = meta.get("sha256")
    placeholder = expected is not None and str(expected).startswith("PLACEHOLDER_")

    if dest.exists():
        stored = _load_stored_hash(dest)
        if placeholder:
            actual = stored or _sha256(dest)
            _save_stored_hash(dest, actual)
            meta["sha256"] = actual
            return dest
        if stored == expected or _sha256(dest) == expected:
            if stored != expected:
                _save_stored_hash(dest, expected)
            return dest
        print(f"Hash mismatch for {name}, re-downloading...")
        dest.unlink()

    if placeholder:
        raise RuntimeError(
            f"Model {name} has no pinned sha256 yet. Pass "
            "UPSCALE_TRUST_MODEL_DOWNLOAD=1 once to pin the first download, "
            "or set a real hash in the registry."
        )

    print(f"Downloading {name} ...")
    _download_with_retries(meta["url"], dest)

    actual = _sha256(dest)
    if expected is not None and actual != expected:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Hash mismatch for {name}: expected {expected}, got {actual}"
        )
    _save_stored_hash(dest, actual)
    return dest


def ensure_model_trust_placeholder(
    name: str, weights_dir: Path = DEFAULT_WEIGHTS_DIR
) -> Path:
    """First-trusted download for PLACEHOLDER_* hashes (explicit opt-in)."""
    import os

    if os.environ.get("UPSCALE_TRUST_MODEL_DOWNLOAD", "").strip() not in (
        "1",
        "true",
        "yes",
    ):
        return ensure_model(name, weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    catalog = all_models(weights_dir)
    if name not in catalog:
        return ensure_model(name, weights_dir)
    meta = catalog[name]
    expected = meta.get("sha256")
    if not (isinstance(expected, str) and expected.startswith("PLACEHOLDER_")):
        return ensure_model(name, weights_dir)
    url_path = (meta.get("url") or "").split("?")[0]
    ext = Path(url_path).suffix or ".pth"
    dest = weights_dir / f"{name}{ext}"
    if dest.exists():
        actual = _sha256(dest)
        _save_stored_hash(dest, actual)
        meta["sha256"] = actual
        return dest
    print(f"Downloading {name} (trusted first pin) ...")
    _download_with_retries(meta["url"], dest)
    actual = _sha256(dest)
    _save_stored_hash(dest, actual)
    meta["sha256"] = actual
    print(f"  Pinned sha256 for {name}: {actual}")
    return dest


def load_model(model_path: Path, device, fp16: bool):
    """Load a model via spandrel; returns an ImageModelDescriptor."""
    try:
        import spandrel
    except ImportError as exc:
        raise ImportError("Install spandrel: pip install spandrel") from exc

    descriptor = spandrel.ModelLoader(device=device).load_from_file(str(model_path))
    descriptor.model.eval()
    if fp16 and getattr(device, "type", None) == "cuda":
        descriptor.model.half()
    return descriptor


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_file(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _load_stored_hash(path: Path) -> str | None:
    hf = _hash_file(path)
    return hf.read_text().strip() if hf.exists() else None


def _save_stored_hash(path: Path, digest: str) -> None:
    _hash_file(path).write_text(digest)


def _download_with_retries(url: str, dest: Path) -> None:
    url = validate_download_url(url)
    partial = dest.with_suffix(dest.suffix + ".partial")
    last_err: Exception | None = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            _stream_download(url, partial)
            partial.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_err = exc
            warnings.warn(
                f"Download attempt {attempt}/{_DOWNLOAD_RETRIES} failed: {exc}",
                stacklevel=2,
            )
            time.sleep(min(2 ** attempt, 8))
    if partial.exists():
        partial.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {url}") from last_err


def _stream_download(url: str, dest: Path, *, max_bytes: int = 2_000_000_000) -> None:
    """Download without auto-following redirects; re-validate every Location."""
    current = validate_download_url(url)
    for _hop in range(8):
        req = urllib.request.Request(
            current,
            headers={"User-Agent": "upscaler/1.0"},
            method="GET",
        )
        # opener that does not follow redirects
        opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler)
        # Replace redirect handler with one that raises
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
                return None

        opener = urllib.request.build_opener(_NoRedirect)
        try:
            resp = opener.open(req, timeout=_DOWNLOAD_TIMEOUT)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location"):
                nxt = urllib.parse.urljoin(current, exc.headers["Location"])
                current = validate_download_url(nxt)
                continue
            raise
        with resp:
            code = getattr(resp, "status", None) or resp.getcode()
            if code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("Location")
                if not loc:
                    raise ValueError("Redirect without Location")
                current = validate_download_url(urllib.parse.urljoin(current, loc))
                continue
            total = resp.headers.get("Content-Length")
            total_size = int(total) if total and total.isdigit() else 0
            if total_size > max_bytes:
                raise ValueError(f"Download too large ({total_size} bytes)")
            downloaded = 0
            last_pct = -1
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(f"Download exceeded {max_bytes} bytes")
                    out.write(chunk)
                    if total_size > 0:
                        pct = min(100, downloaded * 100 // total_size)
                        if pct != last_pct:
                            print(f"\r  {pct}%", end="", flush=True)
                            last_pct = pct
            if total_size > 0:
                print()
            return
    raise ValueError(f"Too many redirects downloading {url}")
