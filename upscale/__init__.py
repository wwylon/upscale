"""AI image upscaler — multi-pass Real-ESRGAN with feather-blended tiles."""

from .config import UpscaleConfig
from .engine import execute
from .io_utils import lanczos_resize, load_rgba, save_image
from .models import DEFAULT_WEIGHTS_DIR, MODELS, ensure_model, load_model, resolve_device
from .pipeline import compute_passes, resolve_model_scale, run, upscale_alpha
from .schema import JobSettings, NamePattern, ProgressEvent

__all__ = [
    "UpscaleConfig",
    "JobSettings",
    "NamePattern",
    "ProgressEvent",
    "MODELS",
    "DEFAULT_WEIGHTS_DIR",
    "compute_passes",
    "run",
    "resolve_model_scale",
    "upscale_alpha",
    "execute",
    "load_rgba",
    "save_image",
    "lanczos_resize",
    "ensure_model",
    "load_model",
    "resolve_device",
]

__version__ = "0.5.0"
