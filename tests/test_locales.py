"""UI catalogue integrity: every key the web UI uses exists in locales/en.json.

Hermetic: reads ui/ sources and locales/*.json only (no server, no browser).
Call shapes recognised (the same ones llm-localize's scanner reads):
``i18n.t('key')`` in JS and ``data-i18n*="key"`` attributes in HTML templates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
LOCALES = ROOT / "locales"

# support.js is the generated DC runtime, i18n.js the lookup runtime (its doc
# comment shows example keys); vendor/_ds are third-party bundles.
UI_SOURCES = sorted(
    [p for p in UI.glob("*.html")]
    + [p for p in UI.glob("*.js") if p.name not in ("support.js", "i18n.js")]
    + list((UI / "bridge").glob("*.js"))
)

T_CALL = re.compile(r"""\bi18n\.t\(\s*(['"])([^'"]+)\1""")
L_CALL = re.compile(r"""\bthis\.L\(\s*(['"])([^'"]+)\1""")
ATTR = re.compile(r"""\bdata-i18n(?:-[a-z-]+)?="([^"]+)\"""")
# Keyed element whose English text sits inline: data-i18n="k">Text</
INLINE = re.compile(r"""data-i18n="([^"]+)"[^>]*>([^<]*)</""")
ATTR_VALUE = re.compile(r"""\b(title|placeholder|aria-label|alt)="([^"]*)"[^>]*?\bdata-i18n-\1="([^"]+)\"""")
PLACEHOLDER = re.compile(r"\{\s*([A-Za-z_]\w*)\s*[,}]")


def _load(code: str) -> dict:
    return json.loads((LOCALES / f"{code}.json").read_text(encoding="utf-8"))


def _used_keys() -> dict[str, set[str]]:
    used: dict[str, set[str]] = {}
    for path in UI_SOURCES:
        text = path.read_text(encoding="utf-8")
        keys = {m.group(2) for m in T_CALL.finditer(text)}
        keys |= {m.group(2) for m in L_CALL.finditer(text)}
        keys |= {m.group(1) for m in ATTR.finditer(text)}
        if keys:
            used[path.relative_to(ROOT).as_posix()] = keys
    return used


def test_catalogues_are_flat_string_maps():
    for code in ("en", "pl"):
        data = _load(code)
        assert isinstance(data, dict) and data, code
        for k, v in data.items():
            assert isinstance(k, str) and re.fullmatch(r"[a-z0-9_.]+", k), (code, k)
            assert isinstance(v, str), (code, k)


def test_every_used_key_exists_in_english():
    en = _load("en")
    used = _used_keys()
    assert used, "no i18n usage found under ui/"
    missing = {f: sorted(k - en.keys()) for f, k in used.items() if k - en.keys()}
    assert not missing, missing


def test_polish_keys_are_subset_of_english():
    en, pl = _load("en"), _load("pl")
    assert not sorted(pl.keys() - en.keys())


def test_polish_keeps_english_placeholders():
    en, pl = _load("en"), _load("pl")
    for k, v in pl.items():
        want = set(PLACEHOLDER.findall(en[k]))
        got = set(PLACEHOLDER.findall(v))
        assert got <= want, (k, got - want)


@pytest.mark.parametrize("path", [p for p in UI_SOURCES if p.suffix == ".html"], ids=lambda p: p.name)
def test_inline_english_matches_catalogue(path: Path):
    """English markup must equal the catalogue so English rendering is unchanged."""
    en = _load("en")
    text = path.read_text(encoding="utf-8")
    for m in INLINE.finditer(text):
        assert en.get(m.group(1)) == m.group(2), (m.group(1), m.group(2))
    for m in ATTR_VALUE.finditer(text):
        assert en.get(m.group(3)) == m.group(2), (m.group(3), m.group(2))


def test_pages_load_runtime_before_dc():
    for name in ("index", "simple", "library", "cli"):
        text = (UI / f"{name}.html").read_text(encoding="utf-8")
        assert text.index("/ui/i18n.js") < text.index("/ui/support.js"), name


def test_no_runtime_text_walker_left():
    text = (UI / "index.html").read_text(encoding="utf-8")
    for marker in ("DICT=", "DICT_X=", "PATS=", "PATS_X=", "createTreeWalker", "loadLocalePack"):
        assert marker not in text, marker
