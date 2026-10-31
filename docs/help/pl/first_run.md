# Pierwsze uruchomienie

1. Instalacja: `pip install -e ".[server]"` (dodaj `[ui]` lub `[desktop]` w razie potrzeby).
2. `upscale models list` — pierwsze powiększenie pobiera wagi do `models/`.
3. `upscale suggest samples/tiny.png` — podpowiedź modelu.
4. `upscale upscale input.png out.png --scale 4 --cpu` bez CUDA.
5. UI: `upscale serve` → http://127.0.0.1:8787/en/ (domyślnie **Widok prosty**; ostatni wybór prosty/zaawansowany jest zapamiętywany).

`upscale presets list` — cele FHD/4K.
