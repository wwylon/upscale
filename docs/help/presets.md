# Presets

List presets with `upscale presets list` or `GET /api/presets`.

Apply via `POST /api/presets/apply` with `{ "id": "photo_4x", "kind": "option" }` or `{ "id": "4k", "kind": "target" }`. The response includes a `settings` fragment to merge into `JobSettings`.

Option presets: photo 4×, art/anime 4×, fast SPAN 4×. Target presets: Full HD, 4K, and large-print long edges, plus exact frames — A4 at 300 DPI (2480×3508), A3 at 300 DPI (3508×4961), social square 1080×1080, and link preview 1200×630.
