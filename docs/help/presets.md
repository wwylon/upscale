# Presets

List presets with `upscale presets list` or `GET /api/presets`.

Apply via `POST /api/presets/apply` with `{ "id": "photo_4x", "kind": "option" }` or `{ "id": "4k", "kind": "target" }`. The response includes a `settings` fragment to merge into `JobSettings`.

Option presets: photo 4×, art/anime 4×, fast SPAN 4×. Target presets: Full HD, 4K, large print long edges.
