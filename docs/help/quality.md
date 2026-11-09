# Quality checks

`GET /api/jobs/{id}/quality?mode=seam` returns a vertical-join score for the finished result. `mode=diff` compares that result with a Lanczos resize of the input and writes a small `quality_diff.png` in the job directory. The scores come from `upscaler.quality` (`strip_seam_score`, `diff_metrics`). There is no seam-view in the web UI yet.
