# Resume (checkpoint)

Long tiled jobs can save **strip flush-row checkpoints** under `<job_dir>/checkpoints/` when a job directory is set (server jobs always have one).

- After each Y-band flush, `strip.json` (+ optional `strip_carry.npz`) records `flush_row` and carry buffers.
- Re-running the same pass with `resume=True` (automatic when `job_dir` is passed into tiling) skips already-flushed bands.
- Successful completion clears the strip checkpoint.

**Pause** / **Resume** during a live run is cooperative **between tiles** in the current process. The web UI also offers **Resume interrupted** when a checkpointed job is detected after a restart or crash.

Huge outputs may allocate a uint8 memmap under the job dir when the frame exceeds ~512 MiB so peak RAM stays dominated by the strip buffers, not a second full float canvas.
