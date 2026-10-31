# Contributing

## Setup

```text
git clone https://github.com/wwylon/upscale.git
cd upscale
pip install -e ".[dev,server,mcp]"
upscale --help
```

## Tests

```text
ruff check upscaler tests
mypy --follow-imports=silent upscaler/schema.py upscaler/job_prep.py
pytest -q
python scripts/cpu_smoke.py
```

## Pull requests

- Open PRs against the public repository.
- Link an issue when fixing a bug; include OS, Python version, and `upscale --help` / GPU notes for runtime problems.
- Maintainers may replay accepted changes through an internal mirror; your commits can appear rewritten in the next public export. Attribution is preserved in CHANGELOG / PR comments when practical.

## Security

See [SECURITY.md](SECURITY.md). Do not file vulnerabilities in public issues.
