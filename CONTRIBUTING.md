# Contributing

Thank you for contributing to Scroll Screenshot Stitcher.

## Principles

- Preserve source pixels and provenance.
- Fail closed when a join is ambiguous.
- Never commit sensitive screenshots or generated reports containing private data.
- Add deterministic synthetic fixtures for matching and seam changes.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . -r requirements-dev.txt
python -m pytest -q
```

Pull requests should describe the failure mode, include a regression test, and report any change to confidence thresholds or output provenance.
