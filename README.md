# Scroll Screenshot Stitcher

A high-confidence, pixel-preserving stitcher for vertically scrolling screenshots.

It is designed for mobile apps and web pages where the top and bottom UI stay fixed while the body scrolls. The tool detects overlap between adjacent screenshots, removes repeated content, chooses safe seams, optionally removes transient scrollbars, and emits a JSON audit trail showing where every output region came from.

## Why this exists

Naive screenshot stitching often fails on repetitive interfaces: transaction rows, chat messages, tables, feeds, and settings screens can look structurally similar. A visually plausible but wrong seam can silently duplicate or omit content.

This project prioritizes **fidelity and verification** over merely producing a long image.

## Key features

- Preserves source pixels; it does not regenerate text or UI
- Automatic fixed header/footer detection
- FFT-based normalized cross-correlation for efficient coarse matching
- Multi-band edge descriptors to retain horizontal structure
- Full-resolution coarse-to-fine overlap refinement
- Duplicate screenshot detection
- Ambiguity scoring with minimum confidence margins
- Content-aware seam selection in low-information rows
- Optional transient scrollbar detection and inpainting
- Refuses to resize mismatched screenshot widths
- JSON provenance report for every stitched region
- Fails closed on low-confidence joins unless explicitly overridden

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Or install the project in editable mode:

```bash
pip install -e .
```

## Usage

Provide screenshots in top-to-bottom scroll order:

```bash
python stitch_scroll.py \
  shot1.jpeg shot2.jpeg shot3.jpeg shot4.jpeg \
  -o combined.png \
  --report combined.json
```

Natural-sort filenames before processing:

```bash
python stitch_scroll.py screenshots/*.jpeg \
  --sort-name \
  -o combined.png
```

For stricter validation:

```bash
python stitch_scroll.py screenshots/*.jpeg \
  --sort-name \
  -o combined.png \
  --min-similarity 0.90 \
  --min-margin 0.015 \
  --max-seam-mae 12
```

Override automatic fixed-UI detection when necessary:

```bash
python stitch_scroll.py screenshots/*.jpeg \
  -o combined.png \
  --header 850 \
  --footer 320
```

Keep scrollbars instead of removing detected transient indicators:

```bash
python stitch_scroll.py screenshots/*.jpeg \
  -o combined.png \
  --keep-scrollbars
```

## Matching pipeline

```text
Load + normalize orientation
        ↓
Build low-resolution previews
        ↓
Detect fixed top/bottom UI
        ↓
Remove adjacent duplicate frames
        ↓
Build multi-band edge descriptors
        ↓
FFT coarse overlap search
        ↓
Full-resolution refinement
        ↓
Confidence + ambiguity validation
        ↓
Content-aware seam selection
        ↓
Stitch original pixels
        ↓
Optional scrollbar cleanup
        ↓
PNG output + JSON provenance report
```

## Safety model

The default behavior is intentionally conservative. A join is rejected when any of the following is true:

- similarity is below the configured minimum;
- the best match is not sufficiently better than another distinct candidate;
- the chosen seam has excessive pixel disagreement;
- the overlap is too small;
- source segment geometry is invalid; or
- source screenshot widths differ.

`--allow-low-confidence` exists for manual review workflows, but should not be the default for financial, legal, archival, or evidentiary screenshots.

## Audit report

The JSON report includes:

- source inputs;
- removed duplicate frames;
- detected fixed UI dimensions;
- scroll displacement and overlap for every join;
- match similarity and ambiguity margin;
- seam position and seam error;
- detected scrollbar components;
- source-to-output provenance ranges; and
- runtime matching settings.

This makes the resulting long screenshot reproducible and inspectable instead of a black-box composite.

## Privacy

Do not commit sensitive source screenshots or generated audit reports unless you intend to store that data in the repository. `.gitignore` excludes common input/output directories and generated reports by default.

## Status

Current release: **v0.1.0**

The core algorithm has been exercised on overlapping mobile billing-history screenshots containing highly repetitive transaction rows, including a difficult join with a relatively small overlap.
