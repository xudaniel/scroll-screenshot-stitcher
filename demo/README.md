# Synthetic Demo

This folder contains a completely fictional mobile-app dataset for demonstrating and regression-testing `scroll-screenshot-stitcher`.

No screenshots or records in this folder contain personal, financial, legal, or user-derived data.

## Files

- `screenshot_01.png` through `screenshot_04.png` — overlapping synthetic mobile screenshots.
- `expected_stitched.png` — the complete reference document with one header, one footer, and all 18 fictional trip records exactly once.
- `contact_sheet.png` — compact preview of the four inputs.
- `manifest.json` — known viewport, scroll positions, and expected overlap values.
- `generate_demo.py` — deterministic generator for rebuilding all demo assets.

## Try the stitcher

From the repository root:

```bash
python stitch_scroll.py \
  demo/screenshot_01.png \
  demo/screenshot_02.png \
  demo/screenshot_03.png \
  demo/screenshot_04.png \
  -o demo/actual_stitched.png \
  --report demo/actual_report.json
```

The generated `actual_stitched.png` can be visually compared with `expected_stitched.png`. The inputs deliberately share large exact overlap regions while retaining fixed top and bottom navigation UI.

## Rebuild the demo

```bash
python demo/generate_demo.py
```

The generator uses Pillow and writes the screenshots, reference output, contact sheet, and manifest directly into this folder.
