#!/usr/bin/env python3
"""Lossless scrolling-screenshot stitcher.

Designed for mobile screenshots with a fixed top UI, a fixed bottom UI,
and a vertically scrolling body. It preserves original pixels, removes
repeated overlap, optionally removes transient scrollbars, and writes a
JSON audit report with source provenance.

Dependencies:
    pip install numpy pillow opencv-python

Example:
    python stitch_scroll.py shot1.jpeg shot2.jpeg shot3.jpeg \
        -o combined.png --report combined.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class Join:
    left: str
    right: str
    scroll_px: int
    overlap_px: int
    seam_left_px: int
    seam_right_px: int
    similarity: float
    distinct_margin: float
    seam_row_mae: float
    coarse_candidates: list[dict]


def natural_key(path: Path) -> list[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def load_rgb(path: Path) -> Image.Image:
    """Load, apply EXIF orientation, and fully materialize RGB pixels."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def make_preview(path: Path, target_width: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    image = load_rgb(path)
    width, height = image.size
    scale = min(1.0, target_width / width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    gray = np.asarray(image.convert("L").resize(size, Image.Resampling.BILINEAR))
    return gray, scale, (width, height)


def detect_static_bands(
    previews: Sequence[np.ndarray],
    *,
    pixel_delta: int = 10,
    row_activity_threshold: float = 0.02,
    cluster_window: int = 12,
    minimum_hits: int = 4,
    safety_rows: int = 2,
) -> tuple[int, int, dict]:
    """Detect fixed top and bottom regions in preview-pixel coordinates.

    A row is considered active when a meaningful percentage of its pixels
    changes between consecutive screenshots. We use the median across pairs,
    which suppresses one-off changes such as the clock or battery indicator.
    """
    if len(previews) < 2:
        return 0, 0, {"warning": "Only one screenshot; static bands not detected."}

    height = min(x.shape[0] for x in previews)
    width = min(x.shape[1] for x in previews)

    pair_activity = []
    for first, second in zip(previews[:-1], previews[1:]):
        difference = np.abs(
            first[:height, :width].astype(np.int16)
            - second[:height, :width].astype(np.int16)
        )
        pair_activity.append((difference > pixel_delta).mean(axis=1))

    activity = np.median(np.stack(pair_activity), axis=0)
    active = activity > row_activity_threshold

    if height < cluster_window:
        return 0, 0, {"warning": "Screenshots are too short for auto-detection."}

    counts = np.convolve(
        active.astype(np.int16),
        np.ones(cluster_window, dtype=np.int16),
        mode="valid",
    )
    clusters = np.flatnonzero(counts >= minimum_hits)

    if clusters.size == 0:
        return 0, 0, {"warning": "No moving body detected; use --header/--footer."}

    first_window = int(clusters[0])
    first_active = first_window + int(
        np.flatnonzero(active[first_window : first_window + cluster_window])[0]
    )

    last_window = int(clusters[-1])
    last_active_rows = np.flatnonzero(active[last_window : last_window + cluster_window])
    last_active = last_window + int(last_active_rows[-1])

    top = max(0, first_active - safety_rows)
    bottom_start = min(height, last_active + safety_rows + 1)
    footer = height - bottom_start

    # Reject implausible auto-detection rather than silently deleting content.
    if top + footer > round(height * 0.75):
        return 0, 0, {
            "warning": "Auto-detected fixed UI consumed too much of the screen; rejected."
        }

    return top, footer, {
        "preview_top_px": top,
        "preview_footer_px": footer,
        "row_activity_threshold": row_activity_threshold,
        "pixel_delta": pixel_delta,
    }


def edge_map(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    x_gradient = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    y_gradient = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return np.log1p(cv2.magnitude(x_gradient, y_gradient)).astype(np.float32)


def row_descriptor(gray_body: np.ndarray, bands: int = 6) -> np.ndarray:
    """Compact, translation-friendly per-row descriptor.

    The body is split into horizontal bands. Edge mean and RMS are recorded
    for each row in each band, retaining enough horizontal structure to
    distinguish repeated transaction rows while keeping FFT matching cheap.
    """
    height, width = gray_body.shape
    x0, x1 = round(width * 0.05), round(width * 0.95)
    features_image = edge_map(gray_body[:, x0:x1])

    boundaries = np.linspace(0, features_image.shape[1], bands + 1, dtype=int)
    channels = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        block = features_image[:, left:right]
        channels.append(block.mean(axis=1))
        channels.append(np.sqrt((block * block).mean(axis=1) + 1e-6))

    descriptor = np.stack(channels, axis=1).astype(np.float32)

    # High-pass vertically so large white regions and slow background changes
    # do not dominate the correlation.
    descriptor -= cv2.GaussianBlur(descriptor, (1, 0), sigmaX=0, sigmaY=6)

    median = np.median(descriptor, axis=0, keepdims=True)
    mad = np.median(np.abs(descriptor - median), axis=0, keepdims=True)
    descriptor = (descriptor - median) / (1.4826 * mad + 1e-3)
    return np.clip(descriptor, -6, 6).astype(np.float32)


def fft_normalized_cross_correlation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    min_overlap: int,
    max_overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every plausible downward scroll offset in O(F * H log H)."""
    first_height, channels = first.shape
    second_height = second.shape[0]

    min_overlap = max(2, min(min_overlap, first_height, second_height))
    max_overlap = max(min_overlap, min(max_overlap, first_height, second_height))

    minimum_lag = max(0, first_height - max_overlap)
    maximum_lag = first_height - min_overlap
    if maximum_lag < minimum_lag:
        raise ValueError("Invalid overlap range.")

    fft_size = 1 << ((first_height + second_height - 1).bit_length())
    numerator = np.zeros(first_height + second_height - 1, dtype=np.float64)

    for channel in range(channels):
        convolution = np.fft.irfft(
            np.fft.rfft(first[:, channel], fft_size)
            * np.fft.rfft(second[::-1, channel], fft_size),
            fft_size,
        )
        numerator += convolution[: first_height + second_height - 1]

    first_energy = (first * first).sum(axis=1, dtype=np.float64)
    second_energy = (second * second).sum(axis=1, dtype=np.float64)
    first_prefix = np.concatenate(([0.0], np.cumsum(first_energy)))
    second_prefix = np.concatenate(([0.0], np.cumsum(second_energy)))

    lags = np.arange(minimum_lag, maximum_lag + 1)
    overlap = np.minimum(first_height - lags, second_height)

    num = numerator[lags + second_height - 1]
    energy_a = first_prefix[lags + overlap] - first_prefix[lags]
    energy_b = second_prefix[overlap]
    scores = num / (np.sqrt(energy_a * energy_b) + 1e-12)
    return lags, scores


def select_distinct_candidates(
    lags: np.ndarray,
    scores: np.ndarray,
    *,
    count: int,
    separation: int,
) -> list[tuple[int, float]]:
    selected: list[tuple[int, float]] = []
    for index in np.argsort(scores)[::-1]:
        lag = int(lags[index])
        score = float(scores[index])
        if all(abs(lag - existing_lag) > separation for existing_lag, _ in selected):
            selected.append((lag, score))
            if len(selected) >= count:
                break
    return selected


@lru_cache(maxsize=3)
def load_gray_body(path_text: str, header: int, footer: int) -> np.ndarray:
    image = load_rgb(Path(path_text))
    gray = np.asarray(image.convert("L"))
    end = gray.shape[0] - footer if footer else gray.shape[0]
    return gray[header:end]


@lru_cache(maxsize=3)
def refinement_map(path_text: str, header: int, footer: int, target_width: int) -> np.ndarray:
    body = load_gray_body(path_text, header, footer)
    height, width = body.shape
    x0, x1 = round(width * 0.05), round(width * 0.95)
    body = body[:, x0:x1]
    if body.shape[1] != target_width:
        body = cv2.resize(body, (target_width, height), interpolation=cv2.INTER_AREA)
    return edge_map(body)


def offset_similarity(
    first: np.ndarray,
    second: np.ndarray,
    scroll: int,
    min_overlap: int,
) -> float:
    if scroll < 0 or scroll >= first.shape[0]:
        return -1.0
    overlap = min(first.shape[0] - scroll, second.shape[0])
    if overlap < min_overlap:
        return -1.0
    first_region = first[scroll : scroll + overlap]
    second_region = second[:overlap]
    return float(
        cv2.matchTemplate(first_region, second_region, cv2.TM_CCOEFF_NORMED)[0, 0]
    )


def choose_content_aware_seam(
    first_gray: np.ndarray,
    second_gray: np.ndarray,
    scroll: int,
    overlap: int,
    *,
    target_width: int = 320,
) -> tuple[int, float]:
    """Choose a seam inside low-information, low-difference rows."""
    first_width = first_gray.shape[1]
    second_width = second_gray.shape[1]

    first_region = first_gray[
        scroll : scroll + overlap,
        round(first_width * 0.05) : round(first_width * 0.95),
    ]
    second_region = second_gray[
        :overlap,
        round(second_width * 0.05) : round(second_width * 0.95),
    ]

    if first_region.shape[1] != target_width:
        first_region = cv2.resize(
            first_region, (target_width, overlap), interpolation=cv2.INTER_AREA
        )
    if second_region.shape[1] != target_width:
        second_region = cv2.resize(
            second_region, (target_width, overlap), interpolation=cv2.INTER_AREA
        )

    row_difference = np.abs(
        first_region.astype(np.float32) - second_region.astype(np.float32)
    ).mean(axis=1)
    row_information = (edge_map(first_region) + edge_map(second_region)).mean(axis=1)

    def robust_unit_interval(values: np.ndarray) -> np.ndarray:
        low, high = np.percentile(values, [10, 90])
        return np.clip((values - low) / (high - low + 1e-6), 0, 1)

    cost = robust_unit_interval(row_difference) + 0.4 * robust_unit_interval(row_information)

    smoothing = min(9, overlap if overlap % 2 else max(1, overlap - 1))
    if smoothing >= 3:
        cost = np.convolve(cost, np.ones(smoothing) / smoothing, mode="same")

    low = max(0, round(overlap * 0.15))
    high = min(overlap, round(overlap * 0.85))
    seam = overlap // 2 if high <= low else low + int(np.argmin(cost[low:high]))
    return seam, float(row_difference[seam])


def match_pair(
    left: Path,
    right: Path,
    left_descriptor: np.ndarray,
    right_descriptor: np.ndarray,
    *,
    preview_scale: float,
    header: int,
    footer: int,
    min_overlap_px: int,
    max_overlap_ratio: float,
    top_candidates: int,
    refinement_radius: int,
    refinement_width: int,
) -> Join:
    min_overlap_preview = max(4, round(min_overlap_px * preview_scale))
    max_overlap_preview = max(
        min_overlap_preview,
        round(min(left_descriptor.shape[0], right_descriptor.shape[0]) * max_overlap_ratio),
    )

    lags, scores = fft_normalized_cross_correlation(
        left_descriptor,
        right_descriptor,
        min_overlap=min_overlap_preview,
        max_overlap=max_overlap_preview,
    )
    coarse = select_distinct_candidates(
        lags,
        scores,
        count=top_candidates,
        separation=max(2, round(8 * preview_scale)),
    )

    left_map = refinement_map(str(left), header, footer, refinement_width)
    right_map = refinement_map(str(right), header, footer, refinement_width)

    tested: list[tuple[float, int]] = []
    for preview_lag, _ in coarse:
        base = round(preview_lag / preview_scale)
        start = max(0, base - refinement_radius)
        stop = min(left_map.shape[0] - min_overlap_px, base + refinement_radius)
        for scroll in range(start, stop + 1):
            tested.append(
                (
                    offset_similarity(left_map, right_map, scroll, min_overlap_px),
                    scroll,
                )
            )

    if not tested:
        raise RuntimeError(f"No valid overlap found: {left.name} -> {right.name}")

    tested.sort(reverse=True)
    similarity, scroll = tested[0]

    # Ignore neighboring one-pixel variants when measuring ambiguity.
    distinct_separation = max(12, refinement_radius * 2)
    second_best = next(
        (score for score, candidate in tested[1:] if abs(candidate - scroll) > distinct_separation),
        0.0,
    )
    margin = similarity - second_best

    left_gray = load_gray_body(str(left), header, footer)
    right_gray = load_gray_body(str(right), header, footer)
    overlap = min(left_gray.shape[0] - scroll, right_gray.shape[0])

    seam_right, seam_mae = choose_content_aware_seam(
        left_gray, right_gray, scroll, overlap, target_width=refinement_width
    )
    seam_left = scroll + seam_right

    return Join(
        left=str(left),
        right=str(right),
        scroll_px=int(scroll),
        overlap_px=int(overlap),
        seam_left_px=int(seam_left),
        seam_right_px=int(seam_right),
        similarity=float(similarity),
        distinct_margin=float(margin),
        seam_row_mae=float(seam_mae),
        coarse_candidates=[
            {"preview_scroll_px": int(lag), "score": float(score)}
            for lag, score in coarse
        ],
    )


def remove_adjacent_duplicate_frames(
    paths: Sequence[Path],
    body_previews: Sequence[np.ndarray],
    *,
    ncc_threshold: float = 0.9995,
    mae_threshold: float = 0.5,
) -> tuple[list[int], list[dict]]:
    keep = [0]
    removed: list[dict] = []

    for index in range(1, len(paths)):
        first = body_previews[keep[-1]]
        second = body_previews[index]
        height = min(first.shape[0], second.shape[0])
        width = min(first.shape[1], second.shape[1])
        first = first[:height, :width].astype(np.float32)
        second = second[:height, :width].astype(np.float32)

        ncc = float(cv2.matchTemplate(first, second, cv2.TM_CCOEFF_NORMED)[0, 0])
        mae = float(np.abs(first - second).mean())

        if ncc >= ncc_threshold and mae <= mae_threshold:
            removed.append(
                {
                    "path": str(paths[index]),
                    "duplicate_of": str(paths[keep[-1]]),
                    "ncc": ncc,
                    "mae": mae,
                }
            )
        else:
            keep.append(index)

    return keep, removed


def remove_transient_scrollbar(
    rgb_body: np.ndarray,
    *,
    right_fraction: float = 0.035,
) -> tuple[np.ndarray, list[dict]]:
    """Detect and inpaint narrow moving scroll indicators near the right edge."""
    height, width, _ = rgb_body.shape
    strip_start = max(0, width - round(width * right_fraction))
    gray = cv2.cvtColor(rgb_body, cv2.COLOR_RGB2GRAY)
    strip = gray[:, strip_start:]

    mask = (((strip >= 90) & (strip <= 235)).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    full_mask = np.zeros_like(gray, dtype=np.uint8)
    components: list[dict] = []

    max_width = max(4, round(width * 0.012))
    min_height = max(24, round(height * 0.02))

    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        aspect = component_height / max(component_width, 1)
        near_outer_edge = x + component_width / 2 >= strip.shape[1] * 0.45

        if (
            component_height >= min_height
            and component_width <= max_width
            and aspect >= 4
            and area >= component_height
            and near_outer_edge
        ):
            local = labels[y : y + component_height, x : x + component_width] == label
            target = full_mask[
                y : y + component_height,
                strip_start + x : strip_start + x + component_width,
            ]
            target[local] = 255
            components.append(
                {
                    "x": int(strip_start + x),
                    "y": int(y),
                    "width": int(component_width),
                    "height": int(component_height),
                }
            )

    if not components:
        return rgb_body, components

    full_mask = cv2.dilate(full_mask, np.ones((5, 5), np.uint8), iterations=1)
    cleaned = cv2.inpaint(rgb_body, full_mask, 3, cv2.INPAINT_TELEA)
    return cleaned, components


def stitch(
    paths: Sequence[Path],
    joins: Sequence[Join],
    *,
    header: int,
    footer: int,
    output: Path,
    remove_scrollbars: bool,
    png_compression: int,
) -> tuple[tuple[int, int], list[dict], list[dict]]:
    sizes = [load_rgb(path).size for path in paths]
    widths = {width for width, _ in sizes}
    if len(widths) != 1:
        raise ValueError("All screenshots must have the same width; resizing is intentionally disabled.")

    width = sizes[0][0]
    body_heights = [height - header - footer for _, height in sizes]

    starts = [0] + [join.seam_right_px for join in joins]
    ends = [join.seam_left_px for join in joins] + [body_heights[-1]]

    for index, (start, end, body_height) in enumerate(zip(starts, ends, body_heights)):
        if not 0 <= start < end <= body_height:
            raise RuntimeError(
                f"Invalid source segment for image {index}: start={start}, end={end}, body={body_height}"
            )

    total_height = header + sum(end - start for start, end in zip(starts, ends)) + footer
    canvas = Image.new("RGB", (width, total_height), "white")

    provenance: list[dict] = []
    scrollbar_report: list[dict] = []
    output_y = 0

    first = load_rgb(paths[0])
    if header:
        canvas.paste(first.crop((0, 0, width, header)), (0, output_y))
        provenance.append(
            {
                "source": str(paths[0]),
                "source_y": [0, header],
                "output_y": [output_y, output_y + header],
                "region": "header",
            }
        )
        output_y += header

    for path, start, end in zip(paths, starts, ends):
        image = load_rgb(path)
        rgb = np.asarray(image)
        body_end = rgb.shape[0] - footer if footer else rgb.shape[0]
        body = rgb[header:body_end]

        if remove_scrollbars:
            body, components = remove_transient_scrollbar(body)
            if components:
                scrollbar_report.append({"source": str(path), "components": components})

        segment = Image.fromarray(body[start:end], mode="RGB")
        canvas.paste(segment, (0, output_y))
        provenance.append(
            {
                "source": str(path),
                "source_y": [header + start, header + end],
                "output_y": [output_y, output_y + segment.height],
                "region": "scroll_body",
            }
        )
        output_y += segment.height

    last = load_rgb(paths[-1])
    if footer:
        source_start = last.height - footer
        canvas.paste(last.crop((0, source_start, width, last.height)), (0, output_y))
        provenance.append(
            {
                "source": str(paths[-1]),
                "source_y": [source_start, last.height],
                "output_y": [output_y, output_y + footer],
                "region": "footer",
            }
        )
        output_y += footer

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", compress_level=png_compression, optimize=False)
    return canvas.size, provenance, scrollbar_report


def parse_band(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Header/footer must be non-negative or 'auto'.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Screenshots in top-to-bottom order.")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="JSON audit report path.")
    parser.add_argument("--sort-name", action="store_true", help="Natural-sort inputs by filename.")
    parser.add_argument("--header", type=parse_band, default=None, help="Fixed top pixels or 'auto'.")
    parser.add_argument("--footer", type=parse_band, default=None, help="Fixed bottom pixels or 'auto'.")
    parser.add_argument("--preview-width", type=int, default=320)
    parser.add_argument("--refinement-width", type=int, default=320)
    parser.add_argument("--min-overlap", type=int, default=80, help="Minimum overlap in full-resolution pixels.")
    parser.add_argument("--max-overlap-ratio", type=float, default=0.98)
    parser.add_argument("--top-candidates", type=int, default=10)
    parser.add_argument("--refinement-radius", type=int, default=8)
    parser.add_argument("--min-similarity", type=float, default=0.82)
    parser.add_argument("--min-margin", type=float, default=0.01)
    parser.add_argument("--max-seam-mae", type=float, default=18.0)
    parser.add_argument("--allow-low-confidence", action="store_true")
    parser.add_argument("--keep-scrollbars", action="store_true")
    parser.add_argument("--png-compression", type=int, choices=range(0, 10), default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = list(args.inputs)
    if args.sort_name:
        paths.sort(key=natural_key)

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    previews: list[np.ndarray] = []
    scales: list[float] = []
    sizes: list[tuple[int, int]] = []
    for path in paths:
        preview, scale, size = make_preview(path, args.preview_width)
        previews.append(preview)
        scales.append(scale)
        sizes.append(size)

    if len({width for width, _ in sizes}) != 1:
        raise ValueError("Input widths differ. Refusing to resize because it would alter evidence pixels.")
    if max(scales) - min(scales) > 1e-9:
        raise ValueError("Preview scales differ unexpectedly.")
    preview_scale = scales[0]

    auto_top, auto_footer, detection = detect_static_bands(previews)
    header = args.header if args.header is not None else round(auto_top / preview_scale)
    footer = args.footer if args.footer is not None else round(auto_footer / preview_scale)

    body_previews = [
        preview[
            round(header * preview_scale) : preview.shape[0] - round(footer * preview_scale)
            if footer
            else preview.shape[0]
        ]
        for preview in previews
    ]

    keep_indices, duplicate_report = remove_adjacent_duplicate_frames(paths, body_previews)
    paths = [paths[index] for index in keep_indices]
    body_previews = [body_previews[index] for index in keep_indices]

    if len(paths) == 1:
        image = load_rgb(paths[0])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output, format="PNG", compress_level=args.png_compression, optimize=False)
        report = {
            "inputs": [str(path) for path in paths],
            "duplicates_removed": duplicate_report,
            "output": str(args.output),
            "output_size": list(image.size),
            "note": "Only one unique screenshot; no stitching was required.",
        }
        report_path = args.report or args.output.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    descriptors = [row_descriptor(body) for body in body_previews]

    joins: list[Join] = []
    for index in range(len(paths) - 1):
        join = match_pair(
            paths[index],
            paths[index + 1],
            descriptors[index],
            descriptors[index + 1],
            preview_scale=preview_scale,
            header=header,
            footer=footer,
            min_overlap_px=args.min_overlap,
            max_overlap_ratio=args.max_overlap_ratio,
            top_candidates=args.top_candidates,
            refinement_radius=args.refinement_radius,
            refinement_width=args.refinement_width,
        )

        low_confidence = (
            join.similarity < args.min_similarity
            or join.distinct_margin < args.min_margin
            or join.seam_row_mae > args.max_seam_mae
        )
        if low_confidence and not args.allow_low_confidence:
            raise RuntimeError(
                "Unsafe join rejected:\n"
                f"  {Path(join.left).name} -> {Path(join.right).name}\n"
                f"  similarity={join.similarity:.4f} (minimum {args.min_similarity})\n"
                f"  distinct margin={join.distinct_margin:.4f} (minimum {args.min_margin})\n"
                f"  seam MAE={join.seam_row_mae:.2f} (maximum {args.max_seam_mae})\n"
                "Check input order or override header/footer. Use --allow-low-confidence only after review."
            )
        joins.append(join)

    output_size, provenance, scrollbar_report = stitch(
        paths,
        joins,
        header=header,
        footer=footer,
        output=args.output,
        remove_scrollbars=not args.keep_scrollbars,
        png_compression=args.png_compression,
    )

    report = {
        "inputs": [str(path) for path in paths],
        "duplicates_removed": duplicate_report,
        "fixed_ui": {
            "header_px": header,
            "footer_px": footer,
            "auto_detection": detection,
        },
        "joins": [asdict(join) for join in joins],
        "scrollbars_removed": scrollbar_report,
        "provenance": provenance,
        "output": str(args.output),
        "output_size": list(output_size),
        "settings": {
            "preview_width": args.preview_width,
            "refinement_width": args.refinement_width,
            "min_overlap_px": args.min_overlap,
            "max_overlap_ratio": args.max_overlap_ratio,
            "min_similarity": args.min_similarity,
            "min_margin": args.min_margin,
            "max_seam_mae": args.max_seam_mae,
            "png_compression": args.png_compression,
        },
    }

    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved image:  {args.output}")
    print(f"Saved report: {report_path}")
    print(f"Output size:  {output_size[0]} x {output_size[1]}")
    for join in joins:
        print(
            f"{Path(join.left).name} -> {Path(join.right).name}: "
            f"scroll={join.scroll_px}px, overlap={join.overlap_px}px, "
            f"similarity={join.similarity:.4f}, margin={join.distinct_margin:.4f}"
        )


if __name__ == "__main__":
    main()
