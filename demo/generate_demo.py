#!/usr/bin/env python3
"""Generate a fully synthetic, non-personal demo dataset for the stitcher."""

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W, H = 1170, 2532
HEADER = 370
FOOTER = 190
BODY_H = H - HEADER - FOOTER
ROW_H = 285
REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REGULAR, size=size)


TRIPS = [
    ("Vancouver to Toronto", "Air Canada", "Sep 12, 2025", "10:30 AM - 6:45 PM", "TRIP-1001"),
    ("Toronto to Montreal", "Air Canada", "Oct 03, 2025", "8:15 AM - 9:45 AM", "TRIP-1002"),
    ("Montreal to New York", "Delta Airlines", "Oct 18, 2025", "1:30 PM - 3:45 PM", "TRIP-1003"),
    ("New York to Boston", "JetBlue", "Nov 05, 2025", "7:20 AM - 8:40 AM", "TRIP-1004"),
    ("Boston to Chicago", "United", "Nov 20, 2025", "11:10 AM - 1:25 PM", "TRIP-1005"),
    ("Chicago to San Francisco", "United", "Dec 05, 2025", "2:15 PM - 5:40 PM", "TRIP-1006"),
    ("San Francisco to Seattle", "Alaska Airlines", "Dec 20, 2025", "9:00 AM - 11:10 AM", "TRIP-1007"),
    ("Seattle to Vancouver", "Air Canada", "Jan 08, 2026", "1:45 PM - 3:10 PM", "TRIP-1008"),
    ("Vancouver to Tokyo", "ANA", "Jan 22, 2026", "12:00 PM - 4:30 PM", "TRIP-1009"),
    ("Tokyo to Seoul", "Korean Air", "Feb 10, 2026", "9:30 AM - 12:10 PM", "TRIP-1010"),
    ("Seoul to Singapore", "Singapore Airlines", "Feb 28, 2026", "7:45 PM - 1:15 AM", "TRIP-1011"),
    ("Singapore to Sydney", "Qantas", "Mar 18, 2026", "8:20 AM - 6:45 PM", "TRIP-1012"),
    ("Sydney to Auckland", "Air New Zealand", "Apr 05, 2026", "10:15 AM - 3:20 PM", "TRIP-1013"),
    ("Auckland to Los Angeles", "United", "Apr 22, 2026", "6:50 PM - 2:10 PM", "TRIP-1014"),
    ("Los Angeles to Vancouver", "Air Canada", "May 10, 2026", "9:00 AM - 12:20 PM", "TRIP-1015"),
    ("Vancouver to Calgary", "WestJet", "May 28, 2026", "7:40 AM - 10:05 AM", "TRIP-1016"),
    ("Calgary to Winnipeg", "WestJet", "Jun 11, 2026", "2:25 PM - 4:50 PM", "TRIP-1017"),
    ("Winnipeg to Toronto", "Porter", "Jun 29, 2026", "6:10 PM - 9:30 PM", "TRIP-1018"),
]

CONTENT_H = len(TRIPS) * ROW_H + 40


def draw_header(img):
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, HEADER), fill="white")
    d.text((55, 42), "9:41", font=font(44, True), fill=(20, 20, 20))
    d.text((W // 2, 125), "My Trips", anchor="mm", font=font(52, True), fill=(15, 15, 15))
    d.text((54, 120), "<", font=font(58, True), fill=(0, 122, 255))
    x0, y0, x1, y1 = 60, 215, W - 60, 320
    d.rounded_rectangle((x0, y0, x1, y1), radius=48, fill=(242, 244, 247))
    seg_w = (x1 - x0) // 3
    d.rounded_rectangle((x0, y0, x0 + seg_w, y1), radius=48, fill=(0, 122, 255))
    d.text((x0 + seg_w / 2, (y0 + y1) / 2), "Upcoming", anchor="mm", font=font(33, True), fill="white")
    d.text((x0 + seg_w * 1.5, (y0 + y1) / 2), "Past", anchor="mm", font=font(33, True), fill=(80, 80, 80))
    d.text((x0 + seg_w * 2.5, (y0 + y1) / 2), "Cancelled", anchor="mm", font=font(33, True), fill=(80, 80, 80))
    d.line((0, HEADER - 1, W, HEADER - 1), fill=(225, 225, 225), width=2)


def draw_footer(img):
    d = ImageDraw.Draw(img)
    y = H - FOOTER
    d.rectangle((0, y, W, H), fill="white")
    d.line((0, y, W, y), fill=(220, 220, 220), width=2)
    labels = ["Home", "Trips", "Explore", "Account"]
    xs = [120, 390, 675, 960]
    for label, x in zip(labels, xs):
        color = (0, 122, 255) if label == "Trips" else (120, 120, 120)
        d.ellipse((x - 18, y + 30, x + 18, y + 66), outline=color, width=4)
        d.text((x, y + 118), label, anchor="mm", font=font(28, label == "Trips"), fill=color)


def make_body():
    body = Image.new("RGB", (W, CONTENT_H), "white")
    d = ImageDraw.Draw(body)
    y = 20
    for idx, (route, airline, date, tm, ref) in enumerate(TRIPS):
        d.rounded_rectangle((45, y, W - 45, y + ROW_H - 22), radius=28,
                            fill=(250, 250, 252), outline=(230, 230, 233), width=2)
        color = ((idx * 37) % 190 + 40, (idx * 71) % 170 + 50, (idx * 23) % 140 + 70)
        d.rounded_rectangle((65, y + 28, 255, y + ROW_H - 50), radius=24, fill=color)
        tx = 285
        d.text((tx, y + 28), route, font=font(34, True), fill=(20, 20, 20))
        d.text((tx, y + 83), airline, font=font(29), fill=(65, 65, 65))
        d.text((tx, y + 126), date, font=font(29), fill=(65, 65, 65))
        d.text((tx, y + 169), tm, font=font(29), fill=(65, 65, 65))
        d.rounded_rectangle((tx, y + 214, tx + 190, y + 255), radius=18, fill=(214, 247, 222))
        d.text((tx + 95, y + 234), "Confirmed", anchor="mm", font=font(23, True), fill=(25, 125, 55))
        d.text((W - 75, y + 220), ref, anchor="ra", font=font(22), fill=(115, 115, 115))
        y += ROW_H
    return body


def main():
    body = make_body()
    scroll_positions = [0, 1020, 2040, 3060]
    scroll_positions = [min(p, CONTENT_H - BODY_H) for p in scroll_positions]

    shots = []
    for i, pos in enumerate(scroll_positions, 1):
        img = Image.new("RGB", (W, H), "white")
        draw_header(img)
        img.paste(body.crop((0, pos, W, pos + BODY_H)), (0, HEADER))
        draw_footer(img)
        path = OUT / f"screenshot_{i:02d}.png"
        img.save(path)
        shots.append(path)

    stitched_h = HEADER + CONTENT_H + FOOTER
    stitched = Image.new("RGB", (W, stitched_h), "white")
    tmp = Image.new("RGB", (W, H), "white")
    draw_header(tmp)
    draw_footer(tmp)
    stitched.paste(tmp.crop((0, 0, W, HEADER)), (0, 0))
    stitched.paste(body, (0, HEADER))
    stitched.paste(tmp.crop((0, H - FOOTER, W, H)), (0, HEADER + CONTENT_H))
    stitched.save(OUT / "expected_stitched.png")

    scale = 0.20
    tw, th = int(W * scale), int(H * scale)
    margin = 24
    sheet = Image.new("RGB", (4 * tw + 5 * margin, th + 2 * margin), (242, 242, 242))
    x = margin
    for path in shots:
        im = Image.open(path).resize((tw, th), Image.Resampling.LANCZOS)
        sheet.paste(im, (x, margin))
        x += tw + margin
    sheet.save(OUT / "contact_sheet.png")

    manifest = {
        "synthetic": True,
        "contains_personal_data": False,
        "viewport": {
            "width": W,
            "height": H,
            "header": HEADER,
            "footer": FOOTER,
            "scroll_body_height": BODY_H,
        },
        "scroll_positions_px": scroll_positions,
        "adjacent_body_overlap_px": [
            BODY_H - (scroll_positions[i + 1] - scroll_positions[i])
            for i in range(len(scroll_positions) - 1)
        ],
        "records": len(TRIPS),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
