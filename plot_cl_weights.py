"""Plot the saved CL-weight histogram against uniform CL, using Pillow."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.report.read_text(encoding="utf-8"))
    output = args.output or args.report.with_suffix(".png")

    groups = [
        ("0.75–1.00", "[0.75,1)"),
        ("1.00–1.25", "[1,1.25)"),
        ("1.25–1.50", "[1.25,1.5)"),
        ("1.50–2.00", "[1.5,2)"),
    ]
    counts = data["relative_weight_bins"]
    n = data["n_users"]
    weighted = [100 * counts[key] / n for _, key in groups]
    uniform = [0, 100, 0, 0]  # Every user has weight exactly 1.

    img = Image.new("RGB", (1400, 880), "#ffffff")
    draw = ImageDraw.Draw(img)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title = ImageFont.truetype(font_path, 42)
    body = ImageFont.truetype(font_path, 27)
    small = ImageFont.truetype(font_path, 22)
    draw.text((86, 55), "CLの相対重み分布：重み付き vs 一様", font=title, fill="#17243a")
    draw.text((88, 120), f"{data['direction']}  |  seed {data['seed']}  |  {n:,} users",
              font=body, fill="#526174")

    left, right, top, bottom = 120, 1320, 210, 682
    for tick in (0, 25, 50, 75, 100):
        y = bottom - tick / 100 * (bottom - top)
        draw.line((left, y, right, y), fill="#e4eaf0", width=2)
        draw.text((48, y - 15), f"{tick}%", font=small, fill="#526174")
    group_width = (right - left) / len(groups)
    bar_width = 76
    for i, ((label, _), a, b) in enumerate(zip(groups, weighted, uniform)):
        center = left + group_width * (i + .5)
        for value, x, color in ((a, center - 83, "#2d73bd"),
                                (b, center + 7, "#e58a3b")):
            y = bottom - value / 100 * (bottom - top)
            if value > 0:
                draw.rectangle((x, y, x + bar_width, bottom), fill=color)
            draw.text((x - 3, max(top - 31, y - 33)), f"{value:.1f}%", font=small, fill=color)
        box = draw.textbbox((0, 0), label, font=small)
        draw.text((center - (box[2] - box[0]) / 2, bottom + 18), label,
                  font=small, fill="#26374a")

    draw.rectangle((370, 764, 397, 791), fill="#2d73bd")
    draw.text((409, 757), "重み付きCL", font=body, fill="#26374a")
    draw.rectangle((700, 764, 727, 791), fill="#e58a3b")
    draw.text((739, 757), "一様CL（全員1）", font=body, fill="#26374a")
    draw.text((90, 832), "値は平均1に正規化。各区間のユーザー割合を表示。", font=small, fill="#526174")
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)
    print(output)


if __name__ == "__main__":
    main()
