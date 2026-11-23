"""Plot quantile-based popularity-only CL-weight simulations from saved reports."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
KEYS = {1: "p01", 5: "p05", 25: "p25", 50: "median", 75: "p75", 95: "p95", 99: "p99"}


def simulated_quantiles(popularity, gamma):
    return [math.exp(-gamma * popularity[KEYS[100 - p]]) for p in PERCENTILES]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    rows.sort(key=lambda item: item["direction"])
    image = Image.new("RGB", (1550, 1050), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title = ImageFont.truetype(font_path, 38)
    body = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 19)
    draw.text((65, 36), "人気度のみのCL重み：分位点からのシミュレーション", font=title,
              fill="#17243a")
    draw.text((65, 92), "保存済み最終モデルの人気度傾向を固定。再学習後の分布ではありません。",
              font=body, fill="#526174")
    colors = {"現在 (β=0.5, γ=2)": "#2d73bd", "人気度のみ (γ=1)": "#2a9d77",
              "人気度のみ (γ=2)": "#db8040"}
    for idx, row in enumerate(rows):
        col, line = idx % 2, idx // 2
        left, top = 75 + col * 755, 165 + line * 390
        width, height = 650, 265
        draw.text((left, top), row["direction"], font=body, fill="#17243a")
        x0, x1, y0, y1 = left + 55, left + width, top + 40, top + height
        for value in (.2, .4, .6):
            y = y1 - (value - .1) / .6 * (y1 - y0)
            draw.line((x0, y, x1, y), fill="#e5ebf0", width=2)
            draw.text((left, y - 12), f"{value:.1f}", font=small, fill="#526174")
        curves = {
            "現在 (β=0.5, γ=2)": [row["raw_weight"][KEYS[p]] for p in PERCENTILES],
            "人気度のみ (γ=1)": simulated_quantiles(row["popularity_risk"], 1),
            "人気度のみ (γ=2)": simulated_quantiles(row["popularity_risk"], 2),
        }
        for name, values in curves.items():
            points = [(x0 + (p - 1) / 98 * (x1 - x0),
                       y1 - (value - .1) / .6 * (y1 - y0))
                      for p, value in zip(PERCENTILES, values)]
            draw.line(points, fill=colors[name], width=5, joint="curve")
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[name])
        for p in (1, 25, 50, 75, 99):
            x = x0 + (p - 1) / 98 * (x1 - x0)
            draw.text((x - 18, y1 + 10), f"{p}%", font=small, fill="#526174")
    for idx, (name, color) in enumerate(colors.items()):
        x = 130 + idx * 470
        draw.line((x, 979, x + 50, 979), fill=color, width=6)
        draw.text((x + 63, 962), name, font=body, fill="#26374a")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
