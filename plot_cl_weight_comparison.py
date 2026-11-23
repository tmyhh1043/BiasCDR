"""Compare raw Selective-CL weight quantiles across saved JSON reports."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relative", action="store_true",
                        help="Plot weights divided by their direction-wide mean; show uniform CL at 1")
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    rows.sort(key=lambda row: row["direction"])
    width, height = 1600, 250 + 170 * len(rows)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title = ImageFont.truetype(font_path, 40)
    body = ImageFont.truetype(font_path, 24)
    small = ImageFont.truetype(font_path, 20)
    heading = "CLの実効重みの近似：一様CLとの比較" if args.relative else "CLの生の重み：データセット間比較"
    draw.text((65, 40), heading, font=title, fill="#17243a")
    draw.text((65, 102), "点＝中央値、太線＝中央50%、細線＝5～95パーセンタイル", font=body,
              fill="#526174")
    plot_left, plot_right = 540, 1510
    x_min, x_max = (.5, 1.7) if args.relative else (.25, .75)
    def xpos(value):
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)
    ticks = (.6, .8, 1., 1.2, 1.4, 1.6) if args.relative else (.3, .4, .5, .6, .7)
    for tick in ticks:
        x = xpos(tick)
        color = "#e58a3b" if args.relative and tick == 1. else "#e5ebf0"
        draw.line((x, 175, x, height - 90), fill=color, width=3 if tick == 1. else 2)
        draw.text((x - 22, height - 68), f"{tick:.1f}", font=small, fill="#526174")
    if args.relative:
        draw.text((xpos(1.) + 12, 163), "一様CL = 1", font=small, fill="#b66824")
    for i, row in enumerate(rows):
        y = 225 + i * 170
        q = row["relative_weight" if args.relative else "raw_weight"]
        draw.text((65, y - 24), row["direction"], font=body, fill="#17243a")
        draw.text((65, y + 13), f"seed {row['seed']}  |  {row['n_users']:,} users",
                  font=small, fill="#637185")
        draw.line((xpos(q["p05"]), y, xpos(q["p95"]), y), fill="#7eb3e4", width=8)
        draw.line((xpos(q["p25"]), y, xpos(q["p75"]), y), fill="#2d73bd", width=22)
        mx = xpos(q["median"])
        draw.ellipse((mx - 12, y - 12, mx + 12, y + 12), fill="#173d6d")
        draw.text((mx - 40, y + 28), f"{q['median']:.3f}", font=small, fill="#173d6d")
        if not args.relative:
            draw.text((plot_right - 130, y - 55), f"mean {q['mean']:.3f}", font=small,
                      fill="#526174")
    axis_label = "平均1の相対重み（バッチ実効重みの近似）" if args.relative else "正規化前の重み"
    draw.text((plot_left + 160, height - 42), axis_label, font=body, fill="#26374a")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
