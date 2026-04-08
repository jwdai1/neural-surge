#!/usr/bin/env python3
"""
CYBERTRUCK — Tesla Stock Data Art (Variation B)
Daily OHLC data as angular polygon mesh.
Low price = coarse polygons, high price = refined mesh.
Stainless steel aesthetic with angular brutalism.

Output: output/cybertruck.png (6000x4000)
"""

import json
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import WIDTH, HEIGHT, BG_COLOR, SEED, pseudo_random, postprocess_and_save

# ─── Cybertruck Palette: Stainless Steel ─────────────────────────────
BG = (8, 8, 12)  # near-black with blue tint
STEEL_STOPS = [
    (-1.0, (40,  42,  48)),    # dark steel — strong decline
    (-0.5, (60,  65,  72)),    # gunmetal
    (-0.1, (90,  95,  105)),   # brushed steel dark
    ( 0.0, (130, 135, 145)),   # brushed steel
    ( 0.3, (170, 175, 185)),   # polished steel
    ( 0.6, (200, 205, 215)),   # bright steel
    ( 0.8, (220, 225, 235)),   # chrome
    ( 1.0, (245, 248, 255)),   # white chrome highlight
]

# Accent: orange construction sparks for extreme moves
SPARK_COLOR = (255, 140, 40)

def steel_color(norm_val):
    if norm_val <= STEEL_STOPS[0][0]:
        return STEEL_STOPS[0][1]
    if norm_val >= STEEL_STOPS[-1][0]:
        return STEEL_STOPS[-1][1]
    for i in range(len(STEEL_STOPS) - 1):
        t0, c0 = STEEL_STOPS[i]
        t1, c1 = STEEL_STOPS[i + 1]
        if t0 <= norm_val <= t1:
            f = (norm_val - t0) / (t1 - t0)
            f = f * f * (3 - 2 * f)
            return tuple(int(c0[j] + (c1[j] - c0[j]) * f) for j in range(3))
    return STEEL_STOPS[0][1]


# ─── Data Loading ────────────────────────────────────────────────────
MARGIN_LEFT = 100
MARGIN_RIGHT = 80
MARGIN_TOP = 150
MARGIN_BOTTOM = 150

def load_daily():
    """Load daily data and resample to weekly OHLC for larger polygons."""
    path = os.path.join(os.path.dirname(__file__) or '.', "data", "tsla_daily.json")
    with open(path) as f:
        daily = json.load(f)

    # Group by week (every 5 trading days)
    chunk_size = 5
    rows = []
    for start in range(0, len(daily), chunk_size):
        chunk = daily[start:start + chunk_size]
        if not chunk:
            continue
        rows.append({
            "date": chunk[0]["date"],
            "open": chunk[0]["open"],
            "high": max(d["high"] for d in chunk),
            "low": min(d["low"] for d in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(d["volume"] for d in chunk),
        })

    # Compute weekly change normalized
    for i, row in enumerate(rows):
        if i == 0:
            row["delta_raw"] = 0.0
        else:
            prev = rows[i - 1]["close"]
            row["delta_raw"] = (row["close"] - prev) / prev if prev > 0 else 0.0

    abs_deltas = sorted(abs(r["delta_raw"]) for r in rows)
    p95 = abs_deltas[int(len(abs_deltas) * 0.95)] if abs_deltas else 1.0
    scale = p95 if p95 > 0 else 1.0
    for row in rows:
        row["delta_norm"] = max(-1.0, min(1.0, row["delta_raw"] / scale))

    max_vol = max(r["volume"] for r in rows)
    for row in rows:
        row["vol_norm"] = row["volume"] / max_vol

    return rows


def map_coords(rows):
    n = len(rows)
    usable_w = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    usable_h = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    all_prices = []
    for r in rows:
        all_prices.extend([r["open"], r["high"], r["low"], r["close"]])
    log_min = math.log(max(0.1, min(all_prices)))
    log_max = math.log(max(all_prices))
    log_range = log_max - log_min if log_max > log_min else 1.0

    def price_to_y(p):
        log_p = math.log(max(0.1, p))
        norm = (log_p - log_min) / log_range
        return MARGIN_TOP + (1.0 - norm) * usable_h

    for i, row in enumerate(rows):
        row["x"] = MARGIN_LEFT + (i / max(1, n - 1)) * usable_w
        row["y_open"] = price_to_y(row["open"])
        row["y_high"] = price_to_y(row["high"])
        row["y_low"] = price_to_y(row["low"])
        row["y_close"] = price_to_y(row["close"])

    return rows


# ─── Layer 1: Angular Grid Background ───────────────────────────────
def render_grid(img):
    draw = ImageDraw.Draw(img)
    # Triangular grid pattern — very faint
    spacing = 40
    grid_c = (20, 20, 26)
    for y in range(0, HEIGHT, spacing):
        offset = (spacing // 2) if (y // spacing) % 2 else 0
        for x in range(-spacing, WIDTH + spacing, spacing):
            ax, ay = x + offset, y
            bx, by = x + spacing // 2 + offset, y + spacing
            cx_, cy_ = x + spacing + offset, y
            draw.line([(ax, ay), (bx, by)], fill=grid_c, width=1)
            draw.line([(bx, by), (cx_, cy_)], fill=grid_c, width=1)
            draw.line([(ax, ay), (cx_, cy_)], fill=grid_c, width=1)


# ─── Layer 2: OHLC Polygon Mesh ─────────────────────────────────────
def render_polygon_mesh(img, rows):
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    n = len(rows)
    # Sample every N days to control density — adaptive based on price level
    # Low price era = coarse (skip more), high price era = fine (skip less)
    max_price = max(r["close"] for r in rows)

    for i in range(n - 1):
        row = rows[i]
        nxt = rows[i + 1]

        # Adaptive sampling: high price = draw more, low price = skip more
        price_frac = row["close"] / max_price
        skip_prob = 0.25 * (1.0 - price_frac)  # high price → almost all drawn
        if pseudo_random(i, 0, 77) < skip_prob:
            continue

        x0 = row["x"]
        x1 = nxt["x"]

        # 4 vertices from OHLC: form angular quad
        # Top-left: high of current, Top-right: high of next
        # Bottom-left: low of current, Bottom-right: low of next
        # Amplify high-low spread for visual impact
        spread_amp = 4.0
        mid_y0 = (row["y_high"] + row["y_low"]) / 2
        mid_y1 = (nxt["y_high"] + nxt["y_low"]) / 2
        half0 = (row["y_low"] - row["y_high"]) / 2 * spread_amp
        half1 = (nxt["y_low"] - nxt["y_high"]) / 2 * spread_amp

        tl = (x0, mid_y0 - half0)
        tr = (x1, mid_y1 - half1)
        bl = (x0, mid_y0 + half0)
        br = (x1, mid_y1 + half1)

        # Split quad into 2 triangles (Cybertruck angular aesthetic)
        delta = row["delta_norm"]
        color = steel_color(delta)

        # Alpha based on volume and time
        time_frac = i / max(1, n - 1)
        base_alpha = int(50 + time_frac * 50 + abs(delta) * 70 + row["vol_norm"] * 40)
        base_alpha = min(base_alpha, 200)

        # Triangle 1: top-left, top-right, bottom-left
        draw.polygon([tl, tr, bl], fill=color + (base_alpha,))
        # Triangle 2: top-right, bottom-right, bottom-left
        draw.polygon([tr, br, bl], fill=color + (int(base_alpha * 0.7),))

        # Edge lines — sharp angular strokes
        edge_alpha = min(255, int(base_alpha * 1.5))
        edge_color = tuple(min(255, c + 40) for c in color) + (edge_alpha,)
        draw.line([tl, tr], fill=edge_color, width=2)
        draw.line([bl, br], fill=edge_color, width=2)

        # Diagonal — the Cybertruck crease
        draw.line([tr, bl], fill=edge_color, width=2)
        # Outer edges
        draw.line([tl, bl], fill=edge_color, width=1)
        draw.line([tr, br], fill=edge_color, width=1)

        # For very high volatility, add extra subdivision triangles
        if abs(delta) > 0.5:
            mid_x = (x0 + x1) / 2
            mid_y_top = (row["y_high"] + nxt["y_high"]) / 2
            mid_y_bot = (row["y_low"] + nxt["y_low"]) / 2
            mid_top = (mid_x, mid_y_top)
            mid_bot = (mid_x, mid_y_bot)

            sub_alpha = int(base_alpha * 0.8)
            brighter = tuple(min(255, c + 50) for c in color)
            draw.polygon([tl, mid_top, bl], fill=brighter + (sub_alpha,))
            draw.polygon([mid_top, tr, mid_bot], fill=brighter + (int(sub_alpha * 0.6),))

            # Extra edges
            draw.line([tl, mid_bot], fill=edge_color, width=1)
            draw.line([mid_top, br], fill=edge_color, width=1)

    result = Image.alpha_composite(img.convert('RGBA'), layer)
    return result.convert('RGB')


# ─── Layer 3: Price Spine (angular polyline) ─────────────────────────
def render_spine(img, rows):
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Draw close price as angular polyline (no curves — Cybertruck has no curves)
    n = len(rows)
    # Sample at reasonable density
    step = max(1, n // 600)

    sampled = [rows[i] for i in range(0, n, step)]
    if sampled[-1] != rows[-1]:
        sampled.append(rows[-1])

    for i in range(len(sampled) - 1):
        r0 = sampled[i]
        r1 = sampled[i + 1]
        delta = r0["delta_norm"]
        color = steel_color(delta)
        # Brighter than mesh
        bright = tuple(min(255, c + 60) for c in color)

        w = max(2, int(2 + abs(delta) * 4 + r0["vol_norm"] * 3))
        alpha = int(180 + abs(delta) * 60)
        alpha = min(alpha, 255)

        draw.line([(r0["x"], r0["y_close"]), (r1["x"], r1["y_close"])],
                  fill=bright + (alpha,), width=w)

    result = Image.alpha_composite(img.convert('RGBA'), layer)
    return result.convert('RGB')


# ─── Layer 4: Spark Effects (extreme days) ───────────────────────────
def render_sparks(img, rows):
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for i, row in enumerate(rows):
        if abs(row["delta_norm"]) < 0.6:
            continue

        cx, cy = row["x"], row["y_close"]
        intensity = abs(row["delta_norm"])

        # Spark: small bright angular lines radiating outward
        n_sparks = int(4 + intensity * 12)
        for s in range(n_sparks):
            rand_a = pseudo_random(i, s, 200)
            rand_l = pseudo_random(i, s, 201)
            rand_w = pseudo_random(i, s, 202)

            angle = rand_a * math.pi * 2
            length = 10 + rand_l * 50 * intensity

            # Sparks are straight lines (angular, no curves)
            ex = cx + math.cos(angle) * length
            ey = cy + math.sin(angle) * length

            # Midpoint kink for angular feel
            mid_angle = angle + (pseudo_random(i, s, 203) - 0.5) * 0.8
            mid_len = length * (0.3 + rand_w * 0.4)
            mx = cx + math.cos(mid_angle) * mid_len
            my = cy + math.sin(mid_angle) * mid_len

            spark_alpha = int(120 + intensity * 100)
            spark_alpha = min(spark_alpha, 255)

            # Orange-white spark
            if rand_w > 0.7:
                sc = (255, 220, 180)
            else:
                sc = SPARK_COLOR

            draw.line([(cx, cy), (mx, my)], fill=sc + (spark_alpha,), width=2)
            draw.line([(mx, my), (ex, ey)], fill=sc + (int(spark_alpha * 0.5),), width=1)

        # Core point
        core_r = int(2 + intensity * 4)
        draw.ellipse([(cx-core_r, cy-core_r), (cx+core_r, cy+core_r)],
                     fill=(255, 200, 150, int(200 + intensity * 55)))

    layer_blurred = layer.filter(ImageFilter.GaussianBlur(radius=4))
    combined = Image.alpha_composite(layer_blurred, layer)
    result = Image.alpha_composite(img.convert('RGBA'), combined)
    return result.convert('RGB')


# ─── Main ────────────────────────────────────────────────────────────
def render():
    print(f"Rendering CYBERTRUCK at {WIDTH}x{HEIGHT}...")
    np.random.seed(SEED)

    rows = load_daily()
    rows = map_coords(rows)
    print(f"  Data: {len(rows)} days, ${rows[0]['close']:.2f} to ${rows[-1]['close']:.2f}")

    img = Image.new('RGB', (WIDTH, HEIGHT), BG)

    print("  Layer 1: Angular grid...")
    render_grid(img)

    print("  Layer 2: Polygon mesh...")
    img = render_polygon_mesh(img, rows)

    print("  Layer 3: Price spine...")
    img = render_spine(img, rows)

    print("  Layer 4: Sparks...")
    img = render_sparks(img, rows)

    os.makedirs(os.path.join(os.path.dirname(__file__) or '.', "output"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__) or '.', "output", "cybertruck.png")
    print("  Post-processing...")

    # Custom post-process: lighter glow, no warm tint
    from common import apply_glow, apply_vignette
    print("  Applying glow...")
    img = apply_glow(img, intensity=0.3, radius=30)
    print("  Applying vignette...")
    img = apply_vignette(img, strength=0.5, center_offset=(-200, 100))
    img.save(out_path, 'PNG', optimize=True)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    print("Done: Cybertruck")


if __name__ == "__main__":
    render()
