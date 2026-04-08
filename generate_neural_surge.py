#!/usr/bin/env python3
"""
NEURAL SURGE — Tesla Stock Data Art
Bezier network + L-system dendrites on circuit grid.
Output: output/neural_surge.png (6000x4000)
"""

import json
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import (
    WIDTH, HEIGHT, BG_COLOR, SEED,
    metric_to_color, catmull_rom_chain, cubic_bezier,
    draw_tapered_line, pseudo_random,
    postprocess_and_save,
)

# ─── Layout Constants ────────────────────────────────────────────────
MARGIN_LEFT = 150
MARGIN_RIGHT = 100
MARGIN_TOP = 200
MARGIN_BOTTOM = 200


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING (Task 2)
# ═══════════════════════════════════════════════════════════════════════
def load_data():
    path = os.path.join(os.path.dirname(__file__) or '.', "data", "tsla_monthly.json")
    with open(path) as f:
        rows = json.load(f)

    for i, row in enumerate(rows):
        if i == 0:
            row["delta"] = 0.0
            row["delta_raw"] = 0.0
        else:
            prev = rows[i - 1]["close"]
            row["delta_raw"] = (row["close"] - prev) / prev
            row["delta"] = row["delta_raw"]

    abs_deltas = sorted(abs(r["delta_raw"]) for r in rows)
    p95 = abs_deltas[int(len(abs_deltas) * 0.95)] if abs_deltas else 1.0
    scale = p95 if p95 > 0 else 1.0
    for row in rows:
        row["delta_norm"] = max(-1.0, min(1.0, row["delta_raw"] / scale))

    volumes = [r["volume"] for r in rows]
    max_vol = max(volumes)
    for row in rows:
        row["vol_norm"] = row["volume"] / max_vol

    return rows


def map_to_canvas(rows):
    n = len(rows)
    usable_w = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    usable_h = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    prices = [r["close"] for r in rows]
    log_min = math.log(max(0.1, min(prices)))
    log_max = math.log(max(prices))
    log_range = log_max - log_min if log_max > log_min else 1.0

    for i, row in enumerate(rows):
        row["cx"] = MARGIN_LEFT + (i / max(1, n - 1)) * usable_w
        log_p = math.log(max(0.1, row["close"]))
        norm_y = (log_p - log_min) / log_range
        row["cy"] = MARGIN_TOP + (1.0 - norm_y) * usable_h

    return rows


# ═══════════════════════════════════════════════════════════════════════
# LAYER 1: CIRCUIT GRID (Task 3)
# ═══════════════════════════════════════════════════════════════════════
def render_circuit_grid(img, rows):
    draw = ImageDraw.Draw(img)

    # Fine background grid — faint PCB texture
    grid_color_fine = (20, 20, 20)
    for gx in range(0, WIDTH, 60):
        draw.line([(gx, 0), (gx, HEIGHT)], fill=grid_color_fine, width=1)
    for gy in range(0, HEIGHT, 60):
        draw.line([(0, gy), (WIDTH, gy)], fill=grid_color_fine, width=1)

    # Vertical lines: one per year
    years_seen = set()
    for row in rows:
        year = row["date"][:4]
        if year not in years_seen:
            years_seen.add(year)
            x = int(row["cx"])
            draw.line([(x, MARGIN_TOP - 80), (x, HEIGHT - MARGIN_BOTTOM + 80)],
                      fill=(32, 32, 32), width=1)
            # Node at top and bottom
            for ny in [MARGIN_TOP - 80, HEIGHT - MARGIN_BOTTOM + 80]:
                draw.ellipse([(x-4, ny-4), (x+4, ny+4)], fill=(40, 40, 40))

    # Horizontal lines: price levels
    prices_grid = [2, 5, 10, 25, 50, 100, 150, 200, 250, 300, 400]
    prices_all = [r["close"] for r in rows]
    log_min = math.log(max(0.1, min(prices_all)))
    log_max = math.log(max(prices_all))
    log_range = log_max - log_min
    usable_h = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    for price in prices_grid:
        log_p = math.log(price)
        if log_min <= log_p <= log_max:
            norm_y = (log_p - log_min) / log_range
            y = int(MARGIN_TOP + (1.0 - norm_y) * usable_h)
            # Dashed line
            for x in range(MARGIN_LEFT - 50, WIDTH - MARGIN_RIGHT + 50, 16):
                draw.line([(x, y), (x + 8, y)], fill=(28, 28, 28), width=1)

    # Circuit trace accents: L-shaped connectors at quarterly points
    for row in rows:
        month = row["date"][5:7]
        if month in ("01", "04", "07", "10"):
            x, y = int(row["cx"]), int(row["cy"])
            # Cross
            draw.line([(x-6, y), (x+6, y)], fill=(35, 32, 30), width=1)
            draw.line([(x, y-6), (x, y+6)], fill=(35, 32, 30), width=1)
            # L-shaped trace from node to grid
            rand_dir = pseudo_random(int(month), int(row["date"][:4]), 55)
            trace_len = 15 + int(rand_dir * 25)
            if rand_dir > 0.5:
                draw.line([(x, y), (x + trace_len, y)], fill=(30, 28, 26), width=1)
                draw.line([(x + trace_len, y), (x + trace_len, y - trace_len)],
                          fill=(30, 28, 26), width=1)
            else:
                draw.line([(x, y), (x, y + trace_len)], fill=(30, 28, 26), width=1)
                draw.line([(x, y + trace_len), (x - trace_len, y + trace_len)],
                          fill=(30, 28, 26), width=1)


# ═══════════════════════════════════════════════════════════════════════
# LAYER 2: TRUNK NERVE (Task 4)
# ═══════════════════════════════════════════════════════════════════════
def render_trunk(img, rows):
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    control_pts = [(row["cx"], row["cy"]) for row in rows]
    spline_pts = catmull_rom_chain(control_pts, points_per_segment=30)

    n_data = len(rows)
    n_spline = len(spline_pts)

    for i in range(len(spline_pts) - 1):
        data_idx = int((i / n_spline) * n_data)
        data_idx = min(data_idx, n_data - 1)
        row = rows[data_idx]

        color = metric_to_color(row["delta_norm"])

        base_w = 3 + row["vol_norm"] * 10
        volatility_boost = abs(row["delta_norm"]) * 5
        w = max(3, int(base_w + volatility_boost))

        time_frac = data_idx / max(1, n_data - 1)
        alpha = int(160 + time_frac * 80)

        draw.line([spline_pts[i], spline_pts[i+1]],
                  fill=color + (alpha,), width=w)

    result = Image.alpha_composite(img.convert('RGBA'), layer)
    return result.convert('RGB')


# ═══════════════════════════════════════════════════════════════════════
# LAYER 3: L-SYSTEM DENDRITES (Task 5)
# ═══════════════════════════════════════════════════════════════════════
DENDRITE_MAX_DEPTH = 5
DENDRITE_LENGTH_DECAY = 0.618
DENDRITE_WIDTH_DECAY = 0.55
DENDRITE_BASE_LENGTH = 160


def render_dendrite_branch(draw, start, angle, length, width, depth,
                           color, alpha, month_idx, branch_id):
    if depth <= 0 or length < 3 or width < 1:
        return

    end_x = start[0] + math.cos(angle) * length
    end_y = start[1] + math.sin(angle) * length

    ctrl_offset = length * 0.3
    rand1 = pseudo_random(month_idx, branch_id, depth * 7 + 1)
    rand2 = pseudo_random(month_idx, branch_id, depth * 7 + 2)
    perp = angle + math.pi / 2

    cp1 = (start[0] + math.cos(angle) * length * 0.33 + math.cos(perp) * ctrl_offset * (rand1 - 0.5),
           start[1] + math.sin(angle) * length * 0.33 + math.sin(perp) * ctrl_offset * (rand1 - 0.5))
    cp2 = (start[0] + math.cos(angle) * length * 0.66 + math.cos(perp) * ctrl_offset * (rand2 - 0.5),
           start[1] + math.sin(angle) * length * 0.66 + math.sin(perp) * ctrl_offset * (rand2 - 0.5))
    end = (end_x, end_y)

    pts = cubic_bezier(start, cp1, cp2, end, num_points=12)
    fade_alpha = max(10, int(alpha * (0.5 + 0.5 * (depth / DENDRITE_MAX_DEPTH))))
    draw_tapered_line(draw, pts, int(width), max(1, int(width * 0.4)), color, fade_alpha)

    n_children = 2 if depth > 2 else 3
    rand3 = pseudo_random(month_idx, branch_id, depth * 7 + 3)
    if rand3 > 0.6 and depth > 1:
        n_children = 3

    spread = math.pi * 0.35
    for c in range(n_children):
        child_angle_offset = (c / max(1, n_children - 1) - 0.5) * spread * 2
        rand_c = pseudo_random(month_idx, branch_id * 10 + c, depth * 7 + 4)
        child_angle = angle + child_angle_offset + (rand_c - 0.5) * 0.3

        child_length = length * DENDRITE_LENGTH_DECAY
        child_width = width * DENDRITE_WIDTH_DECAY

        render_dendrite_branch(
            draw, end, child_angle, child_length, child_width,
            depth - 1, color, fade_alpha, month_idx, branch_id * 10 + c + 1
        )


def render_dendrites(img, rows):
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for i, row in enumerate(rows):
        cx, cy = row["cx"], row["cy"]
        delta = row["delta_norm"]
        vol = row["vol_norm"]

        intensity = abs(delta) * 0.7 + vol * 0.3
        if intensity < 0.03:
            continue

        color = metric_to_color(delta)
        base_length = DENDRITE_BASE_LENGTH * (0.4 + intensity * 2.0)
        base_width = 3 + intensity * 6

        n_roots = max(3, min(10, int(3 + intensity * 8)))
        max_d = max(2, min(DENDRITE_MAX_DEPTH, int(2 + intensity * 4)))

        time_frac = i / max(1, len(rows) - 1)
        base_alpha = int(80 + time_frac * 120 + intensity * 55)
        base_alpha = min(base_alpha, 240)

        for b in range(n_roots):
            rand_a = pseudo_random(i, b, 0)
            if delta > 0:
                # Mostly upward but with spread
                angle = -math.pi * 0.5 + (rand_a - 0.5) * math.pi * 1.2
            elif delta < 0:
                # Mostly downward but with spread
                angle = math.pi * 0.5 + (rand_a - 0.5) * math.pi * 1.2
            else:
                # Neutral: radial
                angle = rand_a * math.pi * 2

            render_dendrite_branch(
                draw, (cx, cy), angle, base_length, base_width,
                max_d, color, base_alpha, i, b
            )

    result = Image.alpha_composite(img.convert('RGBA'), layer)
    return result.convert('RGB')


# ═══════════════════════════════════════════════════════════════════════
# LAYER 4: SYNAPSE FIRING (Task 6)
# ═══════════════════════════════════════════════════════════════════════
SYNAPSE_THRESHOLD = 0.25


def render_synapses(img, rows):
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for i, row in enumerate(rows):
        if abs(row["delta_norm"]) < SYNAPSE_THRESHOLD:
            continue

        cx, cy = row["cx"], row["cy"]
        intensity = abs(row["delta_norm"])

        # Outer halo (drawn first, behind core)
        color = metric_to_color(row["delta_norm"])
        halo_r = int(12 + intensity * 20)
        halo_alpha = int(50 + intensity * 70)
        draw.ellipse([(cx - halo_r, cy - halo_r), (cx + halo_r, cy + halo_r)],
                     fill=color + (halo_alpha,))

        # Mid halo
        mid_r = int(halo_r * 0.6)
        draw.ellipse([(cx - mid_r, cy - mid_r), (cx + mid_r, cy + mid_r)],
                     fill=color + (int(halo_alpha * 1.3),))

        # Core glow — white-hot center
        core_r = int(3 + intensity * 6)
        core_alpha = int(200 + intensity * 55)
        core_alpha = min(core_alpha, 255)
        draw.ellipse([(cx - core_r, cy - core_r), (cx + core_r, cy + core_r)],
                     fill=(255, 220, 200, core_alpha))

        # Micro-connections
        n_rays = int(6 + intensity * 12)
        for r in range(n_rays):
            rand_a = pseudo_random(i, r, 99)
            rand_l = pseudo_random(i, r, 100)
            ang = rand_a * math.pi * 2
            length = 30 + rand_l * 100 * intensity

            end = (cx + math.cos(ang) * length,
                   cy + math.sin(ang) * length)
            mid = (cx + math.cos(ang) * length * 0.5 + (pseudo_random(i, r, 101) - 0.5) * 20,
                   cy + math.sin(ang) * length * 0.5 + (pseudo_random(i, r, 102) - 0.5) * 20)

            pts = cubic_bezier((cx, cy), mid, mid, end, num_points=8)
            ray_alpha = int(60 + intensity * 80)
            draw_tapered_line(draw, pts, max(1, int(2 * intensity)), 1,
                              color, ray_alpha)

    # Blur for soft glow, then composite sharp on top
    layer_blurred = layer.filter(ImageFilter.GaussianBlur(radius=12))
    combined = Image.alpha_composite(layer_blurred, layer)

    result = Image.alpha_composite(img.convert('RGBA'), combined)
    return result.convert('RGB')


# ═══════════════════════════════════════════════════════════════════════
# MAIN RENDER PIPELINE (Task 7)
# ═══════════════════════════════════════════════════════════════════════
def render():
    print(f"Rendering NEURAL SURGE at {WIDTH}x{HEIGHT}...")
    np.random.seed(SEED)

    rows = load_data()
    rows = map_to_canvas(rows)
    print(f"  Data: {len(rows)} months, ${rows[0]['close']:.2f} to ${rows[-1]['close']:.2f}")

    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)

    print("  Layer 1: Circuit grid...")
    render_circuit_grid(img, rows)

    print("  Layer 2: Dendrites...")
    img = render_dendrites(img, rows)

    print("  Layer 3: Trunk nerve...")
    img = render_trunk(img, rows)

    print("  Layer 4: Synapses...")
    img = render_synapses(img, rows)

    os.makedirs(os.path.join(os.path.dirname(__file__) or '.', "output"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__) or '.', "output", "neural_surge.png")
    print("  Post-processing...")
    postprocess_and_save(img, out_path)
    print("Done: Neural Surge")


if __name__ == "__main__":
    render()
