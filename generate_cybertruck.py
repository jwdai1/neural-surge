#!/usr/bin/env python3
"""
CYBERTRUCK — Tesla Stock Data Art (Variation B)
"Shattered Glass": Voronoi crack pattern inspired by the Cybertruck
window-breaking incident. Impact points at major stock events.
Each shard colored by price data. Stainless steel palette.

Output: output/cybertruck.png (6000x4000)
"""

import json
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import Voronoi

from common import WIDTH, HEIGHT, SEED, pseudo_random

# ─── Palette ─────────────────────────────────────────────────────────
BG = (8, 8, 12)

STEEL_STOPS = [
    (0.0, (25,  27,  32)),     # darkest steel
    (0.15, (45,  48,  55)),    # dark gunmetal
    (0.3, (70,  75,  85)),     # gunmetal
    (0.5, (110, 115, 130)),    # brushed steel
    (0.7, (160, 165, 178)),    # polished
    (0.85, (200, 205, 218)),   # bright chrome
    (1.0, (240, 243, 255)),    # white chrome
]

CRACK_COLOR = (180, 200, 230)  # bright crack lines
CRACK_COLOR_DIM = (60, 70, 90)
IMPACT_GLOW = (200, 220, 255)
SPARK_ORANGE = (255, 140, 40)


def steel_color(price_norm):
    """Map normalized price (0..1) to steel gradient."""
    if price_norm <= STEEL_STOPS[0][0]:
        return STEEL_STOPS[0][1]
    if price_norm >= STEEL_STOPS[-1][0]:
        return STEEL_STOPS[-1][1]
    for i in range(len(STEEL_STOPS) - 1):
        t0, c0 = STEEL_STOPS[i]
        t1, c1 = STEEL_STOPS[i + 1]
        if t0 <= price_norm <= t1:
            f = (price_norm - t0) / (t1 - t0)
            f = f * f * (3 - 2 * f)
            return tuple(int(c0[j] + (c1[j] - c0[j]) * f) for j in range(3))
    return STEEL_STOPS[0][1]


# ─── Data Loading ────────────────────────────────────────────────────
def load_monthly():
    path = os.path.join(os.path.dirname(__file__) or '.', "data", "tsla_monthly.json")
    with open(path) as f:
        rows = json.load(f)

    prices = [r["close"] for r in rows]
    min_p, max_p = min(prices), max(prices)
    p_range = max_p - min_p if max_p > min_p else 1.0
    for row in rows:
        row["price_norm"] = (row["close"] - min_p) / p_range

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


# ─── Impact Points (key events) ─────────────────────────────────────
IMPACT_EVENTS = [
    ("2013-05", "First profit", 0.6),
    ("2017-07", "Model 3", 0.5),
    ("2020-01", "Shanghai GF", 0.7),
    ("2020-08", "5:1 split", 0.8),
    ("2020-12", "S&P 500", 1.0),
    ("2021-11", "ATH", 0.9),
    ("2022-12", "Crash low", 0.9),
    ("2023-07", "Cybertruck", 0.7),
]


def get_impact_centers(rows):
    """Map impact events to canvas (x, y) coordinates."""
    n = len(rows)
    centers = []
    for i, row in enumerate(rows):
        date = row["date"][:7]
        for event_date, label, intensity in IMPACT_EVENTS:
            if date == event_date:
                # Map time to x, price to y
                x = int(80 + (i / max(1, n - 1)) * (WIDTH - 160))
                y = int(80 + (1.0 - row["price_norm"]) * (HEIGHT - 160))
                centers.append((x, y, intensity, label, row))
    return centers


# ─── Generate Voronoi Seed Points ────────────────────────────────────
def generate_seed_points(rows, impacts):
    """Generate Voronoi seed points: denser near impact centers, sparser elsewhere."""
    np.random.seed(SEED)
    points = []

    # Base grid of points — sparse
    for i in range(300):
        x = pseudo_random(i, 0, 500) * WIDTH
        y = pseudo_random(i, 0, 501) * HEIGHT
        points.append((x, y))

    # Dense points near each impact center — radiating cracks
    for cx, cy, intensity, _, _ in impacts:
        n_ring = int(40 + intensity * 80)
        for j in range(n_ring):
            angle = pseudo_random(j, int(cx), 600) * math.pi * 2
            # Radial distribution: many near center, fewer far
            r_frac = pseudo_random(j, int(cy), 601) ** 0.5  # sqrt for uniform area
            radius = r_frac * (300 + intensity * 500)
            px = cx + math.cos(angle) * radius
            py = cy + math.sin(angle) * radius
            if 0 <= px <= WIDTH and 0 <= py <= HEIGHT:
                points.append((px, py))

    # Points along time axis — one per month, scattered vertically
    n = len(rows)
    for i, row in enumerate(rows):
        x = 80 + (i / max(1, n - 1)) * (WIDTH - 160)
        # Scatter a few points at this x position
        n_scatter = 2 + int(abs(row["delta_norm"]) * 5 + row["vol_norm"] * 3)
        for s in range(n_scatter):
            y = pseudo_random(i, s, 700) * HEIGHT
            points.append((x + (pseudo_random(i, s, 701) - 0.5) * 40, y))

    return np.array(points)


# ─── Map each Voronoi cell to data ──────────────────────────────────
def cell_to_data(cx, cy, rows):
    """Find the nearest month's data for a given cell center."""
    n = len(rows)
    # X position maps to time
    time_frac = max(0, min(1, (cx - 80) / (WIDTH - 160)))
    idx = int(time_frac * (n - 1))
    idx = max(0, min(n - 1, idx))
    return rows[idx]


# ─── Rendering ───────────────────────────────────────────────────────
def render():
    print(f"Rendering CYBERTRUCK (Shattered Glass) at {WIDTH}x{HEIGHT}...")
    np.random.seed(SEED)

    rows = load_monthly()
    impacts = get_impact_centers(rows)
    print(f"  Data: {len(rows)} months, {len(impacts)} impact events")

    # Generate Voronoi
    seed_pts = generate_seed_points(rows, impacts)
    print(f"  Voronoi seeds: {len(seed_pts)}")

    # Add mirror points outside bounds to handle edge cells
    mirrored = []
    for px, py in seed_pts:
        mirrored.append((px, py))
        if px < 200: mirrored.append((px - WIDTH, py))
        if px > WIDTH - 200: mirrored.append((px + WIDTH, py))
        if py < 200: mirrored.append((px, py - HEIGHT))
        if py > HEIGHT - 200: mirrored.append((px, py + HEIGHT))
    mirrored = np.array(mirrored)

    vor = Voronoi(mirrored)
    print("  Voronoi computed")

    img = Image.new('RGB', (WIDTH, HEIGHT), BG)
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # ── Draw filled Voronoi cells ──
    print("  Drawing cells...")
    for idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue

        vertices = [vor.vertices[v] for v in region]
        # Skip cells entirely outside canvas
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        if max(xs) < -50 or min(xs) > WIDTH + 50:
            continue
        if max(ys) < -50 or min(ys) > HEIGHT + 50:
            continue

        # Cell center
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        # Get data for this cell
        row = cell_to_data(cx, cy, rows)
        color = steel_color(row["price_norm"])

        # Brightness modulation by volatility
        vol_mod = 1.0 + row["vol_norm"] * 0.3
        color = tuple(min(255, int(c * vol_mod)) for c in color)

        # Distance to nearest impact — cells near impacts are brighter
        min_impact_dist = float('inf')
        nearest_intensity = 0
        for ix, iy, ii, _, _ in impacts:
            d = math.sqrt((cx - ix)**2 + (cy - iy)**2)
            if d < min_impact_dist:
                min_impact_dist = d
                nearest_intensity = ii

        impact_glow = max(0, 1.0 - min_impact_dist / 600) * nearest_intensity
        color = tuple(min(255, int(c + impact_glow * 40)) for c in color)

        # Alpha: slightly transparent for layering effect
        alpha = int(180 + row["price_norm"] * 60 + impact_glow * 15)
        alpha = min(alpha, 250)

        poly = [(v[0], v[1]) for v in vertices]
        draw.polygon(poly, fill=color + (alpha,))

        # Per-shard lighting: simulate top-left light by brightening
        # upper-left portion. Draw a smaller offset polygon slightly brighter.
        min_y_vert = min(ys)
        max_y_vert = max(ys)
        shard_h = max_y_vert - min_y_vert
        if shard_h > 10:
            # Light gradient: top of shard gets a highlight
            highlight_y = min_y_vert + shard_h * 0.3
            highlight_verts = [(v[0], v[1]) for v in vertices if v[1] < highlight_y]
            if len(highlight_verts) >= 3:
                bright = tuple(min(255, c + 15) for c in color)
                draw.polygon(highlight_verts, fill=bright + (int(alpha * 0.5),))

    # ── Draw crack lines (Voronoi edges) ──
    print("  Drawing cracks...")
    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        v1_idx, v2_idx = vor.ridge_vertices[ridge_idx]
        if v1_idx < 0 or v2_idx < 0:
            continue

        v1 = vor.vertices[v1_idx]
        v2 = vor.vertices[v2_idx]

        # Skip edges outside canvas
        if (v1[0] < -50 and v2[0] < -50) or (v1[0] > WIDTH + 50 and v2[0] > WIDTH + 50):
            continue
        if (v1[1] < -50 and v2[1] < -50) or (v1[1] > HEIGHT + 50 and v2[1] > HEIGHT + 50):
            continue

        # Edge midpoint
        mx, my = (v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2

        # Distance to nearest impact
        min_dist = float('inf')
        near_int = 0
        for ix, iy, ii, _, _ in impacts:
            d = math.sqrt((mx - ix)**2 + (my - iy)**2)
            if d < min_dist:
                min_dist = d
                near_int = ii

        # Cracks are brighter near impacts
        impact_factor = max(0, 1.0 - min_dist / 800) * near_int

        # Line width: thicker near impacts
        width = max(1, int(1 + impact_factor * 3))

        # Color: bright near impact, dim far
        alpha = int(40 + impact_factor * 200)
        alpha = min(alpha, 255)

        if impact_factor > 0.5:
            c = (220, 235, 255, alpha)  # near-white crack
        elif impact_factor > 0.2:
            c = CRACK_COLOR + (alpha,)
        else:
            c = CRACK_COLOR_DIM + (alpha,)

        draw.line([tuple(v1), tuple(v2)], fill=c, width=width)

        # Secondary thin line for edge highlight (rim effect)
        if width > 1 and impact_factor > 0.2:
            highlight_alpha = min(255, int(alpha * 0.4))
            draw.line([tuple(v1), tuple(v2)],
                      fill=(240, 245, 255, highlight_alpha), width=1)

    # Composite cells+cracks
    img = Image.alpha_composite(img.convert('RGBA'), layer).convert('RGB')

    # ── Impact glow centers ──
    print("  Drawing impact glows...")
    glow_layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    for cx, cy, intensity, label, row in impacts:
        # Multi-ring glow
        for ring in range(5):
            r = int((5 - ring) * 30 * intensity)
            ring_alpha = int(20 + ring * 15 * intensity)
            glow_draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                              fill=IMPACT_GLOW + (ring_alpha,))

        # Core
        core_r = int(8 + intensity * 12)
        glow_draw.ellipse([(cx - core_r, cy - core_r), (cx + core_r, cy + core_r)],
                          fill=(255, 240, 220, int(180 + intensity * 75)))

        # Radial crack lines from impact center
        n_cracks = int(8 + intensity * 16)
        for c in range(n_cracks):
            angle = pseudo_random(int(cx), c, 800) * math.pi * 2
            length = 60 + pseudo_random(int(cy), c, 801) * 200 * intensity

            # Angular crack: 2-3 segments with sharp turns
            pts = [(cx, cy)]
            px, py = cx, cy
            seg_len = length / 3
            for seg in range(3):
                # Slight angle deviation at each segment
                angle += (pseudo_random(int(cx), c * 10 + seg, 802) - 0.5) * 0.6
                px += math.cos(angle) * seg_len
                py += math.sin(angle) * seg_len
                pts.append((px, py))

            for s in range(len(pts) - 1):
                fade = 1.0 - s / len(pts)
                alpha = int(160 * intensity * fade)
                w = max(1, int(3 * intensity * fade))
                glow_draw.line([pts[s], pts[s+1]], fill=CRACK_COLOR + (alpha,), width=w)

        # Orange sparks near impact
        n_sparks = int(3 + intensity * 8)
        for s in range(n_sparks):
            sa = pseudo_random(int(cx), s, 900) * math.pi * 2
            sr = 20 + pseudo_random(int(cy), s, 901) * 80 * intensity
            sx = cx + math.cos(sa) * sr
            sy = cy + math.sin(sa) * sr
            spark_r = max(1, int(2 + intensity * 3))
            glow_draw.ellipse([(sx-spark_r, sy-spark_r), (sx+spark_r, sy+spark_r)],
                              fill=SPARK_ORANGE + (int(150 + intensity * 80),))

    # Blur glow, composite sharp on soft
    glow_blurred = glow_layer.filter(ImageFilter.GaussianBlur(radius=15))
    img = Image.alpha_composite(img.convert('RGBA'), glow_blurred).convert('RGB')
    img = Image.alpha_composite(img.convert('RGBA'), glow_layer).convert('RGB')

    # ── Post-process ──
    from common import apply_glow, apply_vignette
    print("  Post-processing...")

    # 1) Edge-detect overlay for rim lighting on cracks
    print("    Adding rim lighting...")
    edges = img.filter(ImageFilter.FIND_EDGES)
    edges_arr = np.array(edges, dtype=np.float32)
    # Tint edges with cool steel-blue
    tinted = np.zeros_like(edges_arr)
    tinted[:, :, 0] = edges_arr[:, :, 0] * 0.6   # less red
    tinted[:, :, 1] = edges_arr[:, :, 1] * 0.75   # some green
    tinted[:, :, 2] = edges_arr[:, :, 2] * 1.0    # full blue
    # Screen blend
    img_arr = np.array(img, dtype=np.float32) / 255.0
    rim_arr = tinted / 255.0 * 0.4  # intensity
    blended = 1.0 - (1.0 - img_arr) * (1.0 - rim_arr)
    img = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8), 'RGB')

    # 3) Glow
    img = apply_glow(img, intensity=0.4, radius=40)

    # 4) Vignette
    img = apply_vignette(img, strength=0.5, center_offset=(0, 0))

    # 5) Micro-blur for smoother polygon edges
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    os.makedirs(os.path.join(os.path.dirname(__file__) or '.', "output"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__) or '.', "output", "cybertruck.png")
    img.save(out_path, 'PNG', optimize=True)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    print("Done: Cybertruck (Shattered Glass)")


if __name__ == "__main__":
    render()
