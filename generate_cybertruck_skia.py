#!/usr/bin/env python3
"""
CYBERTRUCK — Tesla Stock Data Art (Variation B)
"Shattered Glass" rendered with Skia for GPU-quality anti-aliasing,
gradients, blur, and shadows.

Requires: python3.12 with skia-python, numpy, scipy
Run: source .venv/bin/activate && python generate_cybertruck_skia.py

Output: output/cybertruck.png (6000x4000)
"""

import json
import math
import os
import numpy as np
import skia
from scipy.spatial import Voronoi

WIDTH = 6000
HEIGHT = 4000
SEED = 42
BG = (8, 8, 12)


# ─── Pseudo-random (deterministic) ──────────────────────────────────
def pseudo_random(a, b, c):
    h = a * 65537 + b * 2147483647 + c * 16807
    h = (((h >> 16) ^ h) * 0x45d9f3b) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * 0x45d9f3b) & 0xFFFFFFFF
    h = ((h >> 16) ^ h) & 0xFFFFFFFF
    return (h % 10000) / 10000.0


# ─── Steel color palette ────────────────────────────────────────────
STEEL_STOPS = [
    (0.0, (25, 27, 32)),
    (0.15, (45, 48, 55)),
    (0.3, (70, 75, 85)),
    (0.5, (110, 115, 130)),
    (0.7, (160, 165, 178)),
    (0.85, (200, 205, 218)),
    (1.0, (240, 243, 255)),
]

def steel_color(price_norm):
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


# ─── Data ────────────────────────────────────────────────────────────
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
    n = len(rows)
    centers = []
    for i, row in enumerate(rows):
        date = row["date"][:7]
        for event_date, label, intensity in IMPACT_EVENTS:
            if date == event_date:
                x = int(80 + (i / max(1, n - 1)) * (WIDTH - 160))
                y = int(80 + (1.0 - row["price_norm"]) * (HEIGHT - 160))
                centers.append((x, y, intensity, label, row))
    return centers


def generate_seed_points(rows, impacts):
    np.random.seed(SEED)
    points = []
    for i in range(300):
        points.append((pseudo_random(i, 0, 500) * WIDTH, pseudo_random(i, 0, 501) * HEIGHT))
    for cx, cy, intensity, _, _ in impacts:
        n_ring = int(40 + intensity * 80)
        for j in range(n_ring):
            angle = pseudo_random(j, int(cx), 600) * math.pi * 2
            r_frac = pseudo_random(j, int(cy), 601) ** 0.5
            radius = r_frac * (300 + intensity * 500)
            px = cx + math.cos(angle) * radius
            py = cy + math.sin(angle) * radius
            if 0 <= px <= WIDTH and 0 <= py <= HEIGHT:
                points.append((px, py))
    n = len(rows)
    for i, row in enumerate(rows):
        x = 80 + (i / max(1, n - 1)) * (WIDTH - 160)
        n_scatter = 2 + int(abs(row["delta_norm"]) * 5 + row["vol_norm"] * 3)
        for s in range(n_scatter):
            y = pseudo_random(i, s, 700) * HEIGHT
            points.append((x + (pseudo_random(i, s, 701) - 0.5) * 40, y))
    return np.array(points)


def cell_to_data(cx, cy, rows):
    n = len(rows)
    time_frac = max(0, min(1, (cx - 80) / (WIDTH - 160)))
    idx = int(time_frac * (n - 1))
    idx = max(0, min(n - 1, idx))
    return rows[idx]


def nearest_impact_dist(cx, cy, impacts):
    min_dist = float('inf')
    near_int = 0
    for ix, iy, ii, _, _ in impacts:
        d = math.sqrt((cx - ix)**2 + (cy - iy)**2)
        if d < min_dist:
            min_dist = d
            near_int = ii
    return min_dist, near_int


# ─── Rendering with Skia ────────────────────────────────────────────
def render():
    print(f"Rendering CYBERTRUCK (Skia) at {WIDTH}x{HEIGHT}...")
    np.random.seed(SEED)

    rows = load_monthly()
    impacts = get_impact_centers(rows)
    seed_pts = generate_seed_points(rows, impacts)
    print(f"  Data: {len(rows)} months, {len(impacts)} impacts, {len(seed_pts)} seeds")

    # Mirror points for edge handling
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

    # Create Skia surface
    surface = skia.Surface(WIDTH, HEIGHT)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color(*BG))

    # ── Draw filled cells with gradient per shard ──
    print("  Drawing cells...")
    for idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        vertices = [vor.vertices[v] for v in region]
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        if max(xs) < -50 or min(xs) > WIDTH + 50:
            continue
        if max(ys) < -50 or min(ys) > HEIGHT + 50:
            continue

        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        row = cell_to_data(cx, cy, rows)
        base_color = steel_color(row["price_norm"])

        min_dist, near_int = nearest_impact_dist(cx, cy, impacts)
        impact_glow = max(0, 1.0 - min_dist / 600) * near_int
        base_color = tuple(min(255, int(c + impact_glow * 40)) for c in base_color)

        # Build path
        path = skia.Path()
        path.moveTo(vertices[0][0], vertices[0][1])
        for v in vertices[1:]:
            path.lineTo(v[0], v[1])
        path.close()

        # Per-shard gradient: simulate directional light (top-left bright)
        min_y = min(ys)
        max_y = max(ys)
        bright = tuple(min(255, c + 20) for c in base_color)
        dark = tuple(max(0, c - 10) for c in base_color)

        alpha = int(190 + row["price_norm"] * 50 + impact_glow * 15)
        alpha = min(alpha, 250)

        shader = skia.GradientShader.MakeLinear(
            points=[skia.Point(cx, min_y), skia.Point(cx, max_y)],
            colors=[skia.Color4f(bright[0]/255, bright[1]/255, bright[2]/255, alpha/255),
                    skia.Color4f(dark[0]/255, dark[1]/255, dark[2]/255, alpha/255)],
        )

        paint = skia.Paint(AntiAlias=True)
        paint.setShader(shader)
        canvas.drawPath(path, paint)

    # ── Draw crack lines ──
    print("  Drawing cracks...")
    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        v1_idx, v2_idx = vor.ridge_vertices[ridge_idx]
        if v1_idx < 0 or v2_idx < 0:
            continue
        v1 = vor.vertices[v1_idx]
        v2 = vor.vertices[v2_idx]
        if (v1[0] < -50 and v2[0] < -50) or (v1[0] > WIDTH + 50 and v2[0] > WIDTH + 50):
            continue
        if (v1[1] < -50 and v2[1] < -50) or (v1[1] > HEIGHT + 50 and v2[1] > HEIGHT + 50):
            continue

        mx, my = (v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2
        min_dist, near_int = nearest_impact_dist(mx, my, impacts)
        impact_factor = max(0, 1.0 - min_dist / 800) * near_int

        width = max(0.5, 0.5 + impact_factor * 3)
        alpha = int(30 + impact_factor * 220)
        alpha = min(alpha, 255)

        if impact_factor > 0.5:
            cr, cg, cb = 220, 235, 255
        elif impact_factor > 0.2:
            cr, cg, cb = 180, 200, 230
        else:
            cr, cg, cb = 55, 65, 85

        paint = skia.Paint(AntiAlias=True)
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(width)
        paint.setColor(skia.ColorSetARGB(alpha, cr, cg, cb))
        canvas.drawLine(v1[0], v1[1], v2[0], v2[1], paint)

    # ── Impact glows (draw with blur) ──
    print("  Drawing impact glows...")
    for icx, icy, intensity, label, irow in impacts:
        # Soft radial glow
        for ring in range(6):
            r = int((6 - ring) * 35 * intensity)
            ring_alpha = int(15 + ring * 12 * intensity)
            paint = skia.Paint(AntiAlias=True)
            paint.setColor(skia.ColorSetARGB(ring_alpha, 200, 220, 255))
            paint.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 20))
            canvas.drawCircle(icx, icy, r, paint)

        # Sharp core
        core_r = 6 + intensity * 10
        paint = skia.Paint(AntiAlias=True)
        paint.setColor(skia.ColorSetARGB(int(200 + intensity * 55), 255, 240, 220))
        canvas.drawCircle(icx, icy, core_r, paint)

        # Radial crack lines from impact
        n_cracks = int(8 + intensity * 16)
        for c in range(n_cracks):
            angle = pseudo_random(int(icx), c, 800) * math.pi * 2
            length = 60 + pseudo_random(int(icy), c, 801) * 200 * intensity

            path = skia.Path()
            path.moveTo(icx, icy)
            px, py = float(icx), float(icy)
            seg_len = length / 3
            for seg in range(3):
                angle += (pseudo_random(int(icx), c * 10 + seg, 802) - 0.5) * 0.6
                px += math.cos(angle) * seg_len
                py += math.sin(angle) * seg_len
                path.lineTo(px, py)

            fade_alpha = int(180 * intensity)
            paint = skia.Paint(AntiAlias=True)
            paint.setStyle(skia.Paint.kStroke_Style)
            paint.setStrokeWidth(1.5 * intensity)
            paint.setColor(skia.ColorSetARGB(fade_alpha, 180, 200, 230))
            canvas.drawPath(path, paint)

        # Orange sparks
        n_sparks = int(3 + intensity * 8)
        for s in range(n_sparks):
            sa = pseudo_random(int(icx), s, 900) * math.pi * 2
            sr = 20 + pseudo_random(int(icy), s, 901) * 80 * intensity
            sx = icx + math.cos(sa) * sr
            sy = icy + math.sin(sa) * sr
            spark_r = 1.5 + intensity * 2.5
            paint = skia.Paint(AntiAlias=True)
            paint.setColor(skia.ColorSetARGB(int(170 + intensity * 80), 255, 140, 40))
            canvas.drawCircle(sx, sy, spark_r, paint)

    # ── Save ──
    print("  Saving...")
    image = surface.makeImageSnapshot()
    os.makedirs(os.path.join(os.path.dirname(__file__) or '.', "output"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__) or '.', "output", "cybertruck.png")
    image.save(out_path, skia.kPNG)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    print("Done: Cybertruck (Skia)")


if __name__ == "__main__":
    render()
