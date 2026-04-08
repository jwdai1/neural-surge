#!/usr/bin/env python3
"""
AUTOPILOT VISION — Tesla Stock Data Art (Variation C)
FSD-style driver POV: stock price as road elevation,
volume as LiDAR point cloud, key events as HUD annotations.

Output: output/autopilot.png (6000x4000)
"""

import json
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import WIDTH, HEIGHT, SEED, pseudo_random

# ─── Autopilot Palette: Tesla FSD UI ─────────────────────────────────
BG = (6, 6, 15)          # deep navy-black
ROAD_COLOR = (35, 35, 50)
LANE_LINE = (80, 200, 255)    # Tesla blue
LANE_LINE_DIM = (40, 100, 180)
LIDAR_BASE = (50, 170, 255)   # blue point cloud
LIDAR_HOT = (120, 255, 210)   # green-cyan for high volume
HUD_COLOR = (80, 200, 255)    # UI elements
HUD_ALERT = (255, 80, 80)     # red alert
HORIZON_GLOW = (15, 30, 60)


# ─── Data ────────────────────────────────────────────────────────────
def load_data():
    path = os.path.join(os.path.dirname(__file__) or '.', "data", "tsla_monthly.json")
    with open(path) as f:
        rows = json.load(f)

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

    # Normalize price to 0..1
    prices = [r["close"] for r in rows]
    min_p, max_p = min(prices), max(prices)
    price_range = max_p - min_p if max_p > min_p else 1.0
    for row in rows:
        row["price_norm"] = (row["close"] - min_p) / price_range

    return rows


# ─── Perspective Projection ──────────────────────────────────────────
# Vanishing point at top-center, road stretches toward it
VP_X = WIDTH * 0.5
VP_Y = HEIGHT * 0.22  # vanishing point — higher up
ROAD_BOTTOM_Y = HEIGHT + 200  # extend past canvas
ROAD_BOTTOM_HALF_W = WIDTH * 0.52


def perspective_map(depth_frac, lateral_offset=0):
    """Map a depth fraction (0=near/bottom, 1=far/vanishing point) to screen coords."""
    # Less aggressive perspective — keep more data visible
    t = depth_frac ** 1.3

    y = ROAD_BOTTOM_Y + (VP_Y - ROAD_BOTTOM_Y) * t
    road_half_w = ROAD_BOTTOM_HALF_W * (1.0 - t * 0.92)
    x = VP_X + lateral_offset * road_half_w

    return x, y


# ─── Layer 1: Sky / Horizon ─────────────────────────────────────────
def render_sky(img):
    draw = ImageDraw.Draw(img)
    horizon_y = int(VP_Y)
    for y in range(horizon_y + 300):
        t = y / (horizon_y + 300)
        # Very soft gradient glow — no sharp line
        dist_from_horizon = abs(y - horizon_y) / 300.0
        glow = max(0, 1.0 - dist_from_horizon) ** 3 * 0.4
        r = int(BG[0] + HORIZON_GLOW[0] * glow)
        g = int(BG[1] + HORIZON_GLOW[1] * glow)
        b = int(BG[2] + HORIZON_GLOW[2] * glow)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))


# ─── Layer 2: Road Surface ──────────────────────────────────────────
def render_road(img, rows):
    draw = ImageDraw.Draw(img)
    n = len(rows)

    # Draw road segments from far (oldest data) to near (newest)
    for i in range(n):
        depth_frac = 1.0 - (i / max(1, n - 1))  # oldest = far, newest = near
        depth_next = 1.0 - ((i + 1) / max(1, n - 1)) if i < n - 1 else 0.0

        # Road quad for this segment
        tl_x, tl_y = perspective_map(depth_frac, -1.0)
        tr_x, tr_y = perspective_map(depth_frac, 1.0)
        bl_x, bl_y = perspective_map(depth_next, -1.0)
        br_x, br_y = perspective_map(depth_next, 1.0)

        # Road color varies with price — darker when low, slightly lighter when high
        p = rows[i]["price_norm"]
        rc = tuple(int(ROAD_COLOR[j] + p * 25) for j in range(3))
        draw.polygon([(tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)], fill=rc)

        # Road surface grid lines — cross-hatching for depth cues
        if i % 3 == 0:
            # Horizontal road line
            near_alpha = int(25 + (1.0 - depth_frac) * 30)
            draw.line([(tl_x, tl_y), (tr_x, tr_y)],
                      fill=(50, 50, 65, near_alpha) if False else (50, 50, 65), width=1)


# ─── Layer 3: Lane Lines (stock price as road elevation feel) ────────
def render_lane_lines(img, rows):
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    n = len(rows)

    for i in range(n - 1):
        depth0 = 1.0 - (i / max(1, n - 1))
        depth1 = 1.0 - ((i + 1) / max(1, n - 1))
        row = rows[i]

        # Lane line brightness based on delta
        intensity = 0.3 + abs(row["delta_norm"]) * 0.7
        alpha = int(80 + intensity * 200 * (1.0 - depth0))  # nearer = brighter
        alpha = min(alpha, 255)

        # Dashed pattern — every other segment
        if i % 3 == 0:
            # Left lane line
            lx0, ly0 = perspective_map(depth0, -0.95)
            lx1, ly1 = perspective_map(depth1, -0.95)
            line_w = max(1, int(3 * (1.0 - depth0) + 1))
            draw.line([(lx0, ly0), (lx1, ly1)], fill=LANE_LINE + (alpha,), width=line_w)

            # Right lane line
            rx0, ry0 = perspective_map(depth0, 0.95)
            rx1, ry1 = perspective_map(depth1, 0.95)
            draw.line([(rx0, ry0), (rx1, ry1)], fill=LANE_LINE + (alpha,), width=line_w)

        # Center dashed line
        if i % 4 < 2:
            cx0, cy0 = perspective_map(depth0, 0.0)
            cx1, cy1 = perspective_map(depth1, 0.0)
            center_alpha = int(alpha * 0.6)
            draw.line([(cx0, cy0), (cx1, cy1)], fill=LANE_LINE_DIM + (center_alpha,), width=1)

        # Additional lane markings for high-volatility periods
        if abs(row["delta_norm"]) > 0.5:
            for offset in [-0.5, 0.5]:
                mx0, my0 = perspective_map(depth0, offset)
                mx1, my1 = perspective_map(depth1, offset)
                draw.line([(mx0, my0), (mx1, my1)],
                          fill=LANE_LINE + (int(alpha * 0.4),), width=1)

    result = Image.alpha_composite(img.convert('RGBA'), layer)
    return result.convert('RGB')


# ─── Layer 4: LiDAR Point Cloud ─────────────────────────────────────
def render_lidar_OLD_UNUSED(img, rows):
    pass

def render_lidar(img, rows):
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    n = len(rows)

    # --- Part A: Terrain walls along road edges ---
    for i, row in enumerate(rows):
        depth_frac = 1.0 - (i / max(1, n - 1))
        vol = row["vol_norm"]
        delta = row["delta_norm"]
        price = row["price_norm"]
        near_t = 1.0 - depth_frac ** 0.5

        max_terrain_h = 500 * near_t
        terrain_h = price * max_terrain_h

        for side in [-1.15, -1.0, 1.0, 1.15]:
            base_x, base_y = perspective_map(depth_frac, side)
            n_vert = max(3, int(terrain_h / 3))
            for v in range(n_vert):
                vy = base_y - (v / max(1, n_vert)) * terrain_h
                rand_j = pseudo_random(i, v, int((side + 2) * 100)) - 0.5
                vx = base_x + rand_j * 10 * near_t
                size = max(1, int(near_t * 3 + pseudo_random(i, v, 350) * 2))
                height_frac = v / max(1, n_vert)
                color_t = height_frac * 0.6 + vol * 0.3 + abs(delta) * 0.1
                cr = int(LIDAR_BASE[0] + (LIDAR_HOT[0] - LIDAR_BASE[0]) * color_t)
                cg = int(LIDAR_BASE[1] + (LIDAR_HOT[1] - LIDAR_BASE[1]) * color_t)
                cb = int(LIDAR_BASE[2] + (LIDAR_HOT[2] - LIDAR_BASE[2]) * min(1, color_t))
                alpha = int(50 + near_t * 160 + height_frac * 30 + vol * 30)
                alpha = min(alpha, 245)
                draw.ellipse([(vx-size, vy-size), (vx+size, vy+size)],
                             fill=(cr, cg, cb, alpha))

    # --- Part B: Scattered atmospheric points ---
    for i, row in enumerate(rows):
        depth_frac = 1.0 - (i / max(1, n - 1))
        vol = row["vol_norm"]
        delta = row["delta_norm"]
        price = row["price_norm"]
        near_t = 1.0 - depth_frac ** 0.5
        n_points = int(25 + vol * 250)
        for p in range(n_points):
            rand_lat = pseudo_random(i, p, 400) * 2 - 1
            rand_vert = pseudo_random(i, p, 401)
            rand_size = pseudo_random(i, p, 402)
            lat = rand_lat * 1.5
            px, py = perspective_map(depth_frac, lat)
            py -= rand_vert * 350 * price * near_t
            size = max(1, int(near_t * (1 + rand_size * 4)))
            color_t = vol * 0.5 + abs(delta) * 0.5
            cr = int(LIDAR_BASE[0] + (LIDAR_HOT[0] - LIDAR_BASE[0]) * color_t)
            cg = int(LIDAR_BASE[1] + (LIDAR_HOT[1] - LIDAR_BASE[1]) * color_t)
            cb = int(LIDAR_BASE[2] + (LIDAR_HOT[2] - LIDAR_BASE[2]) * color_t)
            alpha = int(20 + near_t * 110 + vol * 50)
            alpha = min(alpha, 200)
            draw.ellipse([(px-size, py-size), (px+size, py+size)],
                         fill=(cr, cg, cb, alpha))

    # Slight blur for atmospheric depth
    layer = layer.filter(ImageFilter.GaussianBlur(radius=2))
    result = Image.alpha_composite(img.convert('RGBA'), layer)
    return result.convert('RGB')


# ─── Layer 5: HUD Overlay ───────────────────────────────────────────
def render_hud(img, rows):
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Bounding boxes at key events
    key_events = {
        "2013-05": "First profit",
        "2017-07": "Model 3",
        "2020-01": "Shanghai GF",
        "2020-08": "5:1 split",
        "2020-12": "S&P 500",
        "2021-11": "ATH $414",
        "2022-12": "Low $109",
        "2023-07": "Cybertruck",
        "2024-10": "Robotaxi",
    }

    n = len(rows)
    for i, row in enumerate(rows):
        date = row["date"][:7]
        if date not in key_events:
            continue

        depth_frac = 1.0 - (i / max(1, n - 1))
        cx, cy = perspective_map(depth_frac, 0.0)

        # Bounding box
        box_w = int(180 * (1.0 - depth_frac * 0.6))
        box_h = int(120 * (1.0 - depth_frac * 0.6))
        bx0 = cx - box_w // 2
        by0 = cy - box_h - 20
        bx1 = cx + box_w // 2
        by1 = cy - 20

        is_alert = row["delta_norm"] < -0.3
        box_color = HUD_ALERT if is_alert else HUD_COLOR
        alpha = int(120 + (1.0 - depth_frac) * 100)

        # Box outline
        draw.rectangle([(bx0, by0), (bx1, by1)], outline=box_color + (alpha,), width=2)

        # Corner accents (FSD-style targeting brackets)
        corner_len = min(12, box_w // 4)
        for (cx_, cy_) in [(bx0, by0), (bx1, by0), (bx0, by1), (bx1, by1)]:
            dx = corner_len if cx_ == bx0 else -corner_len
            dy = corner_len if cy_ == by0 else -corner_len
            draw.line([(cx_, cy_), (cx_ + dx, cy_)], fill=box_color + (alpha,), width=2)
            draw.line([(cx_, cy_), (cx_, cy_ + dy)], fill=box_color + (alpha,), width=2)

        # Vertical line from road to box
        draw.line([(cx, cy), (cx, by1)], fill=box_color + (int(alpha * 0.5),), width=1)

        # Small dot on road
        draw.ellipse([(cx-3, cy-3), (cx+3, cy+3)], fill=box_color + (alpha,))

    # Speed / price HUD in bottom-left
    # Simulated digital readout
    latest = rows[-1]
    hud_elements = [
        (80, HEIGHT - 180, f"TSLA"),
        (80, HEIGHT - 140, f"${latest['close']:.0f}"),
        (80, HEIGHT - 100, f"{latest['date']}"),
    ]
    for x, y, text in hud_elements:
        # Simple geometric shapes to suggest text without actual font
        text_w = len(text) * 12
        draw.rectangle([(x, y), (x + text_w, y + 20)],
                       fill=(0, 0, 0, 100), outline=HUD_COLOR + (80,), width=1)
        # Horizontal lines to suggest text
        for j, ch in enumerate(text):
            cx_ = x + 6 + j * 12
            draw.line([(cx_ - 3, y + 7), (cx_ + 3, y + 7)], fill=HUD_COLOR + (120,), width=1)
            draw.line([(cx_ - 2, y + 13), (cx_ + 2, y + 13)], fill=HUD_COLOR + (80,), width=1)

    # Horizontal scan lines across full width (FSD processing feel)
    for y in range(0, HEIGHT, 80):
        scan_alpha = int(8 + 6 * math.sin(y * 0.02))
        draw.line([(0, y), (WIDTH, y)], fill=HUD_COLOR + (scan_alpha,), width=1)

    result = Image.alpha_composite(img.convert('RGBA'), layer)
    return result.convert('RGB')


# ─── Main ────────────────────────────────────────────────────────────
def render():
    print(f"Rendering AUTOPILOT VISION at {WIDTH}x{HEIGHT}...")
    np.random.seed(SEED)

    rows = load_data()
    print(f"  Data: {len(rows)} months, ${rows[0]['close']:.2f} to ${rows[-1]['close']:.2f}")

    img = Image.new('RGB', (WIDTH, HEIGHT), BG)

    print("  Layer 1: Sky / Horizon...")
    render_sky(img)

    print("  Layer 2: Road surface...")
    render_road(img, rows)

    print("  Layer 3: Lane lines...")
    img = render_lane_lines(img, rows)

    print("  Layer 4: LiDAR point cloud...")
    img = render_lidar(img, rows)

    print("  Layer 5: HUD overlay...")
    img = render_hud(img, rows)

    # Post-process: subtle glow only
    from common import apply_glow
    print("  Applying glow...")
    img = apply_glow(img, intensity=0.3, radius=25)

    os.makedirs(os.path.join(os.path.dirname(__file__) or '.', "output"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__) or '.', "output", "autopilot.png")
    img.save(out_path, 'PNG', optimize=True)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    print("Done: Autopilot Vision")


if __name__ == "__main__":
    render()
