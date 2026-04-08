#!/usr/bin/env python3
"""
Neural Surge — Common utilities.
Palette, bezier math, post-processing.
"""

import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SEED = 42
WIDTH = 6000
HEIGHT = 4000
BG_COLOR = (13, 13, 13)  # #0D0D0D


# --- Tesla Red Palette ---------------------------------------------------
COLOR_STOPS = [
    (-1.0, (30,  30,  30)),
    (-0.5, (80,  20,  20)),
    (-0.1, (140, 30,  30)),
    ( 0.0, (204, 0,   0)),
    ( 0.3, (227, 25,  55)),
    ( 0.6, (255, 68,  68)),
    ( 0.8, (255, 102, 0)),
    ( 1.0, (255, 215, 0)),
]

def metric_to_color(norm_val):
    if norm_val <= COLOR_STOPS[0][0]:
        return COLOR_STOPS[0][1]
    if norm_val >= COLOR_STOPS[-1][0]:
        return COLOR_STOPS[-1][1]
    for i in range(len(COLOR_STOPS) - 1):
        t0, c0 = COLOR_STOPS[i]
        t1, c1 = COLOR_STOPS[i + 1]
        if t0 <= norm_val <= t1:
            f = (norm_val - t0) / (t1 - t0)
            f = f * f * (3 - 2 * f)
            return tuple(int(c0[j] + (c1[j] - c0[j]) * f) for j in range(3))
    return COLOR_STOPS[0][1]


# --- Catmull-Rom Spline --------------------------------------------------
def catmull_rom_segment(p0, p1, p2, p3, num_points=20):
    points = []
    for i in range(num_points):
        t = i / num_points
        t2 = t * t
        t3 = t2 * t
        x = 0.5 * ((2 * p1[0]) +
                    (-p0[0] + p2[0]) * t +
                    (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                    (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
        y = 0.5 * ((2 * p1[1]) +
                    (-p0[1] + p2[1]) * t +
                    (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                    (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
        points.append((x, y))
    return points

def catmull_rom_chain(control_points, points_per_segment=20):
    if len(control_points) < 2:
        return list(control_points)
    pts = [control_points[0]] + list(control_points) + [control_points[-1]]
    result = []
    for i in range(1, len(pts) - 2):
        seg = catmull_rom_segment(pts[i-1], pts[i], pts[i+1], pts[i+2], points_per_segment)
        result.extend(seg)
    result.append(control_points[-1])
    return result


# --- Cubic Bezier --------------------------------------------------------
def cubic_bezier(p0, p1, p2, p3, num_points=15):
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        u = 1 - t
        x = u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0]
        y = u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]
        points.append((x, y))
    return points


# --- Tapered Line Drawing ------------------------------------------------
def draw_tapered_line(draw, points, start_width, end_width, color, alpha=255):
    n = len(points)
    if n < 2:
        return
    for i in range(n - 1):
        frac = i / max(1, n - 2)
        w = max(1, int(start_width + (end_width - start_width) * frac))
        c = color + (alpha,)
        draw.line([points[i], points[i+1]], fill=c, width=w)


# --- Post-processing -----------------------------------------------------
def apply_glow(img, intensity=0.4, radius=40):
    glow = img.copy().filter(ImageFilter.GaussianBlur(radius=radius))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    glow_arr = np.array(glow, dtype=np.float32) / 255.0
    blended = 1.0 - (1.0 - img_arr) * (1.0 - glow_arr * intensity)
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8), img.mode)

def apply_vignette(img, strength=0.6):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cx, cy = w / 2, h / 2
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2) / (max(w, h) * 0.5)
    vig = np.clip((dist - 0.5) * 1.0, 0, 1) ** 1.5
    bg = np.array(BG_COLOR, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - vig * strength) + bg[c] * vig * strength
    return Image.fromarray(arr.astype(np.uint8), img.mode)

def postprocess_and_save(img, output_path):
    print("  Applying glow...")
    img = apply_glow(img, intensity=0.4, radius=40)
    if img.mode == 'RGBA':
        final = Image.new('RGB', img.size, BG_COLOR)
        final.paste(img, mask=img.split()[3])
    else:
        final = img
    print("  Applying vignette...")
    final = apply_vignette(final, strength=0.6)
    final.save(output_path, 'PNG', optimize=True)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


# --- Deterministic Pseudo-random -----------------------------------------
def pseudo_random(a, b, c):
    h = a * 65537 + b * 2147483647 + c * 16807
    h = (((h >> 16) ^ h) * 0x45d9f3b) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * 0x45d9f3b) & 0xFFFFFFFF
    h = ((h >> 16) ^ h) & 0xFFFFFFFF
    return (h % 10000) / 10000.0
