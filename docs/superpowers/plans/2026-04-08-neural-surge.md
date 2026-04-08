# Neural Surge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a 6000x4000 data art piece visualizing Tesla's monthly stock price (IPO to present) as a neural network with L-system branching dendrites on a circuit board background.

**Architecture:** Data flows from JSON → compute derived metrics (deltas, volatility, volume normalization) → render layers bottom-up (circuit grid → trunk spline → dendrites → synapse glow) → composite and post-process → save PNG.

**Tech Stack:** Python 3.8, Pillow (PIL), NumPy. No external rendering libraries.

---

## File Structure

```
neural-surge/
├── data/
│   └── tsla_monthly.json          # already fetched (179 months)
├── common.py                      # palette, bezier math, post-processing
├── generate_neural_surge.py       # main renderer
├── output/
│   └── neural_surge.png           # final 6000x4000 output
└── docs/superpowers/plans/
    └── 2026-04-08-neural-surge.md # this plan
```

### Task 1: common.py — Palette, Bezier Math, Utilities

**Files:**
- Create: `common.py`

- [ ] **Step 1: Write Tesla Red palette and color mapping**

```python
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SEED = 42
WIDTH = 6000
HEIGHT = 4000
BG_COLOR = (13, 13, 13)  # #0D0D0D

# Tesla Red palette: maps normalized stock metric to color
COLOR_STOPS = [
    (-1.0, (30,  30,  30)),    # deep gray — strong decline
    (-0.5, (80,  20,  20)),    # dark red — moderate decline
    (-0.1, (140, 30,  30)),    # muted red — slight decline
    ( 0.0, (204, 0,   0)),     # tesla red — stable
    ( 0.3, (227, 25,  55)),    # bright red — growth
    ( 0.6, (255, 68,  68)),    # hot red — strong growth
    ( 0.8, (255, 102, 0)),     # orange — surge
    ( 1.0, (255, 215, 0)),     # gold — peak
]

def metric_to_color(norm_val):
    """Map normalized value (-1..+1) to RGB via palette stops."""
    if norm_val <= COLOR_STOPS[0][0]:
        return COLOR_STOPS[0][1]
    if norm_val >= COLOR_STOPS[-1][0]:
        return COLOR_STOPS[-1][1]
    for i in range(len(COLOR_STOPS) - 1):
        t0, c0 = COLOR_STOPS[i]
        t1, c1 = COLOR_STOPS[i + 1]
        if t0 <= norm_val <= t1:
            f = (norm_val - t0) / (t1 - t0)
            f = f * f * (3 - 2 * f)  # smoothstep
            return tuple(int(c0[j] + (c1[j] - c0[j]) * f) for j in range(3))
    return COLOR_STOPS[0][1]
```

- [ ] **Step 2: Write Catmull-Rom spline interpolation**

```python
def catmull_rom_segment(p0, p1, p2, p3, num_points=20):
    """Generate points along a Catmull-Rom spline segment between p1 and p2."""
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
    """Generate a smooth curve through all control points."""
    if len(control_points) < 2:
        return control_points
    # Duplicate first and last for boundary
    pts = [control_points[0]] + control_points + [control_points[-1]]
    result = []
    for i in range(1, len(pts) - 2):
        seg = catmull_rom_segment(pts[i-1], pts[i], pts[i+1], pts[i+2], points_per_segment)
        result.extend(seg)
    result.append(control_points[-1])
    return result
```

- [ ] **Step 3: Write cubic bezier for dendrites**

```python
def cubic_bezier(p0, p1, p2, p3, num_points=15):
    """Evaluate cubic bezier curve. Each p is (x, y)."""
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        u = 1 - t
        x = u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0]
        y = u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]
        points.append((x, y))
    return points
```

- [ ] **Step 4: Write line drawing with variable width (calligraphy stroke)**

```python
def draw_tapered_line(draw, points, start_width, end_width, color, alpha=255):
    """Draw a polyline with linearly tapering width."""
    n = len(points)
    if n < 2:
        return
    for i in range(n - 1):
        frac = i / max(1, n - 2)
        w = max(1, int(start_width + (end_width - start_width) * frac))
        c = color + (alpha,)
        draw.line([points[i], points[i+1]], fill=c, width=w)
```

- [ ] **Step 5: Write glow and vignette post-processing**

```python
def apply_glow(img, intensity=0.4, radius=40):
    """Screen-blend a blurred copy for glow effect."""
    glow = img.copy().filter(ImageFilter.GaussianBlur(radius=radius))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    glow_arr = np.array(glow, dtype=np.float32) / 255.0
    blended = 1.0 - (1.0 - img_arr) * (1.0 - glow_arr * intensity)
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8), img.mode)

def apply_vignette(img, strength=0.6):
    """Radial darkening toward edges."""
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
    """Glow + vignette + save."""
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
    import os
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
```

- [ ] **Step 6: Write deterministic pseudo-random**

```python
def pseudo_random(a, b, c):
    """Deterministic hash-based random in [0, 1)."""
    h = a * 65537 + b * 2147483647 + c * 16807
    h = (((h >> 16) ^ h) * 0x45d9f3b) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * 0x45d9f3b) & 0xFFFFFFFF
    h = ((h >> 16) ^ h) & 0xFFFFFFFF
    return (h % 10000) / 10000.0
```

- [ ] **Step 7: Commit**

```bash
cd /Users/ryle/data-art/neural-surge
git init && git add common.py data/tsla_monthly.json
git commit -m "feat: add common utilities and TSLA monthly data"
```

---

### Task 2: generate_neural_surge.py — Data Loading and Coordinate Mapping

**Files:**
- Create: `generate_neural_surge.py`

- [ ] **Step 1: Write data loading and metric computation**

```python
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

def load_data():
    """Load TSLA monthly data and compute derived metrics."""
    path = os.path.join(os.path.dirname(__file__), "data", "tsla_monthly.json")
    with open(path) as f:
        rows = json.load(f)

    # Compute month-over-month change rate and normalize
    for i, row in enumerate(rows):
        if i == 0:
            row["delta"] = 0.0
            row["delta_raw"] = 0.0
        else:
            prev = rows[i - 1]["close"]
            row["delta_raw"] = (row["close"] - prev) / prev
            row["delta"] = row["delta_raw"]

    # Normalize deltas to -1..+1 using 95th percentile
    abs_deltas = sorted(abs(r["delta_raw"]) for r in rows)
    p95 = abs_deltas[int(len(abs_deltas) * 0.95)] if abs_deltas else 1.0
    scale = p95 if p95 > 0 else 1.0
    for row in rows:
        row["delta_norm"] = max(-1.0, min(1.0, row["delta_raw"] / scale))

    # Normalize volume
    volumes = [r["volume"] for r in rows]
    max_vol = max(volumes)
    for row in rows:
        row["vol_norm"] = row["volume"] / max_vol  # 0..1

    return rows
```

- [ ] **Step 2: Write coordinate mapping (data points to canvas)**

```python
# Layout constants
MARGIN_LEFT = 300
MARGIN_RIGHT = 200
MARGIN_TOP = 400
MARGIN_BOTTOM = 400

def map_to_canvas(rows):
    """Map each month's data to (x, y) canvas coordinates.
    X = time (left to right), Y = stock price (bottom to top, log scale).
    """
    n = len(rows)
    usable_w = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    usable_h = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    # Log scale for price (Tesla went from $1.30 to $404)
    prices = [r["close"] for r in rows]
    log_min = math.log(max(0.1, min(prices)))
    log_max = math.log(max(prices))
    log_range = log_max - log_min if log_max > log_min else 1.0

    for i, row in enumerate(rows):
        # X: evenly spaced
        row["cx"] = MARGIN_LEFT + (i / max(1, n - 1)) * usable_w
        # Y: log-scaled, inverted (high price = top)
        log_p = math.log(max(0.1, row["close"]))
        norm_y = (log_p - log_min) / log_range
        row["cy"] = MARGIN_TOP + (1.0 - norm_y) * usable_h

    return rows
```

- [ ] **Step 3: Commit**

```bash
git add generate_neural_surge.py
git commit -m "feat: add data loading and coordinate mapping"
```

---

### Task 3: generate_neural_surge.py — Circuit Grid Background Layer

**Files:**
- Modify: `generate_neural_surge.py`

- [ ] **Step 1: Write circuit grid renderer**

```python
def render_circuit_grid(img, rows):
    """Draw subtle circuit-board grid as background layer."""
    draw = ImageDraw.Draw(img)

    # Vertical lines: one per year
    years_seen = set()
    for row in rows:
        year = row["date"][:4]
        if year not in years_seen:
            years_seen.add(year)
            x = int(row["cx"])
            # Main vertical line
            draw.line([(x, MARGIN_TOP - 50), (x, HEIGHT - MARGIN_BOTTOM + 50)],
                      fill=(26, 26, 26), width=1)
            # Small node at top
            draw.ellipse([(x-3, MARGIN_TOP-53), (x+3, MARGIN_TOP-47)],
                         fill=(35, 35, 35))

    # Horizontal lines: price levels ($10, $50, $100, $200, $300, $400)
    import math as _math
    prices_grid = [10, 50, 100, 150, 200, 250, 300, 350, 400]
    prices_all = [r["close"] for r in rows]
    log_min = _math.log(max(0.1, min(prices_all)))
    log_max = _math.log(max(prices_all))
    log_range = log_max - log_min
    usable_h = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    for price in prices_grid:
        log_p = _math.log(price)
        if log_min <= log_p <= log_max:
            norm_y = (log_p - log_min) / log_range
            y = int(MARGIN_TOP + (1.0 - norm_y) * usable_h)
            # Dashed line effect: draw short segments
            for x in range(MARGIN_LEFT, WIDTH - MARGIN_RIGHT, 20):
                draw.line([(x, y), (x + 10, y)], fill=(22, 22, 22), width=1)

    # Circuit trace accents: small L-shaped connectors at grid intersections
    for row in rows:
        year = row["date"][:4]
        month = row["date"][5:7]
        if month in ("01", "07"):  # twice per year
            x, y = int(row["cx"]), int(row["cy"])
            # Tiny cross at node position
            draw.line([(x-4, y), (x+4, y)], fill=(30, 30, 30), width=1)
            draw.line([(x, y-4), (x, y+4)], fill=(30, 30, 30), width=1)
```

- [ ] **Step 2: Commit**

```bash
git add generate_neural_surge.py
git commit -m "feat: add circuit grid background layer"
```

---

### Task 4: generate_neural_surge.py — Trunk Nerve (Main Spline)

**Files:**
- Modify: `generate_neural_surge.py`

- [ ] **Step 1: Write trunk nerve renderer with calligraphy stroke**

```python
def render_trunk(img, rows):
    """Draw the main trunk nerve as a Catmull-Rom spline with variable width."""
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Control points from data
    control_pts = [(row["cx"], row["cy"]) for row in rows]

    # Generate smooth spline
    spline_pts = catmull_rom_chain(control_pts, points_per_segment=30)

    # Map each spline point back to nearest data point for color/width
    n_data = len(rows)
    n_spline = len(spline_pts)

    for i in range(len(spline_pts) - 1):
        # Which data point are we near?
        data_idx = int((i / n_spline) * n_data)
        data_idx = min(data_idx, n_data - 1)
        row = rows[data_idx]

        # Color from delta
        color = metric_to_color(row["delta_norm"])

        # Width from volume (2..8px range)
        base_w = 2 + row["vol_norm"] * 6
        # Boost width during high-volatility periods
        volatility_boost = abs(row["delta_norm"]) * 3
        w = max(2, int(base_w + volatility_boost))

        # Alpha: slight glow for recent data
        time_frac = data_idx / max(1, n_data - 1)
        alpha = int(160 + time_frac * 80)

        draw.line([spline_pts[i], spline_pts[i+1]],
                  fill=color + (alpha,), width=w)

    # Composite onto main image
    img.paste(Image.alpha_composite(
        img.convert('RGBA'), layer
    ).convert('RGB'), (0, 0))
    return img
```

- [ ] **Step 2: Commit**

```bash
git add generate_neural_surge.py
git commit -m "feat: add trunk nerve spline renderer"
```

---

### Task 5: generate_neural_surge.py — L-System Dendrites

**Files:**
- Modify: `generate_neural_surge.py`

- [ ] **Step 1: Write L-system dendrite branching**

```python
# L-system parameters
DENDRITE_MAX_DEPTH = 4
DENDRITE_LENGTH_DECAY = 0.618  # golden ratio
DENDRITE_WIDTH_DECAY = 0.6
DENDRITE_BASE_LENGTH = 80      # pixels, scaled by volatility

def render_dendrite_branch(draw, start, angle, length, width, depth,
                           color, alpha, month_idx, branch_id):
    """Recursively draw a dendrite branch using cubic bezier curves.

    Each branch: bezier curve from start in direction 'angle',
    then split into 2-3 child branches at reduced scale.
    """
    if depth <= 0 or length < 3 or width < 1:
        return

    # End point
    end_x = start[0] + math.cos(angle) * length
    end_y = start[1] + math.sin(angle) * length

    # Bezier control points: slight S-curve
    ctrl_offset = length * 0.3
    rand1 = pseudo_random(month_idx, branch_id, depth * 7 + 1)
    rand2 = pseudo_random(month_idx, branch_id, depth * 7 + 2)
    perp = angle + math.pi / 2

    cp1 = (start[0] + math.cos(angle) * length * 0.33 + math.cos(perp) * ctrl_offset * (rand1 - 0.5),
           start[1] + math.sin(angle) * length * 0.33 + math.sin(perp) * ctrl_offset * (rand1 - 0.5))
    cp2 = (start[0] + math.cos(angle) * length * 0.66 + math.cos(perp) * ctrl_offset * (rand2 - 0.5),
           start[1] + math.sin(angle) * length * 0.66 + math.sin(perp) * ctrl_offset * (rand2 - 0.5))
    end = (end_x, end_y)

    # Draw this branch as bezier
    pts = cubic_bezier(start, cp1, cp2, end, num_points=12)
    fade_alpha = max(10, int(alpha * (0.5 + 0.5 * (depth / DENDRITE_MAX_DEPTH))))
    draw_tapered_line(draw, pts, int(width), max(1, int(width * 0.4)), color, fade_alpha)

    # Branch: 2-3 children
    n_children = 2 if depth > 2 else 3
    rand3 = pseudo_random(month_idx, branch_id, depth * 7 + 3)
    if rand3 > 0.6 and depth > 1:
        n_children = 3

    spread = math.pi * 0.35  # base spread angle
    for c in range(n_children):
        child_angle_offset = (c / max(1, n_children - 1) - 0.5) * spread * 2
        # Add randomness
        rand_c = pseudo_random(month_idx, branch_id * 10 + c, depth * 7 + 4)
        child_angle = angle + child_angle_offset + (rand_c - 0.5) * 0.3

        child_length = length * DENDRITE_LENGTH_DECAY
        child_width = width * DENDRITE_WIDTH_DECAY

        render_dendrite_branch(
            draw, end, child_angle, child_length, child_width,
            depth - 1, color, fade_alpha, month_idx, branch_id * 10 + c + 1
        )


def render_dendrites(img, rows):
    """Render L-system dendrites at each month node."""
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for i, row in enumerate(rows):
        cx, cy = row["cx"], row["cy"]
        delta = row["delta_norm"]
        vol = row["vol_norm"]

        # Dendrite intensity scales with |delta| and volume
        intensity = abs(delta) * 0.6 + vol * 0.4
        if intensity < 0.05:
            continue  # skip very quiet months

        color = metric_to_color(delta)
        base_length = DENDRITE_BASE_LENGTH * (0.3 + intensity * 1.5)
        base_width = 2 + intensity * 4

        # Number of root branches: 2-6 based on intensity
        n_roots = max(2, min(6, int(2 + intensity * 5)))

        # Max depth scales with intensity
        max_d = max(1, min(DENDRITE_MAX_DEPTH, int(1 + intensity * 4)))

        # Alpha: earlier = dimmer
        time_frac = i / max(1, len(rows) - 1)
        base_alpha = int(80 + time_frac * 120 + intensity * 55)
        base_alpha = min(base_alpha, 240)

        for b in range(n_roots):
            # Direction: upward bias for positive delta, downward for negative
            rand_a = pseudo_random(i, b, 0)
            if delta > 0:
                # Upward: -pi to 0 range (above the trunk)
                angle = -math.pi * 0.5 + (rand_a - 0.5) * math.pi * 0.8
            else:
                # Downward: 0 to pi range (below the trunk)
                angle = math.pi * 0.5 + (rand_a - 0.5) * math.pi * 0.8

            render_dendrite_branch(
                draw, (cx, cy), angle, base_length, base_width,
                max_d, color, base_alpha, i, b
            )

    img.paste(Image.alpha_composite(
        img.convert('RGBA'), layer
    ).convert('RGB'), (0, 0))
    return img
```

- [ ] **Step 2: Commit**

```bash
git add generate_neural_surge.py
git commit -m "feat: add L-system dendrite renderer"
```

---

### Task 6: generate_neural_surge.py — Synapse Firing Effect

**Files:**
- Modify: `generate_neural_surge.py`

- [ ] **Step 1: Write synapse glow for extreme months**

```python
SYNAPSE_THRESHOLD = 0.4  # normalized delta threshold for firing

def render_synapses(img, rows):
    """Add bright glow nodes at months with extreme price moves."""
    layer = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for i, row in enumerate(rows):
        if abs(row["delta_norm"]) < SYNAPSE_THRESHOLD:
            continue

        cx, cy = row["cx"], row["cy"]
        intensity = abs(row["delta_norm"])

        # Core glow: white-hot center
        core_r = int(4 + intensity * 8)
        core_color = (255, 200, 180, int(200 + intensity * 55))
        draw.ellipse([(cx - core_r, cy - core_r), (cx + core_r, cy + core_r)],
                     fill=core_color)

        # Outer halo: colored by direction
        halo_r = int(core_r * 2.5)
        color = metric_to_color(row["delta_norm"])
        halo_color = color + (int(80 + intensity * 60),)
        draw.ellipse([(cx - halo_r, cy - halo_r), (cx + halo_r, cy + halo_r)],
                     fill=halo_color)

        # Micro-connections: thin bezier lines radiating outward
        n_rays = int(4 + intensity * 8)
        for r in range(n_rays):
            rand_a = pseudo_random(i, r, 99)
            rand_l = pseudo_random(i, r, 100)
            angle = rand_a * math.pi * 2
            length = 20 + rand_l * 60 * intensity

            end = (cx + math.cos(angle) * length,
                   cy + math.sin(angle) * length)
            mid = (cx + math.cos(angle) * length * 0.5 + (pseudo_random(i,r,101) - 0.5) * 20,
                   cy + math.sin(angle) * length * 0.5 + (pseudo_random(i,r,102) - 0.5) * 20)

            pts = cubic_bezier((cx, cy), mid, mid, end, num_points=8)
            ray_alpha = int(60 + intensity * 80)
            draw_tapered_line(draw, pts, max(1, int(2 * intensity)), 1,
                              color, ray_alpha)

    # Blur this layer for soft glow, then composite
    layer_blurred = layer.filter(ImageFilter.GaussianBlur(radius=6))
    combined = Image.alpha_composite(layer_blurred, layer)  # sharp on soft

    img.paste(Image.alpha_composite(
        img.convert('RGBA'), combined
    ).convert('RGB'), (0, 0))
    return img
```

- [ ] **Step 2: Commit**

```bash
git add generate_neural_surge.py
git commit -m "feat: add synapse firing glow effect"
```

---

### Task 7: generate_neural_surge.py — Main Render Pipeline and Output

**Files:**
- Modify: `generate_neural_surge.py`

- [ ] **Step 1: Write main render function compositing all layers**

```python
def render():
    print(f"Rendering NEURAL SURGE at {WIDTH}x{HEIGHT}...")
    np.random.seed(SEED)

    # Load and prepare data
    rows = load_data()
    rows = map_to_canvas(rows)
    print(f"  Data: {len(rows)} months, ${rows[0]['close']:.2f} to ${rows[-1]['close']:.2f}")

    # Create base image
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)

    # Layer 1: Circuit grid (background)
    print("  Layer 1: Circuit grid...")
    render_circuit_grid(img, rows)

    # Layer 2: Dendrites (behind trunk)
    print("  Layer 2: Dendrites...")
    img = render_dendrites(img, rows)

    # Layer 3: Trunk nerve (main spline)
    print("  Layer 3: Trunk nerve...")
    img = render_trunk(img, rows)

    # Layer 4: Synapse firing (on top)
    print("  Layer 4: Synapses...")
    img = render_synapses(img, rows)

    # Post-process and save
    os.makedirs(os.path.join(os.path.dirname(__file__), "output"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "output", "neural_surge.png")
    print("  Post-processing...")
    postprocess_and_save(img, out_path)
    print("Done: Neural Surge")


if __name__ == "__main__":
    render()
```

- [ ] **Step 2: Run and verify output**

```bash
cd /Users/ryle/data-art/neural-surge
/usr/local/bin/python3 generate_neural_surge.py
```

Expected: `output/neural_surge.png` generated at 6000x4000.

- [ ] **Step 3: Visual inspection and parameter tuning**

Open output image, check:
- Circuit grid visible but subtle
- Trunk nerve flows smoothly left to right
- Dendrites branch organically with golden-ratio decay
- 2020-2021 area is the visual climax (most branches, brightest)
- Synapse glow on extreme months
- Tesla Red palette reads correctly

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete Neural Surge renderer with all layers"
```
