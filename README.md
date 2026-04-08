# Neural Surge

**Tesla Stock Data Art — 2010–2025**

![Neural Surge](output/neural_surge.png)

## Concept

A neural network awakens along Tesla's stock price trajectory. Each month of TSLA's 15-year history becomes a node in a living nervous system — dendrites branch outward when the stock surges, contract inward during declines, and fire synaptic bursts at moments of extreme volatility.

The piece draws from the Neuralink aesthetic and circuit board geometry to frame Tesla not as a car company, but as a technological organism that grows, reacts, and adapts.

## How Data Becomes Art

**179 monthly data points** (June 2010 IPO → April 2025) drive every visual element:

| Data Metric | Visual Expression |
|---|---|
| Closing price | Y-position on canvas (log scale) |
| Month-over-month change | Dendrite direction — up for growth, down for decline |
| Volatility (absolute change) | Dendrite length, branching depth, and spread |
| Trading volume | Trunk nerve thickness, branch density |

### Layers

1. **Circuit Grid** — A faint PCB texture anchors the background. Year lines and price levels form a subtle coordinate system. L-shaped trace connectors at quarterly intervals reinforce the technology metaphor.

2. **L-System Dendrites** — At each monthly node, recursive fractal branches grow using an L-system grammar. Branch length decays by the golden ratio (φ = 0.618) at each generation, up to 5 levels deep. Cubic Bézier curves give each branch an organic S-curve. Growth months branch upward; decline months branch downward.

3. **Trunk Nerve** — A Catmull-Rom spline traces the stock price as the system's central axon. Line width pulses with trading volume — thicker during high-activity months. Color shifts from deep red through Tesla Red to gold as momentum builds.

4. **Synapse Firing** — Months with >25% normalized price movement trigger multi-layered glow nodes: a white-hot core surrounded by colored halos, with micro-connection rays radiating outward in Bézier arcs.

### Color Palette — Tesla Red

| Role | Color |
|---|---|
| Background | Charcoal `#0D0D0D` |
| Strong decline | Deep gray `#1E1E1E` |
| Moderate decline | Dark red `#501414` |
| Stable | Tesla Red `#CC0000` |
| Growth | Bright red `#E31937` |
| Surge | Orange `#FF6600` |
| Peak | Gold `#FFD700` |

### Reading the Timeline

- **2010–2019** (left third): Sparse, thin dendrites. The neural network is dormant — a quiet organism accumulating energy during Tesla's long pre-profitability years.
- **2020–2021** (center-right): Explosive branching. Synapses fire in rapid succession. The network reaches maximum activation — Tesla's stock rises ~1,500% in this period.
- **2022** (sharp dip): Dendrites abruptly contract downward. The network dims. A visible moment of signal loss.
- **2023–2025** (far right): Recovery. The network re-illuminates, though with less intensity than the 2020 mania.

## Technical

- **Resolution:** 6000 × 4000 px (print-ready at 300 DPI for ~50 × 33 cm)
- **Stack:** Python 3.8, Pillow (PIL), NumPy
- **Rendering:** Deterministic (seed = 42). Same code always produces the same image.
- **Data:** Yahoo Finance API, monthly OHLCV

```bash
/usr/local/bin/python3 generate_neural_surge.py
```

## License

Data art by [Jaewon Shim](https://github.com/ryle). Stock data from Yahoo Finance.
