"""Shared chart style for the repo figures, after vercel.com/design.md: Geist type, sentence-case headings, hierarchy
through typography, gridlines quieter than data, direct labels (no legends), one shared label / plot / value lane for a
bar set, zero baselines, a caption that says what to notice. Color carries meaning only: the Kev family is one hue
stepped by size, the hosted reference is a second hue, untrained baselines are neutral.

Colors are Vercel's published tokens (oklch in vercel-brand.css) converted to sRGB hex here.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def oklch_to_hex(L, C, h):
    a, b = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    def srgb(c):
        c = min(1, max(0, c)); return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return "#%02x%02x%02x" % tuple(round(255 * srgb(c)) for c in (r, g, bl))


# vercel-brand.css tokens, light theme
GRAY = {1000: oklch_to_hex(0.205, 0, 0), 900: oklch_to_hex(0.42, 0, 0), 800: oklch_to_hex(0.59, 0, 0), 700: oklch_to_hex(0.65, 0, 0),
        600: oklch_to_hex(0.732, 0, 0), 500: oklch_to_hex(0.836, 0, 0), 400: oklch_to_hex(0.937, 0, 0), 300: oklch_to_hex(0.925, 0, 0),
        200: oklch_to_hex(0.94, 0, 0), 100: oklch_to_hex(0.961, 0, 0)}
BLUE = {1000: oklch_to_hex(0.2667, 0.1099, 254.34), 900: oklch_to_hex(0.5318, 0.2399, 256.99), 700: oklch_to_hex(0.5761, 0.2508, 258.23),
        400: oklch_to_hex(0.9158, 0.0473, 245.116), 100: oklch_to_hex(0.9732, 0.0141, 251.56)}
AMBER = {1000: oklch_to_hex(0.3083, 0.099, 45.48), 900: oklch_to_hex(0.5279, 0.1496, 54.65), 700: oklch_to_hex(0.8187, 0.1969, 76.46), 400: oklch_to_hex(0.9102, 0.1322, 88.25)}
GREEN = {900: oklch_to_hex(0.5175, 0.1453, 147.65), 700: oklch_to_hex(0.6458, 0.1746, 147.27)}
RED = {900: oklch_to_hex(0.5499, 0.232, 25.29), 700: oklch_to_hex(0.6256, 0.2524, 23.03)}

BG = "white"
TEXT, TEXT2, RULE, GRID = GRAY[1000], GRAY[900], GRAY[500], GRAY[400]
# series roles: the Kev family is one hue stepped by size; the hosted reference is a second hue; untrained bases neutral
KEV = {"kev-27b": BLUE[1000], "kev-9b": oklch_to_hex(0.40, 0.19, 257), "kev-8b": BLUE[1000], "kev-4b": BLUE[900], "kev-0.8b": oklch_to_hex(0.72, 0.13, 256), "kev-0.6b": oklch_to_hex(0.72, 0.13, 256),
       "kev-0.5b": oklch_to_hex(0.80, 0.09, 252), "kev-8b-qwen3": BLUE[1000], "kev-4b-qwen3": BLUE[900], "kev-0.6b-qwen3": oklch_to_hex(0.72, 0.13, 256)}
HOLLOW = {"kev-0.5b", "kev-8b-qwen3", "kev-4b-qwen3", "kev-0.6b-qwen3"}   # superseded checkpoints are drawn outlined / hatched so they read as "previous", not as extra sizes
JEV = AMBER[700]
NEUTRAL = GRAY[500]

# dark theme: black background, the gray scale inverted, the Kev hue stepped the other way (largest = brightest) so it stays legible on black
DARK = {"BG": "#000000", "TEXT": oklch_to_hex(0.93, 0, 0), "TEXT2": oklch_to_hex(0.72, 0, 0), "RULE": oklch_to_hex(0.38, 0, 0), "GRID": oklch_to_hex(0.27, 0, 0),
        "NEUTRAL": oklch_to_hex(0.48, 0, 0),
        "KEV": {"kev-27b": oklch_to_hex(0.88, 0.07, 248), "kev-9b": oklch_to_hex(0.76, 0.14, 252), "kev-8b": oklch_to_hex(0.80, 0.12, 250), "kev-4b": oklch_to_hex(0.62, 0.22, 258), "kev-0.8b": oklch_to_hex(0.50, 0.17, 258),
                "kev-0.6b": oklch_to_hex(0.50, 0.17, 258), "kev-0.5b": oklch_to_hex(0.58, 0.10, 256), "kev-8b-qwen3": oklch_to_hex(0.80, 0.12, 250),
                "kev-4b-qwen3": oklch_to_hex(0.62, 0.22, 258), "kev-0.6b-qwen3": oklch_to_hex(0.50, 0.17, 258)}}


def display(name):
    """Hub id -> display name: kev-4b -> Kev-4B, kev-4b-qwen3 -> Kev-4B (Qwen3). Everything else unchanged."""
    import re
    name = re.sub(r"^kev-(\d[\d.]*)b-qwen3", lambda m: f"Kev-{m.group(1)}B (Qwen3)", name)
    return re.sub(r"^kev-(\d[\d.]*)b", lambda m: f"Kev-{m.group(1)}B", name)


def use_style(dark=False):
    """Set rcParams. `dark=True` swaps the module palette in place (KEV is updated, the scalars rebound), so callers that need
    the background or text colors after this call should read them as `chartstyle.BG` etc., not via `from chartstyle import BG`."""
    global BG, TEXT, TEXT2, RULE, GRID, NEUTRAL
    if dark:
        BG, TEXT, TEXT2, RULE, GRID, NEUTRAL = (DARK[k] for k in ("BG", "TEXT", "TEXT2", "RULE", "GRID", "NEUTRAL"))
        KEV.update(DARK["KEV"])
    for f in Path.home().joinpath("Library/Fonts").glob("Geist*.ttf"):
        try: font_manager.fontManager.addfont(str(f))
        except Exception: pass
    for f in Path.home().joinpath("Library/Fonts").glob("GeistMono*.ttf"):
        try: font_manager.fontManager.addfont(str(f))
        except Exception: pass
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "Geist" if "Geist" in have else "DejaVu Sans", "font.weight": "regular",
        "text.color": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT,
        "axes.edgecolor": RULE, "axes.linewidth": 0.8, "axes.grid": False,
        "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 8, "ytick.major.pad": 10,
        "legend.frameon": False,
    })
    return "Geist" in have


def strip(ax, keep=()):
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)
        sp.set_color(RULE)


def hbars(ax, labels, values, colors, *, xlim=(0, 100), fmt="{:.1f}%", height=0.62, emphasize=(), sublabels=None, ticks=None, label_size=12, value_size=13):
    """A bar set with one label lane (left), one plot lane, one value lane (right of each bar, same font). Zero baseline.
    `emphasize` are label indices whose value is set in medium weight."""
    y = list(range(len(labels)))
    ax.barh(y, values, color=colors, height=height, zorder=3)
    for i, v in enumerate(values):
        ax.text(v + (xlim[1] - xlim[0]) * 0.012, i, fmt.format(v), va="center", ha="left", fontsize=value_size, color=TEXT,
                weight="medium" if i in emphasize else "regular")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_size, color=TEXT)
    if sublabels:
        for i, sub in enumerate(sublabels):
            if sub: ax.annotate(sub, xy=(0, i), xycoords=("axes fraction", "data"), xytext=(-8, -13), textcoords="offset points", ha="right", va="center", fontsize=label_size - 2.5, color=TEXT2)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ticks = ticks if ticks is not None else range(int(xlim[0]), int(xlim[1]) + 1, 20)
    ax.set_xticks(list(ticks)); ax.set_xticklabels([f"{t:g}%" if fmt.endswith("%") else f"{t:g}" for t in ticks], fontsize=label_size - 2, color=TEXT2)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0)
    strip(ax)
    return ax


def heading(fig, x, y, text, size=22):
    fig.text(x, y, text, fontsize=size, weight="medium", color=TEXT, ha="left", va="baseline")


def body(fig, x, y, text, size=12, color=None, va="baseline", **kw):
    fig.text(x, y, text, fontsize=size, color=color or TEXT2, ha="left", va=va, linespacing=1.55, **kw)


def rule(fig, x0, x1, y):
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=RULE, lw=0.8))


def stat(fig, x, y, label, value, detail=None, color=TEXT, value_size=34):
    scale = value_size / 34
    body(fig, x, y, label, size=12)
    fig.text(x, y - 0.075 * scale, value, fontsize=value_size, weight="medium", color=color, ha="left", va="baseline")
    if detail: body(fig, x, y - 0.115 * scale, detail, size=11, va="top")
