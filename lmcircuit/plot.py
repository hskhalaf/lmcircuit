"""The figures of evaluation reports, as PDFs:

    lc.plot.frontiers({"OLL BBH": r1, "Decision Index": r2, ...}, "plots/general/frontiers.pdf")   # the paper's Fig. 4
    lc.plot.frontiers_grid({...}, "plots/general/frontiers.pdf")   # each frontier with its calls beside it, 2 per row
    lc.plot.figures(report, "plots/oll_bbh/", "OLL BBH")          # accuracy.pdf, balanced_accuracy.pdf, calibration.pdf
    lc.plot.calibration({"OLL BBH": report, "Decision Index": other}, "plots/general/calibration.pdf")
    lc.plot.brier_calibration({...}, "plots/general/brier_calibration.pdf")   # each Brier score beside its calibration
    lc.plot.runtime({"OLL BBH": r1, "Decision Index": r2, ...}, "plots/general/runtime.pdf")       # fitting time per method

frontiers: one column per dataset. Top, the metric against expected cost: our circuit (the mean over training seeds)
with 95% bars at five costs, every published baseline the report holds (its own line style, three bars each),
majority voting (gold star, with its bar) and each pool judge alone (diamonds; other judges small and grey). Bottom,
the circuit's calls per item to each pool judge it asks (at least MIN_CALLS per item somewhere along its frontier)
along its frontier, cheapest at the bottom, on one count scale for every column. Dataset names on top, one legend between the rows. Bars: 95% intervals from resampling test groups and
training seeds (lmcircuit.evaluation). Cost per item is rescaled by the priciest judge's mean call.
calibration: the reliability of the belief where the circuit stops at budgets 0.01 to 1, and of majority voting's vote
shares.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np

BLUE, GOLD, INK, GREY = "#4A6FC4", "#E8A33D", "#222222", "#C9C9C9"
# pool judges by cost, in thirds: greens (cheapest; dark to light), blues (dark to light), reds (light to dark: the
# priciest the deepest red)
GREENS = ["#24624A", "#44975A", "#8DC79B", "#C2E3C9"]
BLUES = ["#1F3A8A", "#4A6FC4", "#8FB0E8", "#C6D6F5"]
REDS = ["#F2A48C", "#E07A5F", "#C0392B", "#8E2A1E"]
RAMP = ["#A9C8EC", "#7FA7E8", "#4A6FC4", "#2F4FA0", "#14275E"]   # calibration budgets, light (low) to dark (high)
BASELINES = {                                                     # name, colour, dash pattern, in legend order
    "routellm": ("RouteLLM", "#1B998B", (0, (8, 3))),
    "hybridllm": ("HybridLLM", "#D81B60", (0, (1.2, 1.8, 4, 1.8))),
    "frugalgpt": ("FrugalGPT", "#E07B00", (0, (4, 2))),
    "cascaded_selective": ("Cascaded Selective", "#7CB518", (0, (6, 2, 1.5, 2))),
    "cascade_routing": ("Cascade Routing", "#1A1A1A", (0, (1.2, 1.8))),
    "carrot": ("CARROT", "#7B1FA2", (0, (2.5, 2))),
    "consistency_cascade": ("Consistency Cascade", "#00A6D6", (0, (6, 1.5, 1.5, 1.5, 1.5, 1.5))),
    "adaptive_consistency": ("Adaptive-Consistency", "#8D6E63", (0, (1.5, 2))),
}
TICKS = (0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)
LEFT = 0.003                                          # the cost axis starts here, below the cheapest pool judges'
                                                      # mean call (0.004 on OLL BBH)
MIN_CALLS = 0.05                                      # calls per item below which a judge is left out of the calls panel
YLABEL = {"accuracy": "Accuracy (%)", "balanced_accuracy": "Accuracy (%)"}   # balanced accuracy too: the file says which
XLABEL = "Expected cost per item (rescaled)"
OURS = r"LMCircuit $\bf{(Ours)}$"
FIGURE = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
          "font.size": 14, "axes.titlesize": 24, "axes.titleweight": "medium", "axes.labelsize": 22,
          "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 20, "axes.edgecolor": "#D0D0D0",
          "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
          "grid.color": "#E6E6E6", "grid.linewidth": 0.8, "xtick.color": INK, "ytick.color": INK, "xtick.major.size": 0,
          "ytick.major.size": 0, "axes.labelcolor": INK, "text.color": INK, "figure.facecolor": "white",
          "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42}
STYLE = {**{k: FIGURE[k] for k in ("font.family", "font.sans-serif", "axes.edgecolor", "axes.linewidth", "axes.spines.top",
                                   "axes.spines.right", "xtick.color", "ytick.color", "xtick.major.size", "ytick.major.size",
                                   "axes.labelcolor", "text.color", "pdf.fonttype")}}
SIZES = {"Gemma-4-E2B-IT": "2B", "Gemma-4-E4B-IT": "4B", "Gemma-4-26B-A4B-IT": "26B", "Gemma-4-31B-IT": "31B"}
FAMILIES = (("gemma", "Gemma"), ("qwen", "Qwen"), ("granite", "Granite"), ("gpt-oss", "OSS"), ("llama", "Llama"))


def _styled(style):
    def wrap(fn):
        @functools.wraps(fn)
        def drawn(*args, **kwargs):
            import matplotlib
            with matplotlib.rc_context(style):
                return fn(*args, **kwargs)
        return drawn
    return wrap


def pretty(judge: str) -> str:
    """Short judge names: 'Gemma 4B', 'Qwen 32B', 'Granite 3B low', 'OSS 120B high'; others as given (shortened)."""
    base, _, level = judge.partition("::")
    base = base.split("/")[-1]
    family = next((v for k, v in FAMILIES if base.lower().startswith(k)), None)
    if family is None:
        return base if len(base) <= 28 else base[:27] + "…"
    size = SIZES.get(base)
    if size is None:
        m = re.search(r"(\d+(?:\.\d+)?)b(?![a-z0-9])", re.sub(r"-a\d+b", "", base.lower()))
        size = f"{m.group(1)}B" if m else ""
    return f"{family} {size} {level}".replace("  ", " ").strip()


def _window(ymin, ymax, fractions):
    """Round ticks at fixed fractional heights of the axis: the smallest step whose window holds the data, the window
    centred on the data among those."""
    n = len(fractions)
    for step in (0.25, 0.5, 1, 2, 2.5, 3, 4, 5, 10, 15, 20, 25, 50):
        height = step / (fractions[1] - fractions[0])
        fits = [m for m in range(int(np.floor(ymin / step)) - 2, int(np.ceil(ymax / step)) + 2)
                if m * step - fractions[0] * height + 0.02 * height <= ymin
                and ymax <= m * step - fractions[0] * height + 0.98 * height]
        if fits:
            m = min(fits, key=lambda m: abs(m * step - fractions[0] * height + height / 2 - (ymin + ymax) / 2))
            lo = m * step - fractions[0] * height
            return m * step + step * np.arange(n), (lo, lo + height)
    return None


def y_window(ymin: float, ymax: float):
    for fractions in ((0.10, 0.35, 0.60, 0.85), (0.10, 0.50, 0.90)):
        if (out := _window(ymin, ymax, fractions)) is not None:
            return out
    pad = 0.06 * (ymax - ymin + 1e-9)
    return np.linspace(ymin, ymax, 4), (ymin - pad, ymax + pad)


def _cost_colours(members: list[str]) -> dict[str, str]:
    """Each pool judge's colour by cost rank (members cheapest first): the cheapest third green, the middle third blue,
    the priciest third red (the first thirds take the remainder: 7 judges 3/2/2, 10 judges 4/3/3)."""
    n = len(members)
    sizes = [n // 3 + (1 if k < n % 3 else 0) for k in range(3)]
    reds = REDS[max(0, 3 - sizes[2]):][:sizes[2]]
    shades = GREENS[:sizes[0]] + BLUES[:sizes[1]] + reds
    return dict(zip(members, shades))


def _members(report: dict) -> list[str]:
    """The judges in any fold's pool (the ones the circuit asks), cheapest first."""
    refs = report["references"]["judges"]
    return sorted({n for p in report["pools"] for n in p}, key=lambda n: refs[n]["cost"])


def _points(rows: list[dict], metric: str, left: float):
    """A curve's rows by cost; the rows costing at most `left` (the axis' left edge) drawn there as the one reachable
    within it (the largest budget). Returns cost, value (%) and the rows."""
    rows = sorted(rows, key=lambda p: (p["cost"], p["budget"]))
    within = [p for p in rows if p["cost"] <= left]
    pts = ([(left, max(within, key=lambda p: p["budget"]))] if within else []) + [(p["cost"], p) for p in rows
                                                                                  if p["cost"] > left]
    return np.asarray([c for c, _ in pts]), np.asarray([100 * p[metric] for _, p in pts]), [p for _, p in pts]


def _bars(ax, x, y, rows, metric, colour, n, x_min):
    """Vertical 95% bars at n log-spaced costs along a drawn curve, at the rows nearest them."""
    ok = (x >= x_min) & np.asarray(["interval" in r for r in rows])
    if ok.sum() < 2:
        return
    xs, ys, rs = x[ok], y[ok], [r for r, k in zip(rows, ok) if k]
    for target in np.geomspace(xs.min() * 1.15, xs.max() / 1.05, n):
        k = int(np.argmin(np.abs(np.log(xs) - np.log(target))))
        lo, hi = (100 * v for v in rs[k]["interval"][metric])
        ax.errorbar(xs[k], ys[k], yerr=[[max(ys[k] - lo, 0)], [max(hi - ys[k], 0)]], fmt="none", ecolor=colour,
                    elinewidth=1.1, capsize=2.5, alpha=0.9, zorder=4)


def _frontier(ax, report: dict, metric: str, colour: dict, members: list[str], left: float, right: float,
              shown: dict) -> list[float]:
    """One frontier panel; returns the values drawn (for the y window)."""
    ys = []
    for m, (name, col, dash) in BASELINES.items():
        block = report.get("baselines", {}).get(m, {}).get("objectives", {}).get(metric)
        if not block:
            continue
        paid = [r for r in block["curve"] if not r.get("fallback_folds")]
        if not paid:
            continue
        x, y, rows = _points(paid, metric, left)
        ax.plot(x, y, color=col, lw=1.9, ls=dash, zorder=3)
        _bars(ax, x, y, rows, metric, col, 3, max(left, min(r["cost"] for r in paid)))
        shown[m] = (name, col, dash)
        ys += [v for c, v in zip(x, y) if c <= right]
    refs = report["references"]
    for name, j in refs["judges"].items():
        inside = name in members
        ax.scatter(j["cost"], 100 * j[metric], marker="D", s=150 if inside else 22, color=colour.get(name, GREY),
                   edgecolor="white", linewidth=0.8, zorder=5 if inside else 2)
        if inside and left <= j["cost"] <= right:                 # judges off the axis do not set its range
            ys.append(100 * j[metric])
    x, y, rows = _points(report["objectives"][metric]["circuit"]["curve"], metric, left)
    ax.plot(x, y, color=BLUE, lw=2.2, zorder=4, solid_capstyle="round")
    _bars(ax, x, y, rows, metric, BLUE, 5, left)
    ys += list(y)
    mv = refs["majority_vote"]
    if "interval" in mv:
        lo, hi = (100 * v for v in mv["interval"][metric])
        ax.errorbar(mv["cost"], 100 * mv[metric], yerr=[[100 * mv[metric] - lo], [hi - 100 * mv[metric]]], fmt="none",
                    ecolor="#8A5A00", elinewidth=1.0, capsize=2.5, zorder=6)
    ax.scatter(mv["cost"], 100 * mv[metric], marker="*", s=260, color=GOLD, edgecolor="#8A5A00", linewidth=0.8, zorder=6)
    ys.append(100 * mv[metric])
    return ys


def _calls(ax, report: dict, metric: str, colour: dict, members: list[str], left: float, beside: bool = False,
           fontsize: float = 11, one_column: bool = False):
    """The circuit's calls per item to each pool judge along its frontier, stacked cheapest first; returns the costs
    and the stack's height at each."""
    rows = sorted((p for p in report["objectives"][metric]["circuit"]["curve"] if p["cost"] >= left),
                  key=lambda p: (p["cost"], p["budget"]))
    x, bottom = np.asarray([p["cost"] for p in rows]), np.zeros(len(rows))
    for n in members:
        y = np.asarray([p["calls_per_item"].get(n, 0.0) for p in rows])
        if y.max(initial=0.0) < MIN_CALLS:
            continue
        ax.fill_between(x, bottom, bottom + y, color=colour[n], lw=0, antialiased=False, label=pretty(n))
        bottom += y
    ax.grid(False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if abs(v) < 1e-12 else f"{v:g}"))   # no 0: it meets the
    handles, labels = ax.get_legend_handles_labels()            # cost ticks; most expensive on top, as in the stack
    if beside:
        ax.legend(handles[::-1], labels[::-1], fontsize=13, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                  handlelength=1.6)
    else:
        long = max((len(t) for t in labels), default=0) > 16          # long names: one column, or they run off the panel
        ax.legend(handles[::-1], labels[::-1], fontsize=fontsize, loc="upper left", frameon=False, handlelength=1.4,
                  borderaxespad=0.3, ncol=1 if len(handles) <= 5 or long or one_column else 2, columnspacing=1.0)
    return x, bottom


@_styled(FIGURE)
def frontiers(panels: dict[str, dict], path, metric: str = "accuracy", palettes: dict[str, list[str]] | None = None) -> Path:
    """The paper's frontier figure: one column per {title: report}; palettes: {title: colours} for a column's pool
    judges (default: by cost, green for the cheapest to red for the priciest)."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    titles, palettes = list(panels), palettes or {}
    n = len(titles)
    fig = Figure(figsize=(6.4 * n + (4.2 if n == 1 else 0), 11.0))
    axes = np.asarray(fig.subplots(2, n, gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.46, "wspace": 0.20})).reshape(2, n)
    shown, tops, right = {}, [], 0.0
    for report in panels.values():
        refs = report["references"]
        right = max(right, 1.3 * max(max(j["cost"] for j in refs["judges"].values()), refs["majority_vote"]["cost"]))
    ticks = [t for t in TICKS if LEFT <= t <= right]
    for k, title in enumerate(titles):
        report, ax, qx = panels[title], axes[0, k], axes[1, k]
        members = _members(report)
        pal = palettes.get(title)
        colour = {m: pal[i % len(pal)] for i, m in enumerate(members)} if pal else _cost_colours(members)
        ys = _frontier(ax, report, metric, colour, members, LEFT, right, shown)
        yt, yl = y_window(min(ys), max(ys))
        ax.set_yticks(yt)
        ax.set_ylim(*yl)
        tops.append(_calls(qx, report, metric, colour, members, LEFT, beside=n == 1))       # (costs, stack height)
        for a in (ax, qx):
            a.set_xscale("log")
            a.set_xlim(LEFT, right)
            a.set_xticks(ticks, labels=[f"{t:g}" for t in ticks])
            a.xaxis.set_minor_locator(NullLocator())
            a.tick_params(axis="both", labelsize=19)
            a.tick_params(axis="x", pad=8)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    top = 1.08 * max(max(h.max(initial=0.0) for _, h in tops), 0.1)
    for k in range(n):                                             # one count scale for every column ...
        axes[1, k].set_ylim(0, top)
    if n > 1:                                                      # ... with each judge legend (and the first one's
        canvas.draw()                                              # arrow) above the stack beneath it
        for k, (x, h) in enumerate(tops):
            box = axes[1, k].get_legend().get_window_extent(canvas.get_renderer())
            (_, bottom), (reach, _) = axes[1, k].transAxes.inverted().transform(box.get_points())
            reach += 0.3 if k == 0 else 0.0
            edge = LEFT * (right / LEFT) ** min(reach, 1.0)             # that axes fraction as a cost (log axis)
            under = h[x <= edge]
            if len(under):
                top = max(top, under.max() / max(bottom - 0.03, 0.2))
    for k in range(n):
        axes[1, k].set_ylim(0, top)
        if k > 0:
            axes[1, k].set_yticklabels([])
    axes[0, 0].set_ylabel(YLABEL[metric], fontsize=26, labelpad=14)
    axes[1, 0].set_ylabel("Calls per model", fontsize=26, labelpad=14)
    handles = [Line2D([], [], marker="D", color="#7A7A7A", ls="none", markersize=8, label="Individual judge"),
               Line2D([], [], color=BLUE, lw=2.4, label=OURS),
               Line2D([], [], marker="*", color=GOLD, markeredgecolor="#8A5A00", ls="none", markersize=17,
                      label="Majority voting")]
    handles += [Line2D([], [], color=col, lw=1.8, ls=dash, label=name) for name, col, dash in shown.values()]
    if n == 1:                                                     # one column: the legend beside the frontier
        fig.subplots_adjust(top=0.93, bottom=0.11, left=0.12, right=0.6)
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.62, 0.93), frameon=False, fontsize=17,
                   handlelength=2.6)
    else:
        fig.subplots_adjust(top=0.93, bottom=0.11, left=0.06, right=0.99)
        fig.legend(handles=handles, loc="upper center", ncol=-(-len(handles) // 2), frameon=False, fontsize=21, handlelength=2.6,
                   columnspacing=2.0, bbox_to_anchor=(0.5, axes[0, 0].get_position().y0 - 0.048))
    fig.supxlabel(XLABEL, fontsize=27, y=0.015, color=INK, x=0.36 if n == 1 else 0.5)
    for k, title in enumerate(titles):                           # dataset names once, above everything
        bb = axes[0, k].get_position()
        fig.text((bb.x0 + bb.x1) / 2, 0.98, title, ha="center", va="top", fontsize=26, color=INK)
    canvas.draw()                                                  # beside the first calls legend: an arrow
    qx0 = axes[1, 0]
    box = qx0.get_legend().get_window_extent(canvas.get_renderer())
    (lx0, ly0), (lx1, ly1) = qx0.transAxes.inverted().transform(box.get_points())
    xa, trim = lx1 + 0.03, 0.18 * (ly1 - ly0)
    qx0.annotate("", xy=(xa, ly0 + trim), xytext=(xa, ly1 - trim), xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5, mutation_scale=16))
    qx0.text(xa + 0.025, (ly0 + ly1) / 2, "descending\naverage cost", transform=qx0.transAxes, fontsize=17, color=INK,
             ha="left", va="center")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


@_styled(FIGURE)
def frontiers_grid(panels: dict[str, dict], path, metric: str = "accuracy", per_row: int = 2,
                   palettes: dict[str, list[str]] | None = None) -> Path:
    """Every {title: report} as two panels side by side, its frontier and its calls per item, `per_row` datasets per
    row (a shorter last row centred); the marks of frontiers, one legend on top, one cost axis for all."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    titles, palettes = list(panels), palettes or {}
    rows = -(-len(titles) // per_row)
    height = 4.4 * rows + 2.0                                                                  # inches
    fig = Figure(figsize=(8.6 * per_row, height))
    left, right_edge = 0.035, 0.995
    grid = fig.add_gridspec(rows, 2 * per_row, hspace=0.45, wspace=0.17,                       # equal columns
                            top=1 - 2.25 / height, bottom=0.6 / height, left=left, right=right_edge)
    right = max(1.3 * max(max(j["cost"] for j in r["references"]["judges"].values()), r["references"]["majority_vote"]["cost"])
                for r in panels.values())
    ticks = [t for t in TICKS if LEFT <= t <= right]
    shown, drawn = {}, []
    for k, title in enumerate(titles):
        report, (r, c) = panels[title], divmod(k, per_row)
        last = len(titles) - r * per_row                           # datasets on this row, when it is the last
        col = 2 * c + (per_row - last if last < per_row else 0)       # a shorter last row centred
        ax, qx = fig.add_subplot(grid[r, col]), fig.add_subplot(grid[r, col + 1])
        members = _members(report)
        pal = palettes.get(title)
        colour = {m: pal[i % len(pal)] for i, m in enumerate(members)} if pal else _cost_colours(members)
        ys = _frontier(ax, report, metric, colour, members, LEFT, right, shown)
        yt, yl = y_window(min(ys), max(ys))
        ax.set_yticks(yt)
        ax.set_ylim(*yl)
        x, h = _calls(qx, report, metric, colour, members, LEFT, fontsize=10.5, one_column=True)
        qx.set_ylim(0, 1.08 * max(h.max(initial=0.0), 0.1))
        drawn.append((title, ax, qx, x, h))
        for a in (ax, qx):
            a.set_xscale("log")
            a.set_xlim(LEFT, right)
            a.set_xticks(ticks, labels=[f"{t:g}" for t in ticks])
            a.xaxis.set_minor_locator(NullLocator())
            a.tick_params(axis="both", labelsize=12)
        ax.set_ylabel(YLABEL[metric], fontsize=14)
        qx.set_ylabel("Calls per model", fontsize=14)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    for title, ax, qx, x, h in drawn:                              # each judge legend above the stack beneath it
        box = qx.get_legend().get_window_extent(canvas.get_renderer())
        (_, bottom), (reach, _) = qx.transAxes.inverted().transform(box.get_points())
        reach += 0.3 if title == titles[0] else 0.0                # the first panel's arrow and its label
        under = h[x <= LEFT * (right / LEFT) ** min(reach, 1.0)]
        if len(under):
            qx.set_ylim(0, max(qx.get_ylim()[1], under.max() / max(bottom - 0.03, 0.2)))
        a0, a1 = ax.get_position(), qx.get_position()
        fig.text((a0.x0 + a1.x1) / 2, a0.y1 + 0.012, title, ha="center", va="bottom", fontsize=18, color=INK)
        if title == titles[0]:                                      # the first dataset only: how to read the panels
            (lx0, ly0), (lx1, ly1) = qx.transAxes.inverted().transform(
                qx.get_legend().get_window_extent(canvas.get_renderer()).get_points())
            xa, trim = lx1 + 0.03, 0.15 * (ly1 - ly0)
            qx.annotate("", xy=(xa, ly0 + trim), xytext=(xa, ly1 - trim), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3, mutation_scale=14))
            qx.text(xa + 0.025, (ly0 + ly1) / 2, "descending\naverage cost", transform=qx.transAxes, fontsize=11,
                    color=INK, ha="left", va="center")
            ax.annotate("", xy=(0.95, 0.07), xytext=(0.5, 0.07), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3, mutation_scale=14))
            ax.text(0.725, 0.1, r"decreasing $\lambda$", transform=ax.transAxes, fontsize=12, color=INK, ha="center",
                    va="bottom")
        fig.text((a0.x0 + a1.x1) / 2, a0.y0 - 0.42 / height, XLABEL, ha="center", va="top", fontsize=14, color=INK)
    handles = [Line2D([], [], marker="D", color="#7A7A7A", ls="none", markersize=8, label="Individual judge"),
               Line2D([], [], color=BLUE, lw=2.4, label=OURS),
               Line2D([], [], marker="*", color=GOLD, markeredgecolor="#8A5A00", ls="none", markersize=15,
                      label="Majority voting")]
    handles += [Line2D([], [], color=col, lw=1.8, ls=dash, label=name)
                for m, (name, col, dash) in BASELINES.items() if m in shown]
    half = -(-len(handles) // 2)                                    # two rows, each centred on its own
    for k, row in enumerate((handles[:half], handles[half:])):
        fig.legend(handles=row, loc="lower center", ncol=len(row), frameon=False, fontsize=15, handlelength=2.6,
                   columnspacing=1.8, bbox_to_anchor=((left + right_edge) / 2, 1 - (1.1 + 0.42 * k) / height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def _cells(fig, n: int, per_row: int, panels: int = 1, inner: float = 0.2, **grid) -> list[list]:
    """For each of n items, `panels` axes side by side (`inner`: their spacing); `per_row` items per row, a shorter last
    row centred."""
    spec = fig.add_gridspec(-(-n // per_row), 2 * per_row, **grid)
    out = []
    for k in range(n):
        r, c = divmod(k, per_row)
        last = n - r * per_row
        start = 2 * c + (per_row - last if last < per_row else 0)
        cell = spec[r, start:start + 2].subgridspec(1, panels, wspace=inner)
        out.append([fig.add_subplot(cell[0, j]) for j in range(panels)])
    return out


def _reliability(ax, report: dict, objective: str, fontsize: float = 12, aspect: float = 1.0) -> list:
    """The reliability of the belief where the circuit optimising `objective` stops, at each calibration budget (light
    to dark), and of majority voting's vote shares (gold), on a panel `aspect` high per width; returns the legend
    handles."""
    from matplotlib.lines import Line2D
    curves = report["objectives"][objective]["circuit"]["calibration"]
    lines = [(f"budget {b}", c, RAMP[i % len(RAMP)], "o", 4.5) for i, (b, c) in enumerate(curves.items())]
    lines.append(("Majority voting", report["calibration"]["majority_vote"], GOLD, "*", 8))
    ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1.0, ls=(0, (3, 2)), zorder=1)
    for _, c, col, marker, size in lines:
        keep = np.asarray(c["rows"]) > 0
        ax.plot(np.asarray(c["predicted"])[keep], np.asarray(c["observed"])[keep], color=col, lw=1.8, marker=marker,
                markersize=size, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1], labels=["0", "0.5", "1"])
    ax.set_yticks([0.5, 1], labels=["0.5", "1"])
    ax.tick_params(labelsize=fontsize)
    ax.grid(True, color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_box_aspect(aspect)
    return [Line2D([], [], color=col, lw=1.8, marker=m, markersize=size, markeredgecolor="white", label=label)
            for label, _, col, m, size in lines] + [Line2D([], [], color="#AAAAAA", lw=1.0, ls=(0, (3, 2)),
                                                           label="Perfect calibration")]


@_styled(STYLE)
def calibration(panels: dict[str, dict], path, objective: str = "accuracy") -> Path | None:
    """One panel per {title: report} (_reliability). Reports without calibration are skipped."""
    from matplotlib.figure import Figure
    panels = {t: r for t, r in panels.items() if "calibration" in r["objectives"].get(objective, {}).get("circuit", {})}
    if not panels:
        return None
    rows = 1 if len(panels) >= 3 else 2                                     # legend rows
    width, height = max(4.3 * len(panels), 6.0), 4.6 + 0.3 * rows               # inches; below the axes: ticks, label, legend
    fig = Figure(figsize=(width, height))
    axes = np.atleast_1d(fig.subplots(1, len(panels), sharey=True))
    fig.subplots_adjust(left=0.6 / width, right=1 - 0.2 / width, top=1 - 0.45 / height, bottom=(1.0 + 0.3 * rows) / height,
                        wspace=0.2)
    for ax, (title, report) in zip(axes, panels.items()):
        handles = _reliability(ax, report, objective)
        ax.set_title(title, fontsize=14, pad=8)
    axes[0].set_ylabel("Observed frequency", fontsize=15)
    fig.supxlabel("Predicted probability", fontsize=15, y=(0.3 * rows + 0.35) / height, va="center")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=-(-len(handles) // rows),
               frameon=False, fontsize=11.5, handlelength=2.6, columnspacing=1.8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def _brier(ax, report: dict, objective: str, right: float, fontsize: float = 12, aspect: float = 1.0) -> None:
    """The Brier score of the belief where the circuit optimising `objective` stops against expected cost (95% bars at
    five costs), and majority voting's (gold star, with its bar), on a panel `aspect` high per width."""
    from matplotlib.ticker import NullLocator
    x, y, rows = _points(report["objectives"][objective]["circuit"]["curve"], "brier", LEFT)
    ax.plot(x, y, color=INK, lw=2.2, zorder=4, solid_capstyle="round")      # black: the blues are the budgets
    _bars(ax, x, y, rows, "brier", INK, 5, LEFT)
    mv, b = report["references"]["majority_vote"], 100 * report["calibration"]["majority_vote"]["brier"]
    if "brier" in mv.get("interval", {}):
        lo, hi = (100 * v for v in mv["interval"]["brier"])
        ax.errorbar(mv["cost"], b, yerr=[[max(b - lo, 0)], [max(hi - b, 0)]], fmt="none", ecolor="#8A5A00",
                    elinewidth=1.0, capsize=2.5, zorder=6)
    ax.scatter(mv["cost"], b, marker="*", s=260, color=GOLD, edgecolor="#8A5A00", linewidth=0.8, zorder=6)
    yt, yl = y_window(min(min(y), b), max(max(y), b))
    digits = len(f"{(yt[1] - yt[0]) / 100:g}".split(".")[-1])
    ax.set_yticks(yt, labels=[f"{t / 100:.{digits}f}" for t in yt])
    ax.set_ylim(*yl)
    ticks = [t for t in TICKS if LEFT <= t <= right]
    ax.set_xscale("log")
    ax.set_xlim(LEFT, right)
    ax.set_xticks(ticks, labels=[f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_box_aspect(aspect)


@_styled(FIGURE)
def brier_calibration(panels: dict[str, dict], path, objective: str = "accuracy", per_row: int = 2) -> Path:
    """Every {title: report} as two panels side by side, the Brier score against expected cost (_brier) and the
    reliability diagram (_reliability); `per_row` datasets per row (a shorter last row centred), one legend on top."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    rows = -(-len(panels) // per_row)
    width, height = 7.6 * per_row, 3.3 * rows + 1.7                                         # inches
    fig = Figure(figsize=(width, height))
    pairs = _cells(fig, len(panels), per_row, panels=2, inner=0.2, left=0.6 / width, right=1 - 0.15 / width,
                   top=1 - 1.4 / height, bottom=0.6 / height, wspace=0.3, hspace=0.3)
    labelled = len(panels) - per_row - len(panels) % per_row  # x labels from here on: nothing (fully) below them
    right = max(1.3 * max(max(j["cost"] for j in r["references"]["judges"].values()), r["references"]["majority_vote"]["cost"])
                for r in panels.values())
    for k, ((bx, cx), (title, report)) in enumerate(zip(pairs, panels.items())):
        _brier(bx, report, objective, right, aspect=0.8)
        handles = _reliability(cx, report, objective, aspect=0.8)
        bx.set_anchor("E")                                          # the panels close together, the slack between
        cx.set_anchor("W")                                          # datasets
        bx.set_ylabel("Brier score", fontsize=13)
        cx.set_ylabel("Observed frequency", fontsize=13)
        if k >= labelled:
            bx.set_xlabel("Expected cost (rescaled)", fontsize=13)
            cx.set_xlabel("Predicted probability", fontsize=13)
    ours = Line2D([], [], color=INK, lw=2.4, label=OURS)
    rows_ = ([ours, handles[-2], handles[-1]], handles[:-2])        # ours, majority voting, perfect; the budgets
    legends = [fig.legend(handles=row, loc="lower center", ncol=len(row), frameon=False, fontsize=13, handlelength=2.6,
                          columnspacing=1.8, bbox_to_anchor=(0.5, 1 - (0.4 + 0.38 * k) / height))
               for k, row in enumerate(rows_)]
    canvas = FigureCanvasAgg(fig)
    canvas.draw()                                                   # the panels' final positions
    renderer, to_fig = canvas.get_renderer(), fig.transFigure.inverted()
    extent = lambda a: to_fig.transform(a.get_tightbbox(renderer)).T if hasattr(a, "get_tightbbox") \
        else to_fig.transform(a.get_window_extent(renderer)).T     # noqa: E731  ([x0, x1], [y0, y1]) with ticks, labels
    above = min(extent(l)[1][0] for l in legends)                   # each row's titles midway in the space above it
    for r in range(0, len(pairs), per_row):
        row = pairs[r:r + per_row]
        top = max(extent(a)[1][1] for pair in row for a in pair)
        for (bx, cx), title in zip(row, list(panels)[r:r + per_row]):
            fig.text((bx.get_position().x0 + cx.get_position().x1) / 2, (above + top) / 2, title, ha="center",
                     va="center", fontsize=17, color=INK)
        above = min(extent(a)[1][0] for pair in row for a in pair)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def _build_seconds(report: dict) -> float:
    """Our time to build one circuit for one objective, per fold: choosing the pool, fitting the belief (every number of
    types tried) and one validation pass (the target's, when recorded; else the pass of every objective, divided
    among them)."""
    s = {k: v["mean"] for k, v in report["seconds"].items()}
    one = s["validation"] if "validation" in s else s.get("operating_points", 0.0) / max(len(report["objectives"]), 1)
    return s.get("pool", 0.0) + s.get("belief", 0.0) + one


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s" if seconds >= 1 else "< 1 s"
    return f"{seconds / 60:.0f} min" if seconds < 3600 else f"{seconds / 3600:.1f} h"


@_styled(FIGURE)
def runtime(panels: dict[str, dict], path, per_row: int = 4) -> Path:
    """Each method's fitting time per fold (mean over folds, and over seeds for ours), one panel per {title: report},
    `per_row` per row (a shorter last row centred), in minutes with the time printed at each bar's end. Ours:
    _build_seconds; the baselines: their "fit" stage. A method the report
    skipped is marked n/a."""
    from matplotlib.figure import Figure
    titles = list(panels)
    methods = [("circuit", OURS, BLUE)] + [(m, name, col) for m, (name, col, _) in BASELINES.items()]
    rows = -(-len(titles) // per_row)
    height = 4.7 * rows + 0.9                                                                  # inches
    fig = Figure(figsize=(4.6 * per_row + 2.6, height))
    axes = _cells(fig, len(titles), per_row, hspace=0.32, wspace=0.6, top=1 - 0.5 / height, bottom=0.9 / height,
                  left=2.6 / (4.6 * per_row + 2.6), right=0.99)
    y = np.arange(len(methods))[::-1]
    for k, ([ax], title) in enumerate(zip(axes, titles)):
        report, c = panels[title], k % per_row
        seconds = {"circuit": _build_seconds(report)}
        for m, b in report.get("baselines", {}).items():
            if b.get("seconds", {}).get("fit", {}).get("by_fold"):
                seconds[m] = b["seconds"]["fit"]["mean"]
        reach = max(seconds.values()) / 60
        for yi, (m, _, colour) in zip(y, methods):
            if m in seconds:
                ax.barh(yi, seconds[m] / 60, height=0.62, color=colour, edgecolor="white", linewidth=0.8)
                ax.text(seconds[m] / 60 + 0.02 * reach, yi, _duration(seconds[m]), va="center", ha="left", fontsize=17)
            else:
                ax.text(0.02 * reach, yi, "n/a", va="center", ha="left", fontsize=17, color="#9A9A9A")
        ax.set_xlim(0, 1.4 * reach)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=16)
        ax.set_yticks(y, [name for _, name, _ in methods] if c == 0 else [""] * len(y), fontsize=18)
        ax.set_title(title, fontsize=24, pad=10)
        ax.set_xlabel("Fitting time (min)", fontsize=18)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def figures(report: dict, out, title: str = "") -> Path:
    """accuracy.pdf and balanced_accuracy.pdf (the frontier figure, one column titled `title`) and calibration.pdf of
    `report` in the directory `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for metric in report["objectives"]:
        if metric in YLABEL:
            frontiers({title: report}, out / f"{metric}.pdf", metric)
    calibration({title: report}, out / "calibration.pdf")
    return out
