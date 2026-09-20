"""Figures, regenerated deterministically from ``results.json`` and the run rows.

Two figures per target language, both *paired* -- every panel shows all three arms on the same
items, because the whole point of the design is the within-item comparison.

``reliability_paired_<lang>.png``
    Confidence against observed accuracy, per bin. The diagonal is perfect calibration. Bin
    counts are annotated because a bin holding four items says nothing, and a reliability
    diagram that hides its counts is a way of overclaiming.

``selective_accuracy_<lang>.png``
    Accuracy against coverage, most-confident-first. This is the operational question: if you
    automate the top X% by confidence, what accuracy do you get? The dotted verticals mark the
    coverage reached at the p_max >= 0.5 and >= 0.9 thresholds.

PNG metadata is pinned so that re-running ``make reproduce`` on unchanged inputs produces
byte-identical files rather than files that differ only by an embedded timestamp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display, and no dependency on one
import matplotlib.pyplot as plt

from . import metrics as M
from .analyse import COVERAGE_THRESHOLDS, Cell
from .questions import ARMS

ARM_STYLE = {
    "A": {"color": "#4C6EF5", "label": "A · EN state, EN instructions", "marker": "o"},
    "B": {"color": "#F59F00", "label": "B · ES state, EN instructions", "marker": "s"},
    "C": {"color": "#12B886", "label": "C · ES state, ES instructions", "marker": "^"},
}

# Reproducible PNGs: no creation timestamp baked into the file.
PNG_METADATA = {"Software": "jev-acento"}


def _panels(n: int) -> tuple[Any, Any]:
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 5.0 * rows), squeeze=False)
    return fig, axes.ravel()


def reliability_figure(
    cells: dict[tuple[str, str, int], Cell], out_path: Path, lang: str = "es"
) -> Path:
    """Reliability diagram per dataset, all arms overlaid."""
    datasets = sorted({c.dataset for c in cells.values()})
    fig, axes = _panels(len(datasets))

    for ax, dataset in zip(axes, datasets, strict=False):
        ax.plot([0, 1], [0, 1], linestyle="--", color="#adb5bd", linewidth=1,
                label="perfect calibration", zorder=1)

        for arm in ARMS:
            cell = cells.get((dataset, arm, 0))
            if cell is None:
                continue
            style = ARM_STYLE[arm]
            bins = M.reliability_bins(cell.p_max, cell.correct)
            xs = [b.mean_confidence for b in bins if b.count > 0]
            ys = [b.accuracy for b in bins if b.count > 0]
            counts = [b.count for b in bins if b.count > 0]
            ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                    label=style["label"], linewidth=1.6, markersize=5, zorder=3)
            for x, y, n in zip(xs, ys, counts, strict=True):
                ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, -12),
                            ha="center", fontsize=6, color=style["color"], alpha=0.8)

        ax.set_title(dataset, fontsize=11)
        ax.set_xlabel("confidence (p_max)")
        ax.set_ylabel("observed accuracy")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, linewidth=0.5)

    for ax in axes[len(datasets):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        f"Calibration by arm ({lang}) — annotations are bin counts", fontsize=13
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, metadata=PNG_METADATA)
    plt.close(fig)
    return out_path


def selective_accuracy_figure(
    cells: dict[tuple[str, str, int], Cell], out_path: Path, lang: str = "es"
) -> Path:
    """Accuracy as a function of coverage, per dataset, all arms overlaid."""
    datasets = sorted({c.dataset for c in cells.values()})
    fig, axes = _panels(len(datasets))

    for ax, dataset in zip(axes, datasets, strict=False):
        for arm in ARMS:
            cell = cells.get((dataset, arm, 0))
            if cell is None:
                continue
            style = ARM_STYLE[arm]
            coverage, risk = M.risk_coverage_curve(cell.p_max, cell.correct)
            ax.plot(coverage, 1.0 - risk, color=style["color"], label=style["label"],
                    linewidth=1.7)

            for t in COVERAGE_THRESHOLDS:
                cov = M.coverage_at_threshold(cell.p_max, cell.correct, t)["coverage"]
                if cov > 0:
                    ax.axvline(cov, color=style["color"], linestyle=":", linewidth=0.8,
                               alpha=0.55)

        ax.set_title(dataset, fontsize=11)
        ax.set_xlabel("coverage (fraction automated, most confident first)")
        ax.set_ylabel("accuracy on covered items")
        ax.set_xlim(0, 1.0)
        ax.grid(alpha=0.25, linewidth=0.5)

    for ax in axes[len(datasets):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        f"Selective accuracy by arm ({lang}) — dotted lines mark p_max ≥ 0.5 and ≥ 0.9",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, metadata=PNG_METADATA)
    plt.close(fig)
    return out_path


def make_all(cells: dict[tuple[str, str, int], Cell], out_dir: Path, lang: str = "es") -> list[Path]:
    return [
        reliability_figure(cells, out_dir / f"reliability_paired_{lang}.png", lang),
        selective_accuracy_figure(cells, out_dir / f"selective_accuracy_{lang}.png", lang),
    ]
