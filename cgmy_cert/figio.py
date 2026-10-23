"""Figure output: every figure is written as PNG (dpi 200, the committed raster) and as PDF (vector, TrueType fonts
embedded via pdf.fonttype = 42), same content."""

from __future__ import annotations

from pathlib import Path


def save_fig(fig, png_path, **kw) -> None:
    import matplotlib
    png_path = Path(png_path)
    fig.savefig(png_path, dpi=200, **kw)
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(png_path.with_suffix(".pdf"), **kw)
