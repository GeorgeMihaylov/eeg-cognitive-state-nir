"""Tiny dependency-free SVG charts for reproducible analysis reports.

The renderer uses ``currentColor`` plus opacity and line styles instead of a
hard-coded project palette. It is intentionally limited to the plots required
by the statistical report.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _document(title: str, body: str, *, width: int, height: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">\n'
        '<style>text{font-family:system-ui,sans-serif;fill:currentColor}'
        '.axis{stroke:currentColor;stroke-width:1}.grid{stroke:currentColor;'
        'stroke-width:.5;opacity:.18}.mark{stroke:currentColor;fill:currentColor}'
        '.outline{stroke:currentColor;fill:none}</style>\n'
        f'<title>{escape(title)}</title>\n{body}\n</svg>\n'
    )


def _text(x: float, y: float, value: object, *, size: int = 12, anchor: str = "middle", rotate: int | None = None) -> str:
    transform = "" if rotate is None else f' transform="rotate({rotate} {x:.2f} {y:.2f})"'
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'text-anchor="{anchor}"{transform}>{escape(str(value))}</text>'
    )


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if not np.isfinite(value):
        return (start + end) / 2
    if high <= low:
        return (start + end) / 2
    return start + (value - low) / (high - low) * (end - start)


def _axes(
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    width: int,
    height: int,
    x_ticks: Sequence[tuple[float, str]],
    y_ticks: Sequence[tuple[float, str]],
    left: int = 90,
    right: int = 30,
    top: int = 60,
    bottom: int = 105,
) -> list[str]:
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    output = [
        _text(width / 2, 30, title, size=18),
        f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>',
        f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>',
        _text((x0 + x1) / 2, height - 18, xlabel, size=13),
        _text(20, (y0 + y1) / 2, ylabel, size=13, rotate=-90),
    ]
    for x, label in x_ticks:
        output.extend([
            f'<line class="grid" x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y1}"/>',
            _text(x, y1 + 18, label, size=10, rotate=-30),
        ])
    for y, label in y_ticks:
        output.extend([
            f'<line class="grid" x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}"/>',
            _text(x0 - 8, y + 4, label, size=10, anchor="end"),
        ])
    return output


def write_placeholder(path: Path, title: str, message: str) -> None:
    body = "\n".join([
        _text(400, 45, title, size=20),
        _text(400, 190, message, size=14),
    ])
    path.write_text(_document(title, body, width=800, height=360), encoding="utf-8")


def write_scatter(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    horizontal_zero: bool = False,
    width: int = 900,
    height: int = 520,
) -> None:
    all_x = _finite(value for _, xs, _ in series for value in xs)
    all_y = _finite(value for _, _, ys in series for value in ys)
    if all_x.size == 0 or all_y.size == 0:
        write_placeholder(path, title, "No finite values")
        return
    x_low, x_high = float(all_x.min()), float(all_x.max())
    y_low, y_high = float(all_y.min()), float(all_y.max())
    y_margin = max((y_high - y_low) * 0.08, 0.01)
    x_margin = max((x_high - x_low) * 0.03, 0.5)
    x_low, x_high = x_low - x_margin, x_high + x_margin
    y_low, y_high = y_low - y_margin, y_high + y_margin
    if horizontal_zero:
        y_low, y_high = min(y_low, 0.0), max(y_high, 0.0)
    left, right, top, bottom = 90, 30, 60, 105
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    x_tick_values = np.linspace(x_low, x_high, 6)
    y_tick_values = np.linspace(y_low, y_high, 6)
    body = _axes(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        width=width,
        height=height,
        x_ticks=[(_scale(v, x_low, x_high, x0, x1), f"{v:.3g}") for v in x_tick_values],
        y_ticks=[(_scale(v, y_low, y_high, y1, y0), f"{v:.3g}") for v in y_tick_values],
    )
    if horizontal_zero:
        zero_y = _scale(0, y_low, y_high, y1, y0)
        body.append(f'<line class="axis" x1="{x0}" y1="{zero_y:.2f}" x2="{x1}" y2="{zero_y:.2f}" stroke-dasharray="5 4"/>')
    opacities = np.linspace(0.35, 0.9, max(1, len(series)))
    for series_index, (label, xs, ys) in enumerate(series):
        opacity = opacities[series_index]
        for x_value, y_value in zip(xs, ys):
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            x = _scale(float(x_value), x_low, x_high, x0, x1)
            y = _scale(float(y_value), y_low, y_high, y1, y0)
            body.append(f'<circle class="mark" cx="{x:.2f}" cy="{y:.2f}" r="3.4" opacity="{opacity:.2f}"/>')
        body.append(_text(x1 - 6, y0 + 18 + series_index * 17, f"{series_index + 1}: {label}", size=10, anchor="end"))
    path.write_text(_document(title, "\n".join(body), width=width, height=height), encoding="utf-8")


def write_bar(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: Sequence[str],
    values: Sequence[float],
    width: int = 900,
    height: int = 520,
) -> None:
    finite = _finite(values)
    if finite.size == 0:
        write_placeholder(path, title, "No finite values")
        return
    low = min(0.0, float(finite.min()))
    high = max(0.0, float(finite.max()))
    margin = max((high - low) * 0.12, 0.01)
    low, high = low - margin, high + margin
    left, right, top, bottom = 90, 30, 60, 130
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    positions = np.linspace(x0 + 30, x1 - 30, len(labels)) if labels else []
    y_tick_values = np.linspace(low, high, 6)
    body = _axes(
        title=title, xlabel=xlabel, ylabel=ylabel, width=width, height=height,
        x_ticks=[(x, label) for x, label in zip(positions, labels)],
        y_ticks=[(_scale(v, low, high, y1, y0), f"{v:.3g}") for v in y_tick_values],
        bottom=bottom,
    )
    zero = _scale(0.0, low, high, y1, y0)
    bar_width = min(48.0, (x1 - x0) / max(1, len(labels)) * 0.55)
    for index, (x, value) in enumerate(zip(positions, values)):
        if not np.isfinite(value):
            continue
        y = _scale(float(value), low, high, y1, y0)
        top_y, bar_height = min(y, zero), abs(zero - y)
        body.append(
            f'<rect class="mark" x="{x - bar_width / 2:.2f}" y="{top_y:.2f}" '
            f'width="{bar_width:.2f}" height="{bar_height:.2f}" opacity="{0.35 + 0.5 * (index + 1) / max(1, len(labels)):.2f}"/>'
        )
    body.append(f'<line class="axis" x1="{x0}" y1="{zero:.2f}" x2="{x1}" y2="{zero:.2f}"/>')
    path.write_text(_document(title, "\n".join(body), width=width, height=height), encoding="utf-8")


def write_boxplot(
    path: Path,
    *,
    title: str,
    ylabel: str,
    labels: Sequence[str],
    values: Sequence[Sequence[float]],
    width: int = 1100,
    height: int = 560,
) -> None:
    arrays = [_finite(value) for value in values]
    all_values = np.concatenate([array for array in arrays if array.size]) if any(array.size for array in arrays) else np.array([])
    if all_values.size == 0:
        write_placeholder(path, title, "No finite values")
        return
    low, high = float(all_values.min()), float(all_values.max())
    margin = max((high - low) * 0.08, 0.01)
    low, high = low - margin, high + margin
    left, right, top, bottom = 90, 30, 60, 145
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    positions = np.linspace(x0 + 35, x1 - 35, len(labels))
    body = _axes(
        title=title, xlabel="Model and analysis track", ylabel=ylabel,
        width=width, height=height,
        x_ticks=[(x, label) for x, label in zip(positions, labels)],
        y_ticks=[(_scale(v, low, high, y1, y0), f"{v:.3g}") for v in np.linspace(low, high, 6)],
        bottom=bottom,
    )
    for index, (x, array) in enumerate(zip(positions, arrays)):
        if not array.size:
            continue
        q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
        lower, upper = float(array.min()), float(array.max())
        y_q1, y_median, y_q3 = (_scale(v, low, high, y1, y0) for v in (q1, median, q3))
        y_lower, y_upper = (_scale(v, low, high, y1, y0) for v in (lower, upper))
        opacity = 0.35 + 0.5 * (index + 1) / max(1, len(arrays))
        body.extend([
            f'<line class="outline" x1="{x:.2f}" y1="{y_lower:.2f}" x2="{x:.2f}" y2="{y_upper:.2f}"/>',
            f'<rect class="mark" x="{x - 18:.2f}" y="{min(y_q1, y_q3):.2f}" width="36" height="{abs(y_q3 - y_q1):.2f}" opacity="{opacity:.2f}"/>',
            f'<line class="axis" x1="{x - 18:.2f}" y1="{y_median:.2f}" x2="{x + 18:.2f}" y2="{y_median:.2f}"/>',
        ])
    path.write_text(_document(title, "\n".join(body), width=width, height=height), encoding="utf-8")


def write_heatmap(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    matrix: Sequence[Sequence[float]],
    column_labels: Sequence[str],
    width: int = 1000,
    height: int = 760,
) -> None:
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        write_placeholder(path, title, "No finite values")
        return
    rows, columns = values.shape
    left, right, top, bottom = 100, 30, 65, 135
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    cell_width = (x1 - x0) / max(1, columns)
    cell_height = (y1 - y0) / max(1, rows)
    finite = values[np.isfinite(values)]
    max_abs = max(float(np.abs(finite).max()) if finite.size else 1.0, 1e-12)
    body = [
        _text(width / 2, 30, title, size=18),
        _text((x0 + x1) / 2, height - 18, xlabel, size=13),
        _text(20, (y0 + y1) / 2, ylabel, size=13, rotate=-90),
    ]
    for row in range(rows):
        for column in range(columns):
            value = values[row, column]
            opacity = 0.04 if not np.isfinite(value) else 0.12 + 0.78 * abs(value) / max_abs
            dash = ' stroke-dasharray="3 2"' if np.isfinite(value) and value < 0 else ""
            body.append(
                f'<rect class="mark" x="{x0 + column * cell_width:.2f}" '
                f'y="{y0 + row * cell_height:.2f}" width="{cell_width:.2f}" '
                f'height="{cell_height:.2f}" opacity="{opacity:.3f}"{dash}/>'
            )
    for column, label in enumerate(column_labels):
        body.append(_text(x0 + (column + 0.5) * cell_width, y1 + 16, label, size=9, rotate=-35))
    path.write_text(_document(title, "\n".join(body), width=width, height=height), encoding="utf-8")


def write_lines(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    width: int = 900,
    height: int = 520,
) -> None:
    all_x = _finite(value for _, xs, _ in series for value in xs)
    all_y = _finite(value for _, _, ys in series for value in ys)
    if all_x.size == 0 or all_y.size == 0:
        write_placeholder(path, title, "No finite values")
        return
    x_low, x_high = float(all_x.min()), float(all_x.max())
    y_low, y_high = float(all_y.min()), float(all_y.max())
    y_margin = max((y_high - y_low) * 0.12, 0.01)
    y_low, y_high = y_low - y_margin, y_high + y_margin
    left, right, top, bottom = 90, 30, 60, 105
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    body = _axes(
        title=title, xlabel=xlabel, ylabel=ylabel, width=width, height=height,
        x_ticks=[(_scale(v, x_low, x_high, x0, x1), f"{v:.0f}") for v in sorted(set(all_x))],
        y_ticks=[(_scale(v, y_low, y_high, y1, y0), f"{v:.3f}") for v in np.linspace(y_low, y_high, 6)],
    )
    for index, (label, xs, ys) in enumerate(series):
        points = [
            f"{_scale(float(x), x_low, x_high, x0, x1):.2f},{_scale(float(y), y_low, y_high, y1, y0):.2f}"
            for x, y in zip(xs, ys)
            if np.isfinite(x) and np.isfinite(y)
        ]
        opacity = 0.35 + 0.55 * (index + 1) / max(1, len(series))
        dash = "" if index % 2 == 0 else ' stroke-dasharray="6 4"'
        body.append(f'<polyline class="outline" points="{" ".join(points)}" opacity="{opacity:.2f}"{dash}/>' )
        for point in points:
            x, y = point.split(",")
            body.append(f'<circle class="mark" cx="{x}" cy="{y}" r="3.5" opacity="{opacity:.2f}"/>')
        body.append(_text(x1 - 6, y0 + 18 + index * 17, f"{index + 1}: {label}", size=10, anchor="end"))
    path.write_text(_document(title, "\n".join(body), width=width, height=height), encoding="utf-8")
