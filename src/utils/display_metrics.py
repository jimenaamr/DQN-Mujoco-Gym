# src/utils/display_metrics.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.widgets import RadioButtons, Slider, TextBox

try:
    # TensorBoard ships this.
    from tensorboard.backend.event_processing.event_multiplexer import (  # type: ignore
        EventMultiplexer,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "tensorboard is required to run this script. "
        "Install it in your environment: pip install tensorboard"
    ) from e


XMode = Literal["step", "wall_time"]

_BASE_FIGSIZE_IN: tuple[float, float] = (14.0, 8.0)
_BASE_TITLE_FS: float = 13.0
_BASE_SECTION_FS: float = 10.0
_BASE_BODY_FS: float = 11.0
_BASE_SMALL_FS: float = 10.0
_BASE_TICK_FS: float = 9.0
_BASE_LEGEND_FS: float = 9.0
_BASE_PLOT_TITLE_FS: float = 12.0

# Subtle spacing control (normalized figure coordinates).
# Typical range: 0.004 (tight) ... 0.02 (airy).
_UI_GAP: float = 0.020

# Optional layout margins (normalized figure coordinates).
_SIDEBAR_RIGHT_PAD: float = 0.030
_BOTTOM_PAD: float = 0.000


@dataclass(frozen=True)
class ViewerConfig:
    """Configuration for the metrics viewer."""

    run_path: Path
    refresh_s: float
    tag: str | None
    xmode: XMode


def _resolve_run_dir(run_path: Path) -> Path:
    """Resolve the run directory.

    Args:
        run_path: Path to the run directory that contains tfevents files.

    Returns:
        Validated run directory path.

    Raises:
        FileNotFoundError: If the run directory does not exist.
    """
    run_dir: Path = run_path.expanduser()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    return run_dir


def _make_multiplexer(run_dir: Path) -> EventMultiplexer:
    """Create an EventMultiplexer that loads events recursively like TensorBoard.

    Args:
        run_dir: Run directory containing event files (possibly in subdirs).

    Returns:
        Configured EventMultiplexer.
    """
    size_guidance: dict[str, int] = {"scalars": 0}
    mux: EventMultiplexer = EventMultiplexer(size_guidance=size_guidance)
    mux.AddRunsFromDirectory(str(run_dir), name=str(run_dir.name))
    return mux


def _list_runs(mux: EventMultiplexer) -> list[str]:
    """List runs discovered by the multiplexer.

    Args:
        mux: EventMultiplexer.

    Returns:
        Sorted list of run names.
    """
    runs: list[str] = list(mux.Runs().keys())
    runs.sort()
    return runs


def _list_scalar_tags(mux: EventMultiplexer, run: str) -> list[str]:
    """List available scalar tags for a given run.

    Args:
        mux: EventMultiplexer.
        run: Run name as returned by mux.

    Returns:
        Sorted list of scalar tags.
    """
    runs: dict[str, Any] = mux.Runs()
    tag_info: dict[str, Any] = dict(runs.get(run, {}))
    scalars: list[str] = list(tag_info.get("scalars", []))
    scalars.sort()
    return scalars


def _get_scalar_series(
    mux: EventMultiplexer,
    run: str,
    tag: str,
    xmode: XMode,
) -> tuple[list[float], list[float]]:
    """Get a scalar time series for a tag.

    Args:
        mux: EventMultiplexer.
        run: Run name.
        tag: Scalar tag.
        xmode: X-axis mode ("step" or "wall_time").

    Returns:
        (xs, values), where xs are steps or wall_time (relative seconds).
    """
    events = mux.Scalars(run, tag)
    if not events:
        return [], []

    steps: np.ndarray = np.array([int(ev.step) for ev in events], dtype=np.int64)
    values: np.ndarray = np.array([float(ev.value) for ev in events], dtype=np.float64)
    wall: np.ndarray = np.array(
        [float(ev.wall_time) for ev in events], dtype=np.float64
    )

    order: np.ndarray = np.argsort(steps, kind="mergesort")
    steps = steps[order]
    values = values[order]
    wall = wall[order]

    # De-duplicate equal steps by keeping the last occurrence (TensorBoard-like).
    if steps.size > 1:
        is_last: np.ndarray = np.ones_like(steps, dtype=bool)
        is_last[:-1] = steps[:-1] != steps[1:]
        steps = steps[is_last]
        values = values[is_last]
        wall = wall[is_last]

    if xmode == "step":
        xs: np.ndarray = steps.astype(np.float64)
    else:
        xs = wall - wall[0]

    return xs.tolist(), values.tolist()


def _ema_smooth(values: list[float], weight: float) -> list[float]:
    """Exponential moving average smoothing (TensorBoard-like feel).

    Args:
        values: Raw scalar values.
        weight: Smoothing weight in [0, 0.999]. Higher => smoother.

    Returns:
        Smoothed values (same length).
    """
    if not values:
        return []
    w: float = float(min(max(weight, 0.0), 0.999))
    out: list[float] = [float(values[0])]
    prev: float = float(values[0])
    for v in values[1:]:
        prev = prev * w + float(v) * (1.0 - w)
        out.append(prev)
    return out


def _style_radiobuttons(
    rb: RadioButtons,
    fontsize: float,
    marker_size: float,
) -> None:
    """Style RadioButtons with best-effort marker sizing across Matplotlib versions.

    This handles the most common internal representations:
      - PathCollection (scatter)  -> ax.collections, set_sizes (points^2)
      - Line2D markers            -> ax.lines, set_markersize (points)
      - Circle patches            -> ax.patches, set_radius (data coords; best-effort)

    Args:
        rb: RadioButtons instance.
        fontsize: Font size for labels.
        marker_size: Marker "diameter" in points (converted as needed).
    """
    for lab in rb.labels:
        lab.set_fontsize(fontsize)

    area_pts2: float = float(marker_size * marker_size)
    for coll in rb.ax.collections:
        if isinstance(coll, PathCollection):
            try:
                coll.set_sizes([area_pts2])
            except Exception:
                continue

    for line in rb.ax.lines:
        try:
            marker = getattr(line, "get_marker", lambda: None)()
            if marker not in (None, "", "None"):
                line.set_markersize(marker_size)
        except Exception:
            continue

    for patch in rb.ax.patches:
        try:
            if hasattr(patch, "set_radius"):
                patch.set_radius(0.03)
        except Exception:
            continue


class MetricViewer:
    """Interactive TensorBoard-like scalar viewer for one run directory."""

    def __init__(self, cfg: ViewerConfig) -> None:
        self.cfg: ViewerConfig = cfg
        self.run_dir: Path = _resolve_run_dir(run_path=self.cfg.run_path)

        self.mux: EventMultiplexer = _make_multiplexer(run_dir=self.run_dir)

        self.runs: list[str] = []
        self.current_run: str | None = None

        self.all_tags: list[str] = []
        self.filtered_tags: list[str] = []
        self.current_tag: str | None = None

        self.fig: Figure
        self.ax_plot: Axes
        self.ax_status: Axes

        self.raw_line: Line2D
        self.smooth_line: Line2D

        self.radio_run: RadioButtons | None = None
        self.radio_tag: RadioButtons | None = None
        self.radio_x: RadioButtons | None = None
        self.filter_box: TextBox | None = None
        self.smooth_slider: Slider | None = None

        self._sidebar_title: Text | None = None
        self._sidebar_subtitle: Text | None = None
        self._xaxis_title: Text | None = None
        self._runs_title: Text | None = None
        self._tags_title: Text | None = None

        self._init_figure()

    def _scale(self) -> float:
        """Compute a UI scale factor relative to a baseline figure size.

        Returns:
            Scale factor (>= ~0.6, <= ~2.0 typically).
        """
        w_in: float
        h_in: float
        w_in, h_in = self.fig.get_size_inches()
        base_w: float = _BASE_FIGSIZE_IN[0]
        base_h: float = _BASE_FIGSIZE_IN[1]
        s: float = float(min(w_in / base_w, h_in / base_h))
        return float(min(max(s, 0.60), 2.00))

    def _fs(self, base: float) -> float:
        """Scale a base font size.

        Args:
            base: Baseline font size.

        Returns:
            Scaled font size.
        """
        return float(base * self._scale())

    def _gap(self) -> float:
        """Return the current layout gap in figure coordinates.

        Returns:
            Gap in normalized figure coordinates.
        """
        return float(_UI_GAP * self._scale())

    def _right_panel_rects(self) -> dict[str, float]:
        """Compute right-panel rectangles with padding driven by _UI_GAP.

        Returns:
            Dictionary with keys:
                plot_left, plot_bottom, plot_width, plot_height,
                status_left, status_bottom, status_width, status_height,
                slider_left, slider_bottom, slider_width, slider_height
        """
        g: float = self._gap()

        base_left: float = 0.33 + _SIDEBAR_RIGHT_PAD
        base_right: float = 0.98
        base_plot_bottom: float = 0.14 + _BOTTOM_PAD
        base_plot_top: float = 0.94
        base_status_bottom: float = 0.95
        base_status_h: float = 0.04
        base_slider_bottom: float = 0.06 + _BOTTOM_PAD
        base_slider_h: float = 0.04

        left: float = base_left + g
        right: float = base_right - g
        width: float = max(0.10, right - left)

        status_bottom: float = max(0.0, base_status_bottom - g)
        status_h: float = base_status_h
        plot_top: float = max(0.20, base_plot_top - 2.0 * g)
        plot_bottom: float = min(plot_top - 0.05, base_plot_bottom + g)
        plot_h: float = max(0.10, plot_top - plot_bottom)

        slider_bottom: float = base_slider_bottom + 0.5 * g
        slider_h: float = base_slider_h

        return {
            "plot_left": left,
            "plot_bottom": plot_bottom,
            "plot_width": width,
            "plot_height": plot_h,
            "status_left": left,
            "status_bottom": status_bottom,
            "status_width": width,
            "status_height": status_h,
            "slider_left": left,
            "slider_bottom": slider_bottom,
            "slider_width": width,
            "slider_height": slider_h,
        }

    def _apply_layout(self) -> None:
        """Apply layout positions so _UI_GAP affects right-panel paddings too."""
        rects: dict[str, float] = self._right_panel_rects()

        self.ax_plot.set_position([
            rects["plot_left"],
            rects["plot_bottom"],
            rects["plot_width"],
            rects["plot_height"],
        ])
        self.ax_status.set_position([
            rects["status_left"],
            rects["status_bottom"],
            rects["status_width"],
            rects["status_height"],
        ])
        if self.smooth_slider is not None:
            self.smooth_slider.ax.set_position([
                rects["slider_left"],
                rects["slider_bottom"],
                rects["slider_width"],
                rects["slider_height"],
            ])

        self.fig.canvas.draw_idle()

    def _apply_ui_scale(self) -> None:
        """Apply responsive font sizes to the whole UI."""
        title_fs: float = self._fs(_BASE_TITLE_FS)
        section_fs: float = self._fs(_BASE_SECTION_FS)
        body_fs: float = self._fs(_BASE_BODY_FS)
        small_fs: float = self._fs(_BASE_SMALL_FS)
        tick_fs: float = self._fs(_BASE_TICK_FS)
        legend_fs: float = self._fs(_BASE_LEGEND_FS)

        if self._sidebar_title is not None:
            self._sidebar_title.set_fontsize(title_fs)
        if self._sidebar_subtitle is not None:
            self._sidebar_subtitle.set_fontsize(body_fs)
        if self._xaxis_title is not None:
            self._xaxis_title.set_fontsize(section_fs)
        if self._runs_title is not None:
            self._runs_title.set_fontsize(section_fs)
        if self._tags_title is not None:
            self._tags_title.set_fontsize(section_fs)

        if self.filter_box is not None:
            self.filter_box.label.set_fontsize(body_fs)
            self.filter_box.text_disp.set_fontsize(body_fs)

        if self.radio_x is not None:
            _style_radiobuttons(
                self.radio_x,
                fontsize=body_fs,
                marker_size=10.0 * self._scale(),
            )

        if self.radio_run is not None:
            _style_radiobuttons(
                self.radio_run,
                fontsize=small_fs,
                marker_size=9.0 * self._scale(),
            )

        if self.radio_tag is not None:
            _style_radiobuttons(
                self.radio_tag,
                fontsize=small_fs,
                marker_size=9.0 * self._scale(),
            )

        if self.smooth_slider is not None:
            self.smooth_slider.label.set_fontsize(body_fs)
            self.smooth_slider.valtext.set_fontsize(body_fs)

        self.ax_plot.xaxis.label.set_fontsize(body_fs)
        self.ax_plot.yaxis.label.set_fontsize(body_fs)
        self.ax_plot.tick_params(axis="both", labelsize=tick_fs)

        leg = self.ax_plot.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(legend_fs)

        self.fig.canvas.draw_idle()

    def _on_resize(self, _event: Any) -> None:
        """Handle window resize events."""
        self._apply_layout()
        self._apply_ui_scale()
        self._force_redraw()

    def _init_figure(self) -> None:
        """Initialize matplotlib figure and UI widgets."""
        self.fig = plt.figure(figsize=_BASE_FIGSIZE_IN)
        self.fig.canvas.manager.set_window_title("TensorBoard Scalars (viewer)")  # type: ignore[attr-defined]
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

        rects: dict[str, float] = self._right_panel_rects()

        self.ax_plot = self.fig.add_axes([
            rects["plot_left"],
            rects["plot_bottom"],
            rects["plot_width"],
            rects["plot_height"],
        ])
        self.ax_plot.set_xlabel("step")
        self.ax_plot.set_ylabel("value")
        self.ax_plot.grid(True, which="both", linewidth=0.6)
        self.ax_plot.margins(x=0.02, y=0.08)

        (self.raw_line,) = self.ax_plot.plot(
            [],
            [],
            linewidth=1.0,
            alpha=0.30,
            label="raw",
        )
        (self.smooth_line,) = self.ax_plot.plot(
            [],
            [],
            linewidth=2.0,
            alpha=0.95,
            label="smoothed",
        )
        self.ax_plot.legend(loc="upper right", frameon=True)

        self.ax_status = self.fig.add_axes([
            rects["status_left"],
            rects["status_bottom"],
            rects["status_width"],
            rects["status_height"],
        ])
        self.ax_status.set_axis_off()

        self._reload_all(initial=True)
        self._setup_widgets()
        self._apply_layout()
        self._apply_ui_scale()
        self._force_redraw()

    def _setup_widgets(self) -> None:
        """Create UI widgets (filter, xmode, run selector, tag selector, smoothing)."""
        left_x: float = 0.03
        left_w: float = 0.29

        gap: float = self._gap()
        y: float = 0.98

        title_h: float = 0.055
        ax_title: Axes = self.fig.add_axes([left_x, y - title_h, left_w, title_h])
        ax_title.set_axis_off()
        self._sidebar_title = ax_title.text(
            0.0,
            0.62,
            "Scalars",
            ha="left",
            va="center",
            fontweight="bold",
        )
        self._sidebar_subtitle = ax_title.text(
            0.0,
            0.12,
            f"run dir: {self.run_dir.name}",
            ha="left",
            va="center",
            alpha=0.8,
        )
        y -= title_h + gap

        filter_h: float = 0.06
        fax: Axes = self.fig.add_axes([left_x, y - filter_h, left_w, filter_h])
        self.filter_box = TextBox(fax, "filter", initial="")
        self.filter_box.on_submit(self._on_filter_submit)
        y -= filter_h + gap

        x_title_h: float = 0.03
        xax_title: Axes = self.fig.add_axes([left_x, y - x_title_h, left_w, x_title_h])
        xax_title.set_axis_off()
        self._xaxis_title = xax_title.text(
            0.0,
            0.5,
            "X axis",
            va="center",
            fontweight="bold",
        )
        y -= x_title_h + gap

        x_radios_h: float = 0.085
        xax: Axes = self.fig.add_axes([left_x, y - x_radios_h, left_w, x_radios_h])
        self.radio_x = RadioButtons(
            xax,
            labels=["step", "wall_time"],
            active=0 if self.cfg.xmode == "step" else 1,
        )
        self.radio_x.on_clicked(self._on_xmode_clicked)
        y -= x_radios_h + gap

        if len(self.runs) > 1:
            runs_title_h: float = 0.03
            rax_title: Axes = self.fig.add_axes([
                left_x,
                y - runs_title_h,
                left_w,
                runs_title_h,
            ])
            rax_title.set_axis_off()
            self._runs_title = rax_title.text(
                0.0,
                0.5,
                "Runs",
                va="center",
                fontweight="bold",
            )
            y -= runs_title_h + gap

            runs_h: float = 0.14
            rax: Axes = self.fig.add_axes([left_x, y - runs_h, left_w, runs_h])
            active_idx: int = max(0, self.runs.index(self.current_run or self.runs[0]))
            self.radio_run = RadioButtons(rax, self.runs, active=active_idx)
            self.radio_run.on_clicked(self._on_run_clicked)
            y -= runs_h + gap

        tags_title_h: float = 0.03
        tax_title: Axes = self.fig.add_axes([
            left_x,
            y - tags_title_h,
            left_w,
            tags_title_h,
        ])
        tax_title.set_axis_off()
        self._tags_title = tax_title.text(
            0.0,
            0.5,
            "Tags",
            va="center",
            fontweight="bold",
        )
        y -= tags_title_h + gap

        tags_bottom: float = 0.14
        tags_h: float = max(0.08, y - tags_bottom)
        tax: Axes = self.fig.add_axes([left_x, y - tags_h, left_w, tags_h])
        labels: list[str] = (
            self.filtered_tags if self.filtered_tags else ["<no scalars>"]
        )
        self.radio_tag = RadioButtons(tax, labels, active=0)
        self.radio_tag.on_clicked(self._on_tag_clicked)

        rects: dict[str, float] = self._right_panel_rects()
        sax: Axes = self.fig.add_axes([
            rects["slider_left"],
            rects["slider_bottom"],
            rects["slider_width"],
            rects["slider_height"],
        ])
        self.smooth_slider = Slider(
            ax=sax,
            label="smoothing",
            valmin=0.0,
            valmax=0.999,
            valinit=0.0,
        )
        self.smooth_slider.on_changed(self._on_smoothing_changed)

    def _set_status(self, text: str) -> None:
        """Update status line above the plot."""
        self.ax_status.clear()
        self.ax_status.set_axis_off()
        self.ax_status.text(
            0.0,
            0.5,
            text,
            ha="left",
            va="center",
            family="monospace",
            alpha=0.85,
            fontsize=self._fs(_BASE_SMALL_FS),
        )
        self.fig.canvas.draw_idle()

    def _reload_all(self, initial: bool) -> None:
        """Reload TensorBoard data (runs, tags) like TensorBoard."""
        try:
            self.mux.Reload()
        except Exception as e:
            self.runs = []
            self.current_run = None
            self.all_tags = []
            self.filtered_tags = []
            self.current_tag = None
            self._set_status(text=f"Error recargando tfevents: {e}")
            return

        self.runs = _list_runs(mux=self.mux)
        if not self.runs:
            self.current_run = None
            self.all_tags = []
            self.filtered_tags = []
            self.current_tag = None
            msg: str = "Esperando a que aparezcan archivos tfevents en la run..."
            if not initial:
                msg = "No se detectan runs/tfevents todavía."
            self._set_status(text=msg)
            return

        if self.current_run is None or self.current_run not in self.runs:
            preferred: str = str(self.run_dir.name)
            self.current_run = preferred if preferred in self.runs else self.runs[0]

        self.all_tags = _list_scalar_tags(mux=self.mux, run=self.current_run)
        if not self.all_tags:
            self.filtered_tags = []
            self.current_tag = None
            self._set_status(text="No hay scalars todavía (scalars vacío).")
            return

        if self.current_tag is None:
            if self.cfg.tag is not None and self.cfg.tag in self.all_tags:
                self.current_tag = self.cfg.tag
            else:
                self.current_tag = self.all_tags[0]

        current_filter: str = self.filter_box.text if self.filter_box else ""
        self._apply_filter(filter_text=current_filter)

    def _apply_filter(self, filter_text: str) -> None:
        """Apply a substring filter to tag list and keep selection if possible.

        Args:
            filter_text: Substring filter (case-insensitive).
        """
        needle: str = str(filter_text).strip().lower()
        if needle:
            self.filtered_tags = [t for t in self.all_tags if needle in t.lower()]
        else:
            self.filtered_tags = list(self.all_tags)

        if not self.filtered_tags:
            self.current_tag = None
        elif self.current_tag not in self.filtered_tags:
            self.current_tag = self.filtered_tags[0]

        self._rebuild_tag_radio(labels=self.filtered_tags or ["<no scalars>"])

    def _rebuild_tag_radio(self, labels: list[str]) -> None:
        """Rebuild the tag RadioButtons widget.

        Args:
            labels: New labels to display.
        """
        if self.radio_tag is None:
            return
        ax: Axes = self.radio_tag.ax
        ax.clear()
        self.radio_tag = RadioButtons(ax, labels, active=0)
        self.radio_tag.on_clicked(self._on_tag_clicked)
        self._apply_ui_scale()
        self.fig.canvas.draw_idle()

    def _xmode(self) -> XMode:
        """Get current x-axis mode from UI/config.

        Returns:
            Current x-axis mode.
        """
        if self.radio_x is None:
            return self.cfg.xmode
        label: str = str(self.radio_x.value_selected)
        return "wall_time" if label == "wall_time" else "step"

    def _smoothing(self) -> float:
        """Get current smoothing from UI.

        Returns:
            Smoothing weight.
        """
        if self.smooth_slider is None:
            return 0.0
        return float(self.smooth_slider.val)

    def _force_redraw(self) -> None:
        """Redraw plot for current run/tag."""
        if self.current_run is None or self.current_tag is None:
            self.raw_line.set_data([], [])
            self.smooth_line.set_data([], [])
            self.ax_plot.relim()
            self.ax_plot.autoscale_view()
            self.ax_plot.set_title(
                self.run_dir.name, fontsize=self._fs(_BASE_PLOT_TITLE_FS)
            )
            self._set_status("run/tag no seleccionados")
            return

        try:
            xs, values = _get_scalar_series(
                mux=self.mux,
                run=self.current_run,
                tag=self.current_tag,
                xmode=self._xmode(),
            )
        except Exception as e:
            self._set_status(text=f"Error leyendo '{self.current_tag}': {e}")
            return

        self.raw_line.set_data(xs, values)
        smooth_w: float = self._smoothing()
        smooth_vals: list[float] = _ema_smooth(values=values, weight=smooth_w)
        self.smooth_line.set_data(xs, smooth_vals)

        if self._xmode() == "step":
            self.ax_plot.set_xlabel("step")
        else:
            self.ax_plot.set_xlabel("wall_time (s)")

        self.ax_plot.set_title(
            f"{self.current_tag}",
            fontsize=self._fs(_BASE_PLOT_TITLE_FS),
        )

        self.ax_plot.relim()
        self.ax_plot.autoscale_view()

        if xs:
            last_x: float = float(xs[-1])
            last_v: float = float(values[-1])
            status: str = (
                f"run={self.current_run}  "
                f"points={len(xs)}  "
                f"last_x={last_x:.6g}  "
                f"last_v={last_v:.6g}  "
                f"smoothing={smooth_w:.3f}"
            )
            self._set_status(text=status)
        else:
            self._set_status(
                text=f"run={self.current_run}  tag={self.current_tag}  sin puntos"
            )

        self._apply_ui_scale()
        self.fig.canvas.draw_idle()

    def _on_filter_submit(self, _text: str) -> None:
        """Handle tag filter submit.

        Args:
            _text: Ignored (TextBox passes the text).
        """
        self._apply_filter(filter_text=self.filter_box.text if self.filter_box else "")
        self._force_redraw()

    def _on_xmode_clicked(self, _label: str) -> None:
        """Handle xmode change."""
        self._force_redraw()

    def _on_smoothing_changed(self, _val: float) -> None:
        """Handle smoothing slider change."""
        self._force_redraw()

    def _on_run_clicked(self, label: str) -> None:
        """Handle run selection change."""
        self.current_run = str(label)
        self.all_tags = _list_scalar_tags(mux=self.mux, run=self.current_run)
        self.current_tag = self.all_tags[0] if self.all_tags else None
        self._apply_filter(filter_text=self.filter_box.text if self.filter_box else "")
        self._force_redraw()

    def _on_tag_clicked(self, label: str) -> None:
        """Handle tag selection change."""
        if str(label) == "<no scalars>":
            self.current_tag = None
        else:
            self.current_tag = str(label)
        self._force_redraw()

    def _tick(self, _frame: int) -> None:
        """Periodic update callback for matplotlib animation."""
        prev_run: str | None = self.current_run
        prev_tag: str | None = self.current_tag

        self._reload_all(initial=False)

        if prev_run is not None and prev_run in self.runs:
            self.current_run = prev_run

        if self.current_run is not None and prev_tag is not None:
            tags_now: list[str] = _list_scalar_tags(mux=self.mux, run=self.current_run)
            if prev_tag in tags_now:
                self.current_tag = prev_tag

        self._force_redraw()

    def run(self) -> None:
        """Open the interactive window maximized (keeping title bar)."""
        interval_ms: int = int(max(0.1, float(self.cfg.refresh_s)) * 1000.0)

        _ = FuncAnimation(
            self.fig,
            self._tick,
            interval=interval_ms,
            cache_frame_data=False,
        )

        manager = plt.get_current_fig_manager()

        try:
            window = manager.window

            try:
                window.wm_state("zoomed")
            except Exception:
                pass

            try:
                window.attributes("-zoomed", True)
            except Exception:
                pass

            try:
                window.showMaximized()
            except Exception:
                pass

            try:
                window.maximize()
            except Exception:
                pass

        except Exception:
            pass

        plt.show()


def _parse_args() -> ViewerConfig:
    """Parse CLI arguments.

    Returns:
        ViewerConfig built from CLI args.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Viewer tipo TensorBoard para scalars (lectura recursiva de tfevents) "
            "y refresco en tiempo real."
        )
    )
    parser.add_argument(
        "run_path",
        type=str,
        help="Ruta a la run (directorio dentro de runs/ que contiene tfevents).",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=1.0,
        help="Segundos entre refrescos (default: 1.0).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag escalar a mostrar. Si no se pasa, se elige el primero.",
    )
    parser.add_argument(
        "--xmode",
        type=str,
        default="step",
        choices=["step", "wall_time"],
        help="Eje X: step o wall_time (default: step).",
    )

    args = parser.parse_args()

    cfg: ViewerConfig = ViewerConfig(
        run_path=Path(str(args.run_path)).expanduser(),
        refresh_s=float(args.refresh),
        tag=str(args.tag) if args.tag is not None else None,
        xmode="wall_time" if str(args.xmode) == "wall_time" else "step",
    )
    return cfg


def main() -> None:
    """CLI entry point."""
    cfg: ViewerConfig = _parse_args()
    viewer: MetricViewer = MetricViewer(cfg=cfg)
    viewer.run()


if __name__ == "__main__":
    main()
