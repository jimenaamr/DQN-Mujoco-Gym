# src/DQN_walker2d/gui_viewer.py

from __future__ import annotations

import base64
import time
import tkinter as tk
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class _ViewerState:
    """Internal Tk viewer state."""

    root: tk.Tk
    image_label: tk.Label
    text_label: tk.Label
    photo: tk.PhotoImage | None
    last_draw_ts: float

    last_frame_rgb: np.ndarray | None
    last_frame_w: int
    last_frame_h: int

    last_target_w: int
    last_target_h: int


class TkLiveViewer:
    """Tkinter viewer to show a live RGB frame + legend text.

    Notes:
        Tk must be manipulated from the main thread. This viewer uses no threads.
        If `scale_to_window=True`, frames are resized to match the label size.

    Args:
        enabled: If False, methods become no-ops.
        title: Window title.
        max_fps: UI refresh cap.
        auto_resize: If True (and scale_to_window is False), window follows frame.
        scale_to_window: If True, scale image to label size.
        keep_aspect: If True, preserve aspect ratio when scaling.
    """

    def __init__(
        self,
        enabled: bool,
        title: str = "viewer",
        max_fps: float = 30.0,
        auto_resize: bool = False,
        scale_to_window: bool = True,
        keep_aspect: bool = True,
    ) -> None:
        self.enabled: bool = bool(enabled)
        self.title: str = str(title)
        self.max_fps: float = float(max_fps)
        self.auto_resize: bool = bool(auto_resize)
        self.scale_to_window: bool = bool(scale_to_window)
        self.keep_aspect: bool = bool(keep_aspect)

        self._state: _ViewerState | None = None
        self._closed: bool = False

    def start(self) -> None:
        """Create the Tk window (main-thread only)."""
        if not self.enabled or self._closed:
            return
        if self._state is not None:
            return

        root: tk.Tk = tk.Tk()
        root.title(self.title)

        image_label: tk.Label = tk.Label(root, borderwidth=0, highlightthickness=0)
        image_label.pack(side="top", fill="both", expand=True)

        text_label: tk.Label = tk.Label(
            root,
            justify="left",
            anchor="w",
            font=("TkFixedFont", 10),
        )
        text_label.pack(side="bottom", fill="x")

        def _on_close() -> None:
            self.enabled = False
            self.close()

        root.protocol("WM_DELETE_WINDOW", _on_close)

        self._state = _ViewerState(
            root=root,
            image_label=image_label,
            text_label=text_label,
            photo=None,
            last_draw_ts=0.0,
            last_frame_rgb=None,
            last_frame_w=0,
            last_frame_h=0,
            last_target_w=0,
            last_target_h=0,
        )

        root.bind("<Configure>", self._on_configure)

        root.update_idletasks()
        root.update()

    def push(self, frame_rgb: np.ndarray, legend_text: str) -> None:
        """Render one frame and legend text.

        Args:
            frame_rgb: Frame in HWC uint8 RGB format.
            legend_text: Multi-line legend string.
        """
        if not self.enabled or self._closed:
            return
        if self._state is None:
            self.start()
            if self._state is None:
                return

        now: float = time.time()
        min_dt: float = 1.0 / self.max_fps if self.max_fps > 0.0 else 0.0
        if (now - self._state.last_draw_ts) < min_dt:
            self._safe_update()
            return

        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"Expected frame_rgb as (H,W,3), got {frame_rgb.shape}")
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8, copy=False)

        h: int = int(frame_rgb.shape[0])
        w: int = int(frame_rgb.shape[1])

        self._state.last_frame_rgb = frame_rgb
        self._state.last_frame_h = h
        self._state.last_frame_w = w

        self._state.text_label.configure(text=str(legend_text))
        self._state.root.update_idletasks()

        if self.auto_resize and (not self.scale_to_window):
            self._resize_window_to_frame(frame_w=w, frame_h=h)

        # Force repaint on new frames even if the target size doesn't change.
        self._render_last_frame(force=True)
        self._state.last_draw_ts = now

        self._safe_update()

    def close(self) -> None:
        """Destroy the Tk window if present."""
        if self._closed:
            return
        self._closed = True

        if self._state is None:
            return

        try:
            self._state.root.destroy()
        except Exception:
            pass
        finally:
            self._state = None

    def _on_configure(self, _event: object) -> None:
        """Handle window resize by re-rendering the last frame to fit."""
        if self._state is None:
            return
        if not self.scale_to_window:
            return
        # On resize, only repaint if target size changed.
        self._render_last_frame(force=False)

    def _render_last_frame(self, force: bool) -> None:
        """Render cached frame into the label.

        Args:
            force: If True, always repaint even if target size is unchanged.
        """
        if self._state is None:
            return
        if self._state.last_frame_rgb is None:
            return

        frame_rgb: np.ndarray = self._state.last_frame_rgb
        target_w: int
        target_h: int
        target_w, target_h = self._get_target_image_size(
            frame_w=int(self._state.last_frame_w),
            frame_h=int(self._state.last_frame_h),
        )

        size_unchanged: bool = (
            target_w == self._state.last_target_w
            and target_h == self._state.last_target_h
        )
        if (not force) and size_unchanged and (self._state.photo is not None):
            return

        self._state.last_target_w = int(target_w)
        self._state.last_target_h = int(target_h)

        frame_to_draw: np.ndarray = self._resize_frame_if_needed(
            frame_rgb=frame_rgb,
            target_w=int(target_w),
            target_h=int(target_h),
        )

        photo: tk.PhotoImage = self._frame_to_photo(frame_rgb=frame_to_draw)
        self._state.photo = photo
        self._state.image_label.configure(image=photo)

    def _get_target_image_size(self, frame_w: int, frame_h: int) -> tuple[int, int]:
        """Compute image draw size based on viewer configuration.

        Args:
            frame_w: Original frame width.
            frame_h: Original frame height.

        Returns:
            Target (w, h) for drawing.
        """
        if self._state is None:
            return int(frame_w), int(frame_h)

        if not self.scale_to_window:
            return int(frame_w), int(frame_h)

        label_w: int = int(self._state.image_label.winfo_width())
        label_h: int = int(self._state.image_label.winfo_height())

        if label_w <= 1 or label_h <= 1:
            return int(frame_w), int(frame_h)

        if not self.keep_aspect:
            return int(label_w), int(label_h)

        scale: float = min(
            float(label_w) / float(frame_w),
            float(label_h) / float(frame_h),
        )
        target_w: int = int(max(1, round(float(frame_w) * scale)))
        target_h: int = int(max(1, round(float(frame_h) * scale)))
        return target_w, target_h

    def _resize_frame_if_needed(
        self, frame_rgb: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Resize frame to target size if needed.

        Args:
            frame_rgb: Input RGB frame.
            target_w: Target width.
            target_h: Target height.

        Returns:
            Possibly resized RGB frame.
        """
        h: int = int(frame_rgb.shape[0])
        w: int = int(frame_rgb.shape[1])
        if w == int(target_w) and h == int(target_h):
            return frame_rgb

        interp: int = (
            cv2.INTER_AREA if (target_w < w or target_h < h) else cv2.INTER_LINEAR
        )
        resized: np.ndarray = cv2.resize(
            src=frame_rgb,
            dsize=(int(target_w), int(target_h)),
            interpolation=interp,
        )
        return resized

    def _frame_to_photo(self, frame_rgb: np.ndarray) -> tk.PhotoImage:
        """Convert an RGB frame to a Tk PhotoImage.

        Args:
            frame_rgb: RGB uint8 frame.

        Returns:
            Tk PhotoImage instance.
        """
        frame_bgr: np.ndarray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok: bool
        buf: np.ndarray
        ok, buf = cv2.imencode(ext=".png", img=frame_bgr)
        if not ok:
            raise RuntimeError("Failed to encode frame to PNG for Tk viewer.")
        png_b64: bytes = base64.b64encode(buf.tobytes())
        return tk.PhotoImage(data=png_b64)

    def _resize_window_to_frame(self, frame_w: int, frame_h: int) -> None:
        """Resize the window so the frame is not clipped.

        Args:
            frame_w: Frame width.
            frame_h: Frame height.
        """
        if self._state is None:
            return
        text_h: int = int(self._state.text_label.winfo_reqheight())
        content_w: int = int(frame_w)
        content_h: int = int(frame_h + text_h)
        self._state.root.geometry(f"{content_w}x{content_h}")
        self._state.root.minsize(width=content_w, height=content_h)

    def _safe_update(self) -> None:
        """Process pending Tk events safely."""
        if self._state is None:
            return
        try:
            self._state.root.update_idletasks()
            self._state.root.update()
        except tk.TclError:
            self.enabled = False
            self.close()
