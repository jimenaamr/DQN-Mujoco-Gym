# src/DQN_walker2d/gui_viewer.py

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GuiPacket:
    """Single GUI update packet (frame + legend text)."""

    frame_rgb: np.ndarray
    legend_text: str


class TkLiveViewer:
    """Tk window showing image (left) and legend text (right) as real widgets.

    The image is rendered inside a Canvas that controls its own size, so resizing
    the window will not cause feedback loops (unlike Label autosizing to image).
    The image is resized to fit the Canvas while preserving aspect ratio.
    """

    def __init__(self, enabled: bool, title: str = "training_live") -> None:
        self._enabled: bool = bool(enabled)
        self._title: str = str(title)

        self._q: queue.Queue[GuiPacket] = queue.Queue(maxsize=1)
        self._stop_evt: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the Tk viewer thread (no-op if disabled)."""
        if not self._enabled or self._thread is not None:
            return
        th: threading.Thread = threading.Thread(
            target=self._run_tk, name="tk-live-viewer", daemon=True
        )
        self._thread = th
        th.start()

    def close(self) -> None:
        """Request the viewer to stop (no-op if disabled)."""
        if not self._enabled:
            return
        self._stop_evt.set()

    def push(self, frame_rgb: np.ndarray, legend_text: str) -> None:
        """Push a new frame + legend to the GUI (drops older packet)."""
        if not self._enabled:
            return

        if frame_rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame, got {frame_rgb.dtype}")
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3) frame, got {frame_rgb.shape}")

        pkt: GuiPacket = GuiPacket(
            frame_rgb=np.ascontiguousarray(frame_rgb),
            legend_text=str(legend_text),
        )

        try:
            self._q.put_nowait(pkt)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(pkt)
            except queue.Full:
                pass

    def _run_tk(self) -> None:
        """Tk mainloop runner (executed in viewer thread)."""
        import tkinter as tk

        from PIL import Image, ImageTk

        root = tk.Tk()
        root.title(self._title)
        root.configure(bg="white")

        # Left: Canvas that expands with the window
        canvas = tk.Canvas(root, bg="white", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        # Right: Legend
        legend_var = tk.StringVar(value="")
        legend_label = tk.Label(
            root,
            textvariable=legend_var,
            justify="left",
            anchor="nw",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 9),
        )
        legend_label.grid(row=0, column=1, sticky="nw", padx=(8, 8), pady=8)

        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=0)
        root.grid_rowconfigure(0, weight=1)

        # Keep references
        tk_img = None
        img_item_id: int | None = None

        last_frame: np.ndarray | None = None
        last_legend: str = ""

        pending_resize: bool = False

        def _fit_size(
            src_w: int, src_h: int, max_w: int, max_h: int
        ) -> tuple[int, int]:
            """Compute (w,h) to fit inside max_w/max_h keeping aspect ratio."""
            if max_w <= 1 or max_h <= 1:
                return 1, 1
            scale_w: float = max_w / float(src_w)
            scale_h: float = max_h / float(src_h)
            scale: float = min(scale_w, scale_h)
            new_w: int = max(1, int(round(src_w * scale)))
            new_h: int = max(1, int(round(src_h * scale)))
            return new_w, new_h

        def _render(frame_rgb: np.ndarray, legend_text: str) -> None:
            nonlocal tk_img, img_item_id, last_frame, last_legend

            last_frame = frame_rgb
            last_legend = legend_text

            try:
                cw: int = int(canvas.winfo_width())
                ch: int = int(canvas.winfo_height())
                if cw <= 1 or ch <= 1:
                    # Canvas not yet laid out; try again soon
                    root.after(
                        30,
                        lambda: _render(frame_rgb=frame_rgb, legend_text=legend_text),
                    )
                    return

                src_h: int = int(frame_rgb.shape[0])
                src_w: int = int(frame_rgb.shape[1])
                dst_w: int
                dst_h: int
                dst_w, dst_h = _fit_size(src_w=src_w, src_h=src_h, max_w=cw, max_h=ch)

                pil_img = Image.fromarray(frame_rgb, mode="RGB").resize(
                    (dst_w, dst_h), resample=Image.BILINEAR
                )
                tk_img = ImageTk.PhotoImage(image=pil_img)

                # Center on canvas
                x: int = cw // 2
                y: int = ch // 2

                if img_item_id is None:
                    img_item_id = canvas.create_image(
                        x, y, image=tk_img, anchor="center"
                    )
                else:
                    canvas.coords(img_item_id, x, y)
                    canvas.itemconfig(img_item_id, image=tk_img)

                legend_var.set(legend_text)
            except Exception as e:
                traceback.print_exc()
                legend_var.set(f"GUI render error:\n{type(e).__name__}: {e}")

        def on_close() -> None:
            self._stop_evt.set()

        root.protocol("WM_DELETE_WINDOW", on_close)

        def on_canvas_configure(_evt: object) -> None:
            nonlocal pending_resize
            if pending_resize:
                return
            pending_resize = True

            def do_resize() -> None:
                nonlocal pending_resize
                pending_resize = False
                if last_frame is not None:
                    _render(frame_rgb=last_frame, legend_text=last_legend)

            # Debounce resize events
            root.after(60, do_resize)

        canvas.bind("<Configure>", on_canvas_configure)

        def poll() -> None:
            if self._stop_evt.is_set():
                try:
                    root.destroy()
                except Exception:
                    pass
                return

            pkt: GuiPacket | None = None
            try:
                while True:
                    pkt = self._q.get_nowait()
            except queue.Empty:
                pass

            if pkt is not None:
                _render(frame_rgb=pkt.frame_rgb, legend_text=pkt.legend_text)

            root.after(30, poll)

        root.update_idletasks()
        poll()
        root.mainloop()
