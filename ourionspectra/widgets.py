"""
Reusable custom Tkinter widgets.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from .config import NAVY, BORDER, TEXT_SUB, GREEN, GRAY_LINE


def icon_badge(parent, kind, bg="white", size=32, color=NAVY):
    """A small circle-outline badge with a simple glyph drawn inside,
    used before card titles (activity / target / bars / upload)."""
    c = tk.Canvas(parent, width=size, height=size, bg=bg, highlightthickness=0)
    r = size / 2
    c.create_oval(2, 2, size - 2, size - 2, outline=color, width=1.5)
    cx, cy = r, r
    if kind == "activity":
        pts = [cx - 8, cy, cx - 4, cy, cx - 1, cy - 6, cx + 2, cy + 6, cx + 5, cy, cx + 8, cy]
        c.create_line(*pts, fill=color, width=1.6, smooth=False, capstyle=tk.ROUND, joinstyle=tk.ROUND)
    elif kind == "target":
        for rad in (7, 4, 1.5):
            c.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, outline=color, width=1.3)
    elif kind == "bars":
        for i, h in enumerate((4, 8, 6)):
            x0 = cx - 7 + i * 6
            c.create_rectangle(x0, cy + 6 - h, x0 + 4, cy + 6, outline=color, fill="", width=1.3)
    elif kind == "upload":
        c.create_line(cx, cy - 6, cx, cy + 4, fill=color, width=1.6, capstyle=tk.ROUND)
        c.create_line(cx - 4, cy - 2, cx, cy - 6, cx + 4, cy - 2, fill=color, width=1.6,
                       smooth=False, capstyle=tk.ROUND, joinstyle=tk.ROUND)
        c.create_line(cx - 6, cy + 6, cx + 6, cy + 6, fill=color, width=1.6, capstyle=tk.ROUND)
    return c


def bordered_card(parent, bg="white", border_color=BORDER):
    """A frame with a thin 1px border, approximating a rounded card edge."""
    outer = tk.Frame(parent, bg=border_color)
    inner = tk.Frame(outer, bg=bg)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    return outer, inner


class CsvColumnMappingDialog(tk.Toplevel):
    """
    Modal dialog that lets the user map arbitrary CSV columns to
    wavelength/flux, pick a wavelength unit, and choose whether to
    normalize flux. Handles files that aren't already in the app's
    exact 2-column (wavelength_nm, normalized_flux) format.

    Usage:
        dlg = CsvColumnMappingDialog(parent, headers, preview_rows, unit_options)
        parent.wait_window(dlg)
        if dlg.result:
            wl_idx, flux_idx, unit_label, normalize = dlg.result
    """

    def __init__(self, parent, headers, preview_rows, unit_options,
                 guessed_wl_idx=0, guessed_flux_idx=1, suggest_normalize=False):
        super().__init__(parent)
        self.title("Map CSV Columns")
        self.configure(bg="white")
        self.resizable(False, False)
        self.result = None

        tk.Label(self, text="This file has more than 2 columns — tell us which ones to use.",
                 bg="white", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=16, pady=(14, 4))

        preview_frame = tk.Frame(self, bg="white")
        preview_frame.pack(fill="x", padx=16, pady=(4, 10))
        cols = [f"c{i}" for i in range(len(headers))]
        tree = ttk.Treeview(preview_frame, columns=cols, show="headings", height=4)
        for i, h in enumerate(headers):
            tree.heading(f"c{i}", text=h)
            tree.column(f"c{i}", width=90, anchor="w")
        for row in preview_rows[:5]:
            tree.insert("", "end", values=row)
        tree.pack(fill="x")

        form = tk.Frame(self, bg="white")
        form.pack(fill="x", padx=16, pady=(0, 10))

        tk.Label(form, text="Wavelength column:", bg="white", font=("Segoe UI", 9)).grid(
            row=0, column=0, sticky="w", pady=4)
        self.wl_var = tk.StringVar(value=headers[guessed_wl_idx])
        wl_combo = ttk.Combobox(form, textvariable=self.wl_var, values=headers, state="readonly", width=28)
        wl_combo.grid(row=0, column=1, padx=(8, 0), pady=4)

        tk.Label(form, text="Wavelength unit:", bg="white", font=("Segoe UI", 9)).grid(
            row=1, column=0, sticky="w", pady=4)
        self.unit_var = tk.StringVar(value=unit_options[0])
        unit_combo = ttk.Combobox(form, textvariable=self.unit_var, values=unit_options,
                                   state="readonly", width=28)
        unit_combo.grid(row=1, column=1, padx=(8, 0), pady=4)

        tk.Label(form, text="Flux / value column:", bg="white", font=("Segoe UI", 9)).grid(
            row=2, column=0, sticky="w", pady=4)
        self.flux_var = tk.StringVar(value=headers[guessed_flux_idx])
        flux_combo = ttk.Combobox(form, textvariable=self.flux_var, values=headers, state="readonly", width=28)
        flux_combo.grid(row=2, column=1, padx=(8, 0), pady=4)

        self.normalize_var = tk.BooleanVar(value=suggest_normalize)
        tk.Checkbutton(form, text="Normalize flux to a 0–1 range (recommended for raw physical units)",
                        variable=self.normalize_var, bg="white", font=("Segoe UI", 9)).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        btn_row = tk.Frame(self, bg="white")
        btn_row.pack(fill="x", padx=16, pady=(4, 14))
        tk.Button(btn_row, text="Cancel", command=self._on_cancel,
                  relief="solid", bd=1, padx=12, pady=5).pack(side="right", padx=(8, 0))
        tk.Button(btn_row, text="Use These Columns", command=self._on_confirm,
                  bg=NAVY, fg="white", relief="flat", padx=12, pady=5).pack(side="right")

        self._headers = headers
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.transient(parent)
        self.grab_set()

    def _on_confirm(self):
        wl_idx = self._headers.index(self.wl_var.get())
        flux_idx = self._headers.index(self.flux_var.get())
        if wl_idx == flux_idx:
            messagebox.showwarning("Pick different columns",
                                    "Wavelength and flux must be different columns.", parent=self)
            return
        self.result = (wl_idx, flux_idx, self.unit_var.get(), self.normalize_var.get())
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


class CanvasSlider(tk.Canvas):
    """
    A minimal flat slider drawn on a Canvas: a navy track + circular thumb.
    Resizes correctly (redraws on <Configure>), unlike drawing once at
    construction time with a guessed width.

    Usage:
        slider = CanvasSlider(parent, from_=0.0, to=1.0, value=0.3,
                               command=lambda v: print(v))
        slider.pack(fill="x", expand=True)
    """

    def __init__(self, parent, from_=0.0, to=1.0, value=0.0,
                 command=None, resolution=0.01,
                 track_color=NAVY, thumb_color=NAVY,
                 padding=10, track_height=8, thumb_radius=10,
                 height=40, **kwargs):
        super().__init__(parent, height=height, highlightthickness=0, **kwargs)

        self.min_val = from_
        self.max_val = to
        self.resolution = resolution
        self.command = command
        self.track_color = track_color
        self.thumb_color = thumb_color
        self.padding = padding
        self.track_height = track_height
        self.thumb_radius = thumb_radius

        self._value = self._clamp(value)

        self.bind("<Configure>", lambda e: self._redraw())
        self.bind("<Button-1>", self._on_pointer)
        self.bind("<B1-Motion>", self._on_pointer)

    # ---------------------------------------------------------- public API
    def get(self):
        return self._value

    def set(self, value, fire_callback=False):
        self._value = self._clamp(value)
        self._redraw()
        if fire_callback and self.command:
            self.command(self._value)

    # ---------------------------------------------------------- internals
    def _clamp(self, value):
        value = max(self.min_val, min(self.max_val, value))
        if self.resolution:
            steps = round((value - self.min_val) / self.resolution)
            value = self.min_val + steps * self.resolution
        return round(value, 6)

    def _track_bounds(self):
        w = self.winfo_width()
        if w < 10:
            w = int(self["width"] or 200)
        return self.padding, w - self.padding

    def _redraw(self):
        self.delete("all")
        left, right = self._track_bounds()
        track_y = self.padding + self.track_height // 2

        self.create_line(left, track_y, right, track_y,
                          fill=self.track_color, width=self.track_height,
                          capstyle=tk.ROUND)

        frac = 0.0
        if self.max_val > self.min_val:
            frac = (self._value - self.min_val) / (self.max_val - self.min_val)
        x = left + frac * (right - left)
        r = self.thumb_radius
        self.create_oval(x - r, track_y - r, x + r, track_y + r,
                          fill=self.thumb_color, outline=self.thumb_color, width=1)

    def _on_pointer(self, event):
        left, right = self._track_bounds()
        x = min(max(event.x, left), right)
        frac = (x - left) / (right - left) if right > left else 0.0
        value = self.min_val + frac * (self.max_val - self.min_val)
        self.set(value, fire_callback=True)


class WorkflowIndicator(tk.Frame):
    """
    Compact 5-step workflow strip: numbered circles connected by a line,
    each labeled underneath. Reflects real application state only —
    the caller decides which steps are 'done' / 'active' by passing
    indices in; this widget never invents progress on its own.

    Usage:
        wf = WorkflowIndicator(parent, steps=["LOAD DATA", "PREPROCESS", ...])
        wf.pack(fill="x")
        wf.update_state(completed={0, 1}, active=2)
    """

    def __init__(self, parent, steps, bg="white", **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        self.steps = steps
        self.bg = bg
        self._completed = set()
        self._active = None

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, height=64)
        self.canvas.pack(fill="x", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

    def update_state(self, completed=None, active=None):
        self._completed = set(completed or set())
        self._active = active
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        if w < 10:
            w = 900
        n = len(self.steps)
        if n == 0:
            return
        pad = 60
        usable = max(w - 2 * pad, 1)
        step_x = usable / (n - 1) if n > 1 else 0
        cy = 22
        r = 15

        centers = [pad + i * step_x for i in range(n)]

        # connecting line drawn first so circles sit on top
        for i in range(n - 1):
            done = i in self._completed and (i + 1) in self._completed or (
                i in self._completed and self._active == i + 1
            )
            color = GREEN if (i in self._completed) else GRAY_LINE
            self.canvas.create_line(centers[i] + r, cy, centers[i + 1] - r, cy,
                                     fill=color, width=2)

        for i, label in enumerate(self.steps):
            cx = centers[i]
            if i in self._completed:
                fill, outline, text_color, num_color = GREEN, GREEN, NAVY, "white"
                glyph = "✓"
            elif i == self._active:
                fill, outline, text_color, num_color = NAVY, NAVY, NAVY, "white"
                glyph = str(i + 1)
            else:
                fill, outline, text_color, num_color = self.bg, GRAY_LINE, TEXT_SUB, GRAY_LINE
                glyph = str(i + 1)

            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     fill=fill, outline=outline, width=1.6)
            self.canvas.create_text(cx, cy, text=glyph,
                                     fill=(num_color if (i in self._completed or i == self._active) else num_color),
                                     font=("Segoe UI", 10, "bold"))
            weight = "bold" if i == self._active else "normal"
            self.canvas.create_text(cx, cy + r + 14, text=label,
                                     fill=(NAVY if i == self._active else text_color),
                                     font=("Segoe UI", 8, weight))
