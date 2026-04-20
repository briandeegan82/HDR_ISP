#!/usr/bin/env python3
"""
ISP Tuning GUI — live parameter editing with auto-reprocess.

Features
--------
* Blocks tab    : toggle pipeline stages on/off; click a label to edit the block's full YAML.
* Parameters tab: sliders and dropdowns for the most-commonly tuned parameters, wired
                  directly to the in-memory config.
* Auto-process  : enable the checkbox and any parameter change automatically re-runs the
                  pipeline after a short debounce (default 400 ms).
* Save All      : writes the YAML config and the output image in one click (Ctrl+S).
* Status bar    : shows config / raw filenames, modified state, and processing status.

Run from repo root:  python tools/isp_tuning_gui.py
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any

# Repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "TkAgg")

import cv2
import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from brilliant_isp import BrilliantISP
from util.config_merge import load_merged_yaml, pipeline_config_paths

# ── Constants ─────────────────────────────────────────────────────────────────

_BLOCK_ORDER = [
    "crop",
    "dead_pixel_correction",
    "black_level_correction",
    "companding",
    "oecf",
    "digital_gain",
    "lens_shading_correction",
    "bayer_noise_reduction",
    "auto_white_balance",
    "white_balance",
    "tone_mapping",
    "color_correction_matrix",
    "auto_exposure",
    "color_saturation_enhancement",
    "ldci",
    "sharpen",
    "2d_noise_reduction",
    "rgb_conversion",
    "gamma_correction",
    "scale",
    "yuv_conversion_format",
]

_TONE_MAPPER_PARAM_KEYS = frozenset(
    {"hdr_durand", "reinhard_integer", "aces_integer", "hable", "hable_integer", "aces"}
)

_DEMOSAIC_ALGORITHMS = [
    "bilinear", "malvar", "vng_opt", "hamilton_adams", "ppg", "ahd", "lmmse",
]
_TONE_MAPPERS = [
    "reinhard_integer", "aces", "aces_integer", "hable", "hable_integer", "hdr_durand",
]
_GAMMA_CURVES = ["srgb", "gamma"]
_AWB_ALGORITHMS = ["grey_world", "norm_2", "pca"]

_DEBOUNCE_MS = 400  # milliseconds before auto-reprocess fires after a parameter change

_ABOUT_TITLE = "About BrilliantISP Tuning"
_ABOUT_TEXT = (
    "BrilliantISP — Tuning GUI\n\n"
    "BrilliantISP is developed by Brian Deegan; portions of the pipeline derive from "
    "the Infinite-ISP implementation (10xEngineers).\n\n"
    "Workflow\n"
    "  1. File → Open config… — loads a YAML; *_cam.yml files are merged with "
    "config/base_hdr.yml automatically.\n"
    "  2. File → Open raw… — select a RAW or TIFF frame.\n"
    "  3. Adjust parameters in the Blocks or Parameters tab.\n"
    "     • Enable Auto-process for instant feedback (400 ms debounce).\n"
    "     • Or click Process manually at any time.\n"
    "  4. Save All (Ctrl+S) — saves the YAML config and output image together.\n"
    "     File → Save config / Save output image — save them separately.\n\n"
    "Parameters tab\n"
    "  Sliders and dropdowns for the most-commonly tuned parameters.  Every "
    "change is written immediately into the in-memory config and is included in "
    "the next Process run.  Use Save config or Save All to persist to disk.\n\n"
    "Blocks tab\n"
    "  Toggle individual pipeline stages.  Click a block name for a context menu "
    "that opens its full YAML for editing (any parameter, not just is_enable).\n\n"
    f"Project: {REPO_ROOT}\n"
    "Docs: docs/ISP_BLOCKS_AND_TUNING.md\n\n"
    "BrilliantISP — Brian Deegan (based in part on 10xEngineers / Infinite-ISP)."
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _yaml_safe_value(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _yaml_safe_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_yaml_safe_value(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _dump_block_yaml(section: dict[str, Any]) -> str:
    data = _yaml_safe_value(copy.deepcopy(section))
    return yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, default_flow_style=False
    )


def _merge_pipeline_feedback_into(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """
    Copy pipeline-run outputs (AWB gains, AE feedback, inferred raw size) into dst
    without overwriting user-edited blocks wholesale.
    """
    si_s, si_d = src.get("sensor_info"), dst.get("sensor_info")
    if isinstance(si_s, dict) and isinstance(si_d, dict):
        for k in ("width", "height", "bayer_pattern", "bit_depth", "hdr_bit_depth"):
            if k in si_s:
                si_d[k] = copy.deepcopy(si_s[k])
    wb_s, wb_d = src.get("white_balance"), dst.get("white_balance")
    if isinstance(wb_s, dict) and isinstance(wb_d, dict):
        for k in ("r_gain", "b_gain"):
            if k in wb_s:
                wb_d[k] = copy.deepcopy(wb_s[k])
    dg_s, dg_d = src.get("digital_gain"), dst.get("digital_gain")
    if isinstance(dg_s, dict) and isinstance(dg_d, dict):
        for k in ("ae_feedback", "current_gain"):
            if k in dg_s:
                dg_d[k] = copy.deepcopy(dg_s[k])


def _discover_block_keys(cfg: dict[str, Any]) -> list[str]:
    ordered = [
        k for k in _BLOCK_ORDER
        if k in cfg and isinstance(cfg[k], dict) and "is_enable" in cfg[k]
    ]
    extra = sorted(
        k for k in cfg
        if k not in ordered
        and k not in _TONE_MAPPER_PARAM_KEYS
        and k not in ("platform", "sensor_info")
        and isinstance(cfg[k], dict)
        and "is_enable" in cfg[k]
    )
    return ordered + extra


def _label_for_key(key: str) -> str:
    return key.replace("_", " ").strip().title()


# ── Main Application ──────────────────────────────────────────────────────────


class ISPTuningApp:
    def __init__(self, root, default_config: Path | None) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.ttk = ttk

        self.root = root
        root.title("BrilliantISP — Tuning")
        root.minsize(1000, 680)
        root.geometry("1300x820")

        # ── Application state ────────────────────────────────────────────────
        self.raw_path: Path | None = None
        self.config_path: Path | None = None
        self.save_path: Path | None = None
        self.working_config: dict[str, Any] | None = None
        self._block_vars: dict[str, tk.BooleanVar] = {}
        self._processing = False
        self._last_output_rgb: np.ndarray | None = None
        self._modified = False
        self._reprocess_after_id: str | None = None
        self._param_vars: dict[str, tk.Variable] = {}
        self._param_widgets: dict[str, Any] = {}
        self._auto_process_var = tk.BooleanVar(value=False)
        self._image_artist = None
        self._image_height = 0
        self._image_width = 0
        self._zoom_factor = 1.2
        self._pan_start: tuple[float, float, tuple[float, float], tuple[float, float]] | None = None
        self._crop_overlay_artist = None

        # ── Menu bar ─────────────────────────────────────────────────────────
        menubar = tk.Menu(root)
        root.config(menu=menubar)

        file_m = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_m)
        file_m.add_command(label="Open raw…", command=self._open_raw)
        file_m.add_command(label="Open config…", command=self._open_config)
        file_m.add_separator()
        file_m.add_command(
            label="Save All", command=self._save_all, accelerator="Ctrl+S"
        )
        file_m.add_separator()
        file_m.add_command(label="Save config", command=self._save_config)
        file_m.add_command(label="Save config as…", command=self._save_config_as)
        file_m.add_command(
            label="Save output image…", command=self._save_output_image
        )
        file_m.add_separator()
        file_m.add_command(label="Exit", command=root.quit)
        root.bind("<Control-s>", lambda _e: self._save_all())

        help_m = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_m)
        help_m.add_command(label="About…", command=self._show_about)

        # ── Toolbar ──────────────────────────────────────────────────────────
        self._build_toolbar()

        # ── Status bar (built before main content so it packs to bottom) ─────
        self._build_status_bar()

        # ── Main paned window ────────────────────────────────────────────────
        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=330)
        left.pack_propagate(False)
        main.add(left, weight=0)

        right = ttk.Frame(main)
        main.add(right, weight=1)

        # ── Left panel: Notebook (Blocks / Parameters) ────────────────────
        self._notebook = ttk.Notebook(left)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 2))

        blocks_tab = ttk.Frame(self._notebook)
        self._notebook.add(blocks_tab, text="  Blocks  ")
        self._build_blocks_tab(blocks_tab)

        params_tab = ttk.Frame(self._notebook)
        self._notebook.add(params_tab, text="  Parameters  ")
        self._build_params_tab(params_tab)

        # ── Right panel: Image canvas ─────────────────────────────────────
        self._fig = Figure(figsize=(8, 6), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.axis("off")
        self._ax.text(
            0.5, 0.5,
            "Open a RAW file and config,\nthen click  ⟳ Process.",
            ha="center", va="center", transform=self._ax.transAxes,
            fontsize=13, color="#888888",
        )
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._mpl_canvas.mpl_connect("scroll_event", self._on_image_scroll)
        self._mpl_canvas.mpl_connect("button_press_event", self._on_image_press)
        self._mpl_canvas.mpl_connect("button_release_event", self._on_image_release)
        self._mpl_canvas.mpl_connect("motion_notify_event", self._on_image_motion)
        self._mpl_canvas.get_tk_widget().bind(
            "<Enter>", lambda _e: self._mpl_canvas.get_tk_widget().focus_set()
        )

        self.ttk.Label(
            right,
            text=(
                "Tip: Mouse wheel zooms at cursor, left-drag pans. "
                "Keyboard: '+' zoom in, '-' zoom out, 'R' resets view. "
                "Crop section in Parameters shows a green dashed crop region overlay."
            ),
            foreground="#666666",
            anchor=self.tk.W,
            padding=(8, 2),
        ).pack(fill=self.tk.X, side=self.tk.BOTTOM)

        root.bind("+", lambda _e: self._zoom_keyboard(1.0 / self._zoom_factor))
        root.bind("=", lambda _e: self._zoom_keyboard(1.0 / self._zoom_factor))
        root.bind("-", lambda _e: self._zoom_keyboard(self._zoom_factor))
        root.bind("r", lambda _e: self._reset_image_view())
        root.bind("R", lambda _e: self._reset_image_view())

        # ── Initial config ────────────────────────────────────────────────
        if default_config and default_config.is_file():
            self._load_config_path(default_config)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        tk, ttk = self.tk, self.ttk
        tb = ttk.Frame(self.root, relief=tk.GROOVE, padding=(6, 4))
        tb.pack(fill=tk.X, side=tk.TOP)

        ttk.Button(tb, text="Open Raw", command=self._open_raw, width=10).pack(
            side=tk.LEFT, padx=(0, 2)
        )
        ttk.Button(tb, text="Open Config", command=self._open_config, width=12).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Separator(tb, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2
        )
        self.process_btn = ttk.Button(
            tb, text="⟳  Process", command=self._on_process, width=13
        )
        self.process_btn.pack(side=tk.LEFT, padx=2)
        self.save_all_btn = ttk.Button(
            tb, text="💾  Save All", command=self._save_all, width=13
        )
        self.save_all_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="Reset View", command=self._reset_image_view, width=11).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Separator(tb, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2
        )
        ttk.Checkbutton(
            tb, text="Auto-process", variable=self._auto_process_var
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(
            tb,
            text=f"  (re-runs pipeline {_DEBOUNCE_MS} ms after any change)",
            foreground="#888888",
        ).pack(side=tk.LEFT)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        tk, ttk = self.tk, self.ttk
        sb = ttk.Frame(self.root, relief=tk.SUNKEN)
        sb.pack(fill=tk.X, side=tk.BOTTOM)

        self._sb_config_lbl = ttk.Label(
            sb, text="No config loaded", relief=tk.SUNKEN,
            width=24, anchor=tk.W, padding=(6, 2),
        )
        self._sb_config_lbl.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Separator(sb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)

        self._sb_raw_lbl = ttk.Label(
            sb, text="No raw file", relief=tk.SUNKEN,
            width=24, anchor=tk.W, padding=(6, 2),
        )
        self._sb_raw_lbl.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Separator(sb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)

        self._sb_modified_lbl = ttk.Label(
            sb, text="", relief=tk.SUNKEN,
            width=12, anchor=tk.CENTER, padding=(6, 2),
        )
        self._sb_modified_lbl.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Separator(sb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)

        self._sb_status_lbl = ttk.Label(
            sb, text="Ready", relief=tk.SUNKEN,
            anchor=tk.W, padding=(6, 2),
        )
        self._sb_status_lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _update_status(self, msg: str = "") -> None:
        """Update the status bar. Pass a non-empty msg to set the status segment."""
        if msg:
            self._sb_status_lbl.config(text=msg)
        if self.config_path:
            self._sb_config_lbl.config(text=f"Config: {self.config_path.name}")
        if self.raw_path:
            self._sb_raw_lbl.config(text=f"Raw: {self.raw_path.name}")
        if self._modified:
            self._sb_modified_lbl.config(text="● Modified", foreground="#c07000")
        else:
            self._sb_modified_lbl.config(text="✓ Saved", foreground="#008000")

    def _set_modified(self, modified: bool = True) -> None:
        self._modified = modified
        self._update_status()

    # ── Blocks tab ────────────────────────────────────────────────────────────

    def _bind_scroll(self, widget: Any, canvas: Any) -> None:
        """Recursively bind mousewheel / trackpad scroll on *widget* to *canvas*.

        This makes scrolling work even when the cursor is over a child widget
        (label, slider, checkbox, etc.) rather than the bare canvas background.
        """
        def _on_wheel(e: Any) -> None:
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        widget.bind("<MouseWheel>", _on_wheel, add=True)
        widget.bind("<Button-4>", lambda _e: canvas.yview_scroll(-3, "units"), add=True)
        widget.bind("<Button-5>", lambda _e: canvas.yview_scroll(3, "units"), add=True)
        for child in widget.winfo_children():
            self._bind_scroll(child, canvas)

    def _build_blocks_tab(self, parent) -> None:
        tk, ttk = self.tk, self.ttk
        ttk.Label(
            parent,
            text="Toggle stages on/off.  Click a block name to edit its full YAML.",
            wraplength=300, font=("TkDefaultFont", 9), foreground="#666666",
        ).pack(anchor=tk.W, padx=6, pady=(4, 2))

        canvas = tk.Canvas(parent, highlightthickness=0)
        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        self._blocks_frame = ttk.Frame(canvas)
        # Keep scrollregion in sync whenever children resize
        self._blocks_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        # Store the window ID so we can keep its width equal to the canvas width
        _blk_win = canvas.create_window((0, 0), window=self._blocks_frame, anchor=tk.NW)
        # Resize the inner frame to always fill the canvas horizontally
        canvas.bind(
            "<Configure>",
            lambda e, c=canvas, wid=_blk_win: c.itemconfig(wid, width=e.width),
        )
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=2)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=2)

        self._blocks_canvas = canvas  # kept for recursive scroll-binding after rebuild

        canvas.bind("<Enter>", lambda _e: canvas.focus_set())
        canvas.bind(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )
        canvas.bind("<Button-4>", lambda _e: canvas.yview_scroll(-3, "units"))
        canvas.bind("<Button-5>", lambda _e: canvas.yview_scroll(3, "units"))

    def _rebuild_block_checkboxes(self) -> None:
        for w in self._blocks_frame.winfo_children():
            w.destroy()
        self._block_vars.clear()
        if not self.working_config:
            return
        for key in _discover_block_keys(self.working_config):
            var = self.tk.BooleanVar(
                value=bool(self.working_config[key].get("is_enable", False))
            )
            self._block_vars[key] = var
            row = self.ttk.Frame(self._blocks_frame)
            row.pack(fill=self.tk.X, anchor=self.tk.W)
            self.ttk.Checkbutton(
                row, variable=var,
                command=lambda k=key: self._sync_block(k),
            ).pack(side=self.tk.LEFT, anchor=self.tk.N)
            name_lbl = self.ttk.Label(row, text=_label_for_key(key), cursor="hand2")
            name_lbl.pack(side=self.tk.LEFT, padx=(2, 0), anchor=self.tk.W)
            for seq in ("<Button-1>", "<Button-3>"):
                name_lbl.bind(seq, lambda e, k=key: self._show_block_menu(e, k))
        # Propagate scroll events from every child widget to the canvas
        self._bind_scroll(self._blocks_frame, self._blocks_canvas)

    def _show_block_menu(self, event: Any, key: str) -> None:
        menu = self.tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label="Edit configuration…",
            command=lambda: self._edit_block_config(key),
        )
        try:
            menu.tk_popup(int(event.x_root), int(event.y_root))
        finally:
            try:
                menu.grab_release()
            except self.tk.TclError:
                pass

    def _edit_block_config(self, key: str) -> None:
        if not self.working_config or key not in self.working_config:
            return
        section = self.working_config[key]
        if not isinstance(section, dict):
            self.messagebox.showerror("Edit block", f"{key!r} is not a mapping.")
            return

        win = self.tk.Toplevel(self.root)
        win.title(f"Edit block: {key}")
        win.transient(self.root)
        win.grab_set()
        win.minsize(480, 360)
        win.geometry("640x520")

        self.ttk.Label(
            win,
            text=f'YAML for "{key}" — must stay a mapping (dict).',
            wraplength=600,
        ).pack(anchor=self.tk.W, padx=8, pady=(8, 4))

        outer = self.ttk.Frame(win)
        outer.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=4)
        txt = self.tk.Text(
            outer, wrap=self.tk.NONE, width=80, height=24, font=("TkFixedFont", 10)
        )
        scroll_y = self.ttk.Scrollbar(outer, orient=self.tk.VERTICAL, command=txt.yview)
        scroll_x = self.ttk.Scrollbar(outer, orient=self.tk.HORIZONTAL, command=txt.xview)
        txt.config(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        txt.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        txt.insert("1.0", _dump_block_yaml(section))

        btn_row = self.ttk.Frame(win)
        btn_row.pack(fill=self.tk.X, padx=8, pady=8)

        def apply_ok() -> None:
            raw = txt.get("1.0", "end-1c")
            try:
                loaded = yaml.safe_load(raw)
            except yaml.YAMLError as e:
                self.messagebox.showerror("YAML error", str(e))
                return
            if loaded is None:
                self.messagebox.showerror("YAML error", "Empty / null is not allowed.")
                return
            if not isinstance(loaded, dict):
                self.messagebox.showerror(
                    "YAML error", "Root must be a mapping (dict)."
                )
                return
            self.working_config[key] = copy.deepcopy(loaded)
            if key in self._block_vars:
                if "is_enable" in self.working_config[key]:
                    self._block_vars[key].set(
                        bool(self.working_config[key]["is_enable"])
                    )
                else:
                    self.working_config[key]["is_enable"] = bool(
                        self._block_vars[key].get()
                    )
            self._set_modified()
            self._refresh_params_from_config()
            self._schedule_reprocess()
            win.destroy()

        self.ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(
            side=self.tk.RIGHT, padx=(4, 0)
        )
        self.ttk.Button(btn_row, text="OK", command=apply_ok).pack(side=self.tk.RIGHT)

    def _sync_block(self, key: str) -> None:
        if self.working_config and key in self._block_vars:
            self.working_config[key]["is_enable"] = bool(self._block_vars[key].get())
            self._set_modified()
            self._schedule_reprocess()

    def _sync_checkboxes_from_working_config(self) -> None:
        if not self.working_config:
            return
        for key, var in self._block_vars.items():
            sec = self.working_config.get(key)
            if isinstance(sec, dict) and "is_enable" in sec:
                var.set(bool(sec["is_enable"]))

    # ── Parameters tab ────────────────────────────────────────────────────────

    def _build_params_tab(self, parent) -> None:
        """Build the scrollable container for parameter controls."""
        tk, ttk = self.tk, self.ttk
        ttk.Label(
            parent,
            text="Common parameters — sliders update the live config immediately.",
            wraplength=300, font=("TkDefaultFont", 9), foreground="#666666",
        ).pack(anchor=tk.W, padx=6, pady=(4, 2))

        self._params_canvas = tk.Canvas(parent, highlightthickness=0)
        scroll = ttk.Scrollbar(
            parent, orient=tk.VERTICAL, command=self._params_canvas.yview
        )
        self._params_inner = ttk.Frame(self._params_canvas)
        # Keep scrollregion in sync whenever children resize
        self._params_inner.bind(
            "<Configure>",
            lambda e: self._params_canvas.configure(
                scrollregion=self._params_canvas.bbox("all")
            ),
        )
        # Store window ID so we can keep its width equal to the canvas width
        _par_win = self._params_canvas.create_window(
            (0, 0), window=self._params_inner, anchor=tk.NW
        )
        # Resize the inner frame to always fill the canvas horizontally
        self._params_canvas.bind(
            "<Configure>",
            lambda e, wid=_par_win: self._params_canvas.itemconfig(wid, width=e.width),
        )
        self._params_canvas.configure(yscrollcommand=scroll.set)
        self._params_canvas.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=2
        )
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=2)

        self._params_canvas.bind(
            "<Enter>", lambda _e: self._params_canvas.focus_set()
        )
        self._params_canvas.bind(
            "<MouseWheel>",
            lambda e: self._params_canvas.yview_scroll(
                int(-1 * (e.delta / 120)), "units"
            ),
        )
        self._params_canvas.bind(
            "<Button-4>",
            lambda _e: self._params_canvas.yview_scroll(-3, "units"),
        )
        self._params_canvas.bind(
            "<Button-5>",
            lambda _e: self._params_canvas.yview_scroll(3, "units"),
        )

        ttk.Label(
            self._params_inner,
            text="Load a config file to enable parameter controls.",
            foreground="#888888", font=("TkDefaultFont", 9),
        ).pack(padx=8, pady=20)

    def _rebuild_params_tab(self) -> None:
        """Destroy and recreate all parameter widgets from working_config."""
        for w in self._params_inner.winfo_children():
            w.destroy()
        self._param_vars.clear()
        self._param_widgets.clear()

        if not self.working_config:
            self.ttk.Label(
                self._params_inner,
                text="Load a config file to enable parameter controls.",
                foreground="#888888",
            ).pack(padx=8, pady=20)
            return

        cfg = self.working_config

        def changed(*_args: Any) -> None:
            self._push_params_to_config()
            self._set_modified()
            self._redraw_crop_overlay()
            self._schedule_reprocess()

        # ── Digital Gain ──────────────────────────────────────────────────
        dg = cfg.get("digital_gain", {})
        if isinstance(dg, dict):
            sec = self._section("Digital Gain")

            gain_array: list = dg.get("gain_array", [1.0])
            max_idx = max(0, len(gain_array) - 1)
            cur_idx = max(0, min(int(dg.get("current_gain", 5)), max_idx))

            dg_auto_var = self.tk.BooleanVar(value=bool(dg.get("is_auto", False)))
            self._param_vars["digital_gain.is_auto"] = dg_auto_var
            self.ttk.Checkbutton(
                sec, text="Auto gain (AE feedback)", variable=dg_auto_var,
                command=changed,
            ).pack(anchor=self.tk.W, padx=4, pady=(2, 1))

            gain_row = self.ttk.Frame(sec)
            gain_row.pack(fill=self.tk.X)
            self.ttk.Label(
                gain_row, text="Gain index:", width=13, anchor=self.tk.W
            ).pack(side=self.tk.LEFT)
            gain_idx_var = self.tk.IntVar(value=cur_idx)
            self._param_vars["digital_gain.current_gain"] = gain_idx_var
            self.tk.Scale(
                gain_row, variable=gain_idx_var, from_=0, to=max_idx,
                orient=self.tk.HORIZONTAL, showvalue=True, resolution=1,
                length=130, command=changed,
            ).pack(side=self.tk.LEFT, fill=self.tk.X, expand=True)
            gain_val_lbl = self.ttk.Label(gain_row, text="", width=7)
            gain_val_lbl.pack(side=self.tk.LEFT, padx=2)
            self._param_widgets["digital_gain._gain_val_lbl"] = gain_val_lbl
            self._param_widgets["digital_gain._gain_array"] = gain_array
            self._param_widgets["digital_gain._gain_row"] = gain_row

            def _update_gain_label(*_: Any) -> None:
                arr = self._param_widgets.get("digital_gain._gain_array", [])
                lbl = self._param_widgets.get("digital_gain._gain_val_lbl")
                var = self._param_vars.get("digital_gain.current_gain")
                if arr and lbl is not None and var is not None:
                    idx = max(0, min(int(var.get()), len(arr) - 1))
                    lbl.config(text=f"= {arr[idx]}")

            gain_idx_var.trace_add("write", _update_gain_label)
            _update_gain_label()  # initialise label

            def _toggle_gain_row(*_: Any) -> None:
                row = self._param_widgets.get("digital_gain._gain_row")
                if row is None:
                    return
                if dg_auto_var.get():
                    row.pack_forget()
                else:
                    row.pack(fill=self.tk.X)
                changed()

            dg_auto_var.trace_add("write", _toggle_gain_row)
            if dg.get("is_auto", False):
                gain_row.pack_forget()

        # ── Crop ──────────────────────────────────────────────────────────
        crop = cfg.get("crop", {})
        if isinstance(crop, dict):
            sec = self._section("Crop")
            crop_enable_var = self.tk.BooleanVar(value=bool(crop.get("is_enable", False)))
            self._param_vars["crop.is_enable"] = crop_enable_var
            self.ttk.Checkbutton(
                sec,
                text="Enable crop",
                variable=crop_enable_var,
                command=changed,
            ).pack(anchor=self.tk.W, padx=4, pady=(2, 3))

            sensor = cfg.get("sensor_info", {})
            if not isinstance(sensor, dict):
                sensor = {}
            max_w = max(1, int(sensor.get("width", 8192) or 8192))
            max_h = max(1, int(sensor.get("height", 8192) or 8192))

            self._add_int_spinbox(
                sec, "X start:", "crop.crop_x_start",
                int(crop.get("crop_x_start", 0)), 0, max(0, max_w - 1), changed,
            )
            self._add_int_spinbox(
                sec, "Y start:", "crop.crop_y_start",
                int(crop.get("crop_y_start", 0)), 0, max(0, max_h - 1), changed,
            )
            self._add_int_spinbox(
                sec, "Width:", "crop.new_width",
                int(crop.get("new_width", max_w)), 1, max_w, changed,
            )
            self._add_int_spinbox(
                sec, "Height:", "crop.new_height",
                int(crop.get("new_height", max_h)), 1, max_h, changed,
            )

        # ── White Balance ─────────────────────────────────────────────────
        wb = cfg.get("white_balance", {})
        awb = cfg.get("auto_white_balance", {})
        if isinstance(wb, dict):
            sec = self._section("White Balance")
            awb_on = bool(awb.get("is_enable", False)) if isinstance(awb, dict) else False
            awb_note = self.ttk.Label(
                sec,
                text="AWB is enabled — gains are estimated after processing.",
                wraplength=280, font=("TkDefaultFont", 8), foreground="#2277bb",
            )
            self._param_widgets["wb._awb_note"] = awb_note
            if awb_on:
                awb_note.pack(anchor=self.tk.W, padx=4, pady=(0, 2))

            self._add_slider(
                sec, "R Gain:", "white_balance.r_gain",
                float(wb.get("r_gain", 1.0)), 0.1, 4.0, 0.05, changed,
            )
            self._add_slider(
                sec, "B Gain:", "white_balance.b_gain",
                float(wb.get("b_gain", 1.0)), 0.1, 4.0, 0.05, changed,
            )

        # ── Saturation ────────────────────────────────────────────────────
        cse = cfg.get("color_saturation_enhancement", {})
        if isinstance(cse, dict):
            sec = self._section("Saturation")
            self._add_slider(
                sec, "Saturation:", "color_saturation_enhancement.saturation_gain",
                float(cse.get("saturation_gain", 1.0)), 0.0, 3.0, 0.05, changed,
            )

        # ── Tone Mapping ──────────────────────────────────────────────────
        tm = cfg.get("tone_mapping", {})
        if isinstance(tm, dict):
            sec = self._section("Tone Mapping")
            self._add_dropdown(
                sec, "Mapper:", "tone_mapping.tone_mapper",
                str(tm.get("tone_mapper", "reinhard_integer")),
                _TONE_MAPPERS, changed,
            )
            before_var = self.tk.BooleanVar(
                value=bool(tm.get("tone_mapping_before_demosaic", True))
            )
            self._param_vars["tone_mapping.tone_mapping_before_demosaic"] = before_var
            self.ttk.Checkbutton(
                sec, text="Apply before demosaic", variable=before_var,
                command=changed,
            ).pack(anchor=self.tk.W, padx=4, pady=(2, 1))

        # ── Demosaic ──────────────────────────────────────────────────────
        dem = cfg.get("demosaic", {})
        if isinstance(dem, dict):
            sec = self._section("Demosaic")
            self._add_dropdown(
                sec, "Algorithm:", "demosaic.algorithm",
                str(dem.get("algorithm", "bilinear")),
                _DEMOSAIC_ALGORITHMS, changed,
            )

        # ── Gamma Correction ──────────────────────────────────────────────
        gmc = cfg.get("gamma_correction", {})
        if isinstance(gmc, dict):
            sec = self._section("Gamma Correction")
            self._add_dropdown(
                sec, "Curve:", "gamma_correction.curve",
                str(gmc.get("curve", "srgb")),
                _GAMMA_CURVES, changed,
            )
            gamma_row = self._add_slider(
                sec, "Gamma:", "gamma_correction.gamma",
                float(gmc.get("gamma", 2.2)), 0.5, 4.0, 0.05, changed,
            )
            self._param_widgets["gamma_correction._gamma_row"] = gamma_row

            def _toggle_gamma_row(*_: Any) -> None:
                row = self._param_widgets.get("gamma_correction._gamma_row")
                cv = self._param_vars.get("gamma_correction.curve")
                if row is None or cv is None:
                    return
                if cv.get() == "gamma":
                    row.pack(fill=self.tk.X, pady=1)
                else:
                    row.pack_forget()

            curve_var = self._param_vars.get("gamma_correction.curve")
            if curve_var is not None:
                curve_var.trace_add("write", _toggle_gamma_row)
            if gmc.get("curve", "srgb") != "gamma" and gamma_row is not None:
                gamma_row.pack_forget()

        # ── Sharpen ───────────────────────────────────────────────────────
        sha = cfg.get("sharpen", {})
        if isinstance(sha, dict):
            sec = self._section("Sharpen")
            self._add_slider(
                sec, "Strength:", "sharpen.sharpen_strength",
                float(sha.get("sharpen_strength", 1.0)), 0.0, 3.0, 0.1, changed,
            )
            self._add_slider(
                sec, "Sigma:", "sharpen.sharpen_sigma",
                float(sha.get("sharpen_sigma", 3.0)), 0.5, 8.0, 0.1, changed,
            )

        # ── Auto Exposure ─────────────────────────────────────────────────
        ae = cfg.get("auto_exposure", {})
        if isinstance(ae, dict):
            sec = self._section("Auto Exposure")

            # Explain which parameters apply in each mode
            mode = str(ae.get("exposure_correction_mode", "step"))
            dg_auto = bool(dg.get("is_auto", False)) if isinstance(dg, dict) else False
            ae_on = bool(ae.get("is_enable", False))

            if not ae_on or not dg_auto:
                note_txt = (
                    "⚠  AE inactive.\n"
                    "Enable auto_exposure.is_enable AND digital_gain.is_auto\n"
                    "(Blocks tab) for these controls to affect the image."
                )
                note_fg = "#c07000"
            elif mode == "direct":
                note_txt = (
                    "Mode: direct  —  Target luminance drives gain selection.\n"
                    "Center illum / Hist skew are not used in direct mode.\n"
                    "Adjust Target luminance to control brightness."
                )
                note_fg = "#2277bb"
            else:
                note_txt = (
                    "Mode: step  —  Center illum + Hist skew drive gain feedback.\n"
                    "The 3A convergence loop runs automatically when processing."
                )
                note_fg = "#2277bb"

            self.ttk.Label(
                sec, text=note_txt,
                wraplength=280, font=("TkDefaultFont", 8), foreground=note_fg,
            ).pack(anchor=self.tk.W, padx=4, pady=(2, 4))

            # target_luminance: the primary brightness target in "direct" mode.
            # Stored as a fraction [0,1] of the metered full scale (e.g. 0.5 = mid-grey).
            # Default 0.5; absent from most configs so we write it if missing.
            if "target_luminance" not in ae:
                ae["target_luminance"] = 0.5
            self._add_slider(
                sec, "Target lum:", "auto_exposure.target_luminance",
                float(ae.get("target_luminance", 0.5)), 0.05, 1.0, 0.01, changed,
            )

            self._add_slider(
                sec, "Ctr illum:", "auto_exposure.center_illuminance",
                float(ae.get("center_illuminance", 0.4)), 0.05, 0.95, 0.01, changed,
            )
            self._add_slider(
                sec, "Hist skew:", "auto_exposure.histogram_skewness",
                float(ae.get("histogram_skewness", 0.9)), 0.1, 2.0, 0.05, changed,
            )

        # ── LDCI ──────────────────────────────────────────────────────────
        ldci = cfg.get("ldci", {})
        if isinstance(ldci, dict):
            sec = self._section("LDCI")
            self._add_slider(
                sec, "Clip limit:", "ldci.clip_limit",
                float(ldci.get("clip_limit", 2.0)), 0.5, 10.0, 0.1, changed,
            )

        # ── Bayer Noise Reduction ─────────────────────────────────────────
        bnr = cfg.get("bayer_noise_reduction", {})
        if isinstance(bnr, dict):
            sec = self._section("Bayer Noise Reduction")
            fw_var = self.tk.IntVar(value=int(bnr.get("filter_window", 5)))
            self._param_vars["bayer_noise_reduction.filter_window"] = fw_var
            fw_row = self.ttk.Frame(sec)
            fw_row.pack(fill=self.tk.X, pady=2)
            self.ttk.Label(
                fw_row, text="Filter window:", width=13, anchor=self.tk.W
            ).pack(side=self.tk.LEFT)
            self.ttk.Spinbox(
                fw_row, from_=3, to=15, increment=2,
                textvariable=fw_var, width=5, command=changed,
            ).pack(side=self.tk.LEFT, padx=2)
            fw_var.trace_add("write", changed)

        # ── AWB Algorithm ─────────────────────────────────────────────────
        if isinstance(awb, dict) and awb:
            sec = self._section("Auto White Balance")
            self._add_dropdown(
                sec, "Algorithm:", "auto_white_balance.algorithm",
                str(awb.get("algorithm", "pca")),
                _AWB_ALGORITHMS, changed,
            )
            self._add_slider(
                sec, "PCA %:", "auto_white_balance.percentage",
                float(awb.get("percentage", 3.5)), 0.5, 10.0, 0.5, changed,
            )

        # Propagate scroll events from every child widget up to the canvas
        self._bind_scroll(self._params_inner, self._params_canvas)

    # ── Parameter widget factories ────────────────────────────────────────────

    def _section(self, title: str):
        """Create a labelled section frame inside the params inner container."""
        frame = self.ttk.LabelFrame(self._params_inner, text=title, padding=(6, 4))
        frame.pack(fill=self.tk.X, padx=4, pady=3)
        return frame

    def _add_slider(
        self,
        parent: Any,
        label: str,
        key: str,
        initial: float,
        from_: float,
        to: float,
        resolution: float,
        callback: Any,
    ):
        """Add a label + horizontal Scale. Returns the row frame."""
        tk, ttk = self.tk, self.ttk
        var = tk.DoubleVar(value=initial)
        self._param_vars[key] = var
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, width=13, anchor=tk.W).pack(side=tk.LEFT)
        tk.Scale(
            row, variable=var, from_=from_, to=to, resolution=resolution,
            orient=tk.HORIZONTAL, showvalue=True, length=140,
            command=callback,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        return row

    def _add_dropdown(
        self,
        parent: Any,
        label: str,
        key: str,
        initial: str,
        values: list[str],
        callback: Any,
    ):
        """Add a label + Combobox. Returns the StringVar."""
        tk, ttk = self.tk, self.ttk
        var = tk.StringVar(value=initial)
        self._param_vars[key] = var
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=13, anchor=tk.W).pack(side=tk.LEFT)
        cb = ttk.Combobox(
            row, textvariable=var, values=values, state="readonly", width=20
        )
        cb.pack(side=tk.LEFT, padx=2)
        cb.bind("<<ComboboxSelected>>", callback)
        return var

    def _add_int_spinbox(
        self,
        parent: Any,
        label: str,
        key: str,
        initial: int,
        min_value: int,
        max_value: int,
        callback: Any,
    ):
        tk, ttk = self.tk, self.ttk
        var = tk.IntVar(value=initial)
        self._param_vars[key] = var
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, width=13, anchor=tk.W).pack(side=tk.LEFT)
        spin = ttk.Spinbox(
            row,
            from_=min_value,
            to=max_value,
            increment=1,
            textvariable=var,
            width=10,
            command=callback,
        )
        spin.pack(side=tk.LEFT, padx=2)
        var.trace_add("write", callback)
        return row

    # ── Config sync ───────────────────────────────────────────────────────────

    def _push_params_to_config(self) -> None:
        """Write all param widget values into working_config."""
        if not self.working_config:
            return
        for key, var in self._param_vars.items():
            parts = key.split(".", 1)
            if len(parts) != 2:
                continue
            block, param = parts
            block_cfg = self.working_config.get(block)
            if not isinstance(block_cfg, dict):
                continue
            try:
                raw_val = var.get()
            except self.tk.TclError:
                continue
            existing = block_cfg.get(param)
            if isinstance(existing, bool):
                val: Any = bool(raw_val)
            elif isinstance(existing, int) and not isinstance(existing, bool):
                try:
                    val = int(round(float(raw_val)))
                except (TypeError, ValueError):
                    val = raw_val
            else:
                val = raw_val
            block_cfg[param] = val

    def _refresh_params_from_config(self) -> None:
        """Pull working_config values back into the widget variables (e.g. after AWB/AE feedback)."""
        if not self.working_config:
            return
        for key, var in self._param_vars.items():
            parts = key.split(".", 1)
            if len(parts) != 2:
                continue
            block, param = parts
            sec = self.working_config.get(block)
            if not isinstance(sec, dict) or param not in sec:
                continue
            val = sec[param]
            try:
                if isinstance(var, self.tk.BooleanVar):
                    var.set(bool(val))
                elif isinstance(var, self.tk.IntVar):
                    var.set(int(round(float(val))))
                elif isinstance(var, self.tk.DoubleVar):
                    var.set(float(val))
                elif isinstance(var, self.tk.StringVar):
                    var.set(str(val))
            except (TypeError, ValueError, self.tk.TclError):
                pass

    # ── Auto-process / debounce ───────────────────────────────────────────────

    def _schedule_reprocess(self) -> None:
        """Cancel any pending timer and start a new one for auto-reprocess."""
        if not self._auto_process_var.get():
            return
        if not self.raw_path or not self.working_config:
            return
        if self._reprocess_after_id is not None:
            try:
                self.root.after_cancel(self._reprocess_after_id)
            except Exception:
                pass
        self._reprocess_after_id = self.root.after(
            _DEBOUNCE_MS, self._trigger_auto_reprocess
        )

    def _trigger_auto_reprocess(self) -> None:
        self._reprocess_after_id = None
        if not self._processing:
            self._on_process()

    # ── File operations ───────────────────────────────────────────────────────

    def _load_config_path(self, path: Path) -> None:
        paths = pipeline_config_paths(path)
        self.working_config = load_merged_yaml(paths)
        self.config_path = path.resolve()
        self.save_path = path.resolve()
        self._modified = False
        self._rebuild_block_checkboxes()
        self._rebuild_params_tab()
        self._update_status(f"Loaded: {path.name}  ({len(paths)} file(s) merged)")

    def _open_raw(self) -> None:
        p = self.filedialog.askopenfilename(
            title="Open raw image",
            filetypes=[
                ("Raw / TIFF", "*.raw *.RAW *.tiff *.tif"),
                ("All files", "*.*"),
            ],
        )
        if not p:
            return
        self.raw_path = Path(p).resolve()
        if self.working_config is not None:
            self.working_config.setdefault("platform", {})["filename"] = (
                self.raw_path.name
            )
        self._update_status(f"Raw opened: {self.raw_path.name}")

    def _open_config(self) -> None:
        p = self.filedialog.askopenfilename(
            title="Open ISP YAML",
            initialdir=str(REPO_ROOT / "config"),
            filetypes=[("YAML", "*.yml *.yaml"), ("All files", "*.*")],
        )
        if not p:
            return
        try:
            self._load_config_path(Path(p))
        except Exception as e:
            self.messagebox.showerror(
                "Config error", f"{e}\n\n{traceback.format_exc()}"
            )

    def _save_config(self) -> None:
        path = self.save_path or self.config_path
        if not path:
            self._save_config_as()
            return
        self._write_yaml(path, show_dialog=True)

    def _save_config_as(self) -> None:
        p = self.filedialog.asksaveasfilename(
            title="Save ISP YAML",
            defaultextension=".yml",
            filetypes=[("YAML", "*.yml *.yaml"), ("All files", "*.*")],
        )
        if not p:
            return
        self.save_path = Path(p).resolve()
        self._write_yaml(self.save_path, show_dialog=True)

    def _write_yaml(self, path: Path, show_dialog: bool = False) -> bool:
        """Serialise working_config to *path*. Returns True on success."""
        if not self.working_config:
            self.messagebox.showwarning("Save", "No configuration loaded.")
            return False
        self._push_params_to_config()
        self._sync_checkboxes_from_working_config()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _yaml_safe_value(copy.deepcopy(self.working_config))
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                data, f,
                sort_keys=False, allow_unicode=True, default_flow_style=False,
            )
        self._set_modified(False)
        self._update_status(f"Config saved: {path.name}")
        if show_dialog:
            self.messagebox.showinfo("Saved", f"Config written to:\n{path}")
        return True

    def _save_output_image(self) -> None:
        if self._last_output_rgb is None:
            self.messagebox.showwarning(
                "Save image", "No preview yet — run Process first."
            )
            return
        initialfile = (
            f"{self.raw_path.stem}_isp.png" if self.raw_path else "output.png"
        )
        p = self.filedialog.asksaveasfilename(
            title="Save output image",
            initialfile=initialfile,
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("TIFF", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not p:
            return
        self._write_image(Path(p), show_dialog=True)

    def _write_image(self, path: Path, show_dialog: bool = False) -> bool:
        """Write *_last_output_rgb* to *path*. Returns True on success."""
        arr = self._last_output_rgb
        if arr is None:
            return False
        try:
            to_write = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            ext = path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                ok = cv2.imwrite(str(path), to_write, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                ok = cv2.imwrite(str(path), to_write)
            if not ok:
                raise RuntimeError("cv2.imwrite returned False.")
        except Exception as e:
            self.messagebox.showerror(
                "Save image", f"{e}\n\n{traceback.format_exc()}"
            )
            return False
        self._update_status(f"Image saved: {path.name}")
        if show_dialog:
            self.messagebox.showinfo("Saved", f"Image written to:\n{path}")
        return True

    def _save_all(self) -> None:
        """Save config (known path, no dialog) + output image (auto-named beside raw)."""
        if not self.working_config:
            self.messagebox.showwarning("Save All", "No configuration loaded.")
            return

        # ── Config ──
        cfg_path = self.save_path or self.config_path
        if cfg_path:
            ok = self._write_yaml(cfg_path, show_dialog=False)
        else:
            p = self.filedialog.asksaveasfilename(
                title="Save ISP YAML",
                defaultextension=".yml",
                filetypes=[("YAML", "*.yml *.yaml"), ("All files", "*.*")],
            )
            if not p:
                return
            self.save_path = Path(p).resolve()
            ok = self._write_yaml(self.save_path, show_dialog=False)
            cfg_path = self.save_path
        if not ok:
            return

        # ── Image ──
        if self._last_output_rgb is None:
            self.messagebox.showinfo(
                "Save All",
                f"Config saved to:\n{cfg_path}\n\n"
                "(No output image yet — run Process to generate one.)",
            )
            return

        if self.raw_path:
            img_path = self.raw_path.parent / f"{self.raw_path.stem}_isp.png"
        elif cfg_path:
            img_path = cfg_path.parent / "isp_output.png"
        else:
            self._save_output_image()
            return

        self._write_image(img_path, show_dialog=False)
        self.messagebox.showinfo(
            "Save All",
            f"Config saved to:\n{cfg_path}\n\nImage saved to:\n{img_path}",
        )

    # ── Pipeline runner ───────────────────────────────────────────────────────

    def _prepare_run_config(self) -> dict[str, Any]:
        if not self.working_config:
            raise RuntimeError("No configuration loaded.")
        self._push_params_to_config()
        self._sync_checkboxes_from_working_config()
        cfg = copy.deepcopy(self.working_config)
        plat = cfg.setdefault("platform", {})
        plat["disable_progress_bar"] = True
        plat["debug_enabled"] = False
        plat["plot_histograms"] = False
        plat["skip_disabled_modules"] = True
        if self.raw_path:
            plat["filename"] = self.raw_path.name

        # Enable the 3A convergence loop only for step mode.
        #
        # "direct" mode: gain is selected inside run_pipeline's own 2-pass loop
        #   (max_ae_passes=2).  Wrapping it with execute_with_3a_statistics adds
        #   unnecessary passes and previously caused a hang (see off-by-one fix in
        #   execute_with_3a_statistics).  render_3a stays False; direct mode works.
        #
        # "step" mode: skewness feedback adjusts the gain by ±1 per pass.  The only
        #   way centre-illuminance / histogram-skewness ever affect the rendered image
        #   is through the multi-pass loop (execute_with_3a_statistics).  Without it
        #   the feedback is computed and silently discarded every single run.
        ae_cfg = cfg.get("auto_exposure", {})
        dg_cfg = cfg.get("digital_gain", {})
        ae_mode = ae_cfg.get("exposure_correction_mode", "step") if isinstance(ae_cfg, dict) else "step"
        ae_active = (
            isinstance(ae_cfg, dict) and ae_cfg.get("is_enable", False)
            and isinstance(dg_cfg, dict) and dg_cfg.get("is_auto", False)
        )
        plat["render_3a"] = bool(ae_active and ae_mode != "direct")

        return cfg

    def _on_process(self) -> None:
        if self._processing:
            return
        if not self.raw_path or not self.raw_path.is_file():
            self.messagebox.showwarning("Process", "Open a raw file first.")
            return
        if not self.working_config:
            self.messagebox.showwarning("Process", "Open a config first.")
            return

        # Determine whether the 3A convergence loop will run (step mode only)
        ae_cfg = self.working_config.get("auto_exposure", {})
        dg_cfg = self.working_config.get("digital_gain", {})
        _ae_mode = ae_cfg.get("exposure_correction_mode", "step") if isinstance(ae_cfg, dict) else "step"
        _ae_loop = (
            isinstance(ae_cfg, dict) and ae_cfg.get("is_enable", False)
            and isinstance(dg_cfg, dict) and dg_cfg.get("is_auto", False)
            and _ae_mode != "direct"
        )

        self._processing = True
        self.process_btn.config(state="disabled")
        if _ae_loop:
            self._update_status("Processing (3A AE loop running — may take a moment)…")
        else:
            self._update_status("Processing…")

        def work() -> None:
            rgb: np.ndarray | None = None
            isp_done: BrilliantISP | None = None
            err: str | None = None
            try:
                cfg = self._prepare_run_config()
                out_dir = Path(tempfile.mkdtemp(prefix="isp_gui_out_"))
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yml", delete=False, encoding="utf-8"
                ) as tf:
                    yaml.safe_dump(
                        _yaml_safe_value(cfg), tf,
                        sort_keys=False, allow_unicode=True, default_flow_style=False,
                    )
                    tmp_cfg = tf.name
                try:
                    isp_done = BrilliantISP(
                        str(self.raw_path.parent), tmp_cfg,
                        outFileName="", output_path=str(out_dir),
                    )
                    isp_done.execute(img_path=self.raw_path.name)
                    rgb = isp_done.last_output_rgb
                finally:
                    Path(tmp_cfg).unlink(missing_ok=True)
                if rgb is None:
                    err = "Pipeline produced no preview image (last_output_rgb is None)."
            except Exception:
                err = traceback.format_exc()

            def finish() -> None:
                self._processing = False
                self.process_btn.config(state="normal")
                if err:
                    self._update_status("Error during processing.")
                    self.messagebox.showerror("Pipeline error", err)
                    return
                if (
                    isp_done is not None
                    and isp_done.c_yaml is not None
                    and self.working_config is not None
                ):
                    _merge_pipeline_feedback_into(self.working_config, isp_done.c_yaml)
                    self._refresh_params_from_config()
                if rgb is not None:
                    self._show_rgb(rgb)
                # Show converged gain index in status when AE loop ran
                if (
                    _ae_loop
                    and isp_done is not None
                    and isp_done.c_yaml is not None
                ):
                    dg_out = isp_done.c_yaml.get("digital_gain", {})
                    gain_arr = dg_out.get("gain_array", [])
                    idx = int(dg_out.get("current_gain", 0))
                    gain_val = gain_arr[idx] if 0 <= idx < len(gain_arr) else "?"
                    self._update_status(
                        f"Done — {self.raw_path.name}  "
                        f"[AE converged: gain index {idx} = ×{gain_val}]"
                    )
                else:
                    self._update_status(f"Done — {self.raw_path.name}")

            self.root.after(0, finish)

        threading.Thread(target=work, daemon=True).start()

    def _show_rgb(self, rgb: np.ndarray) -> None:
        self._ax.clear()
        self._ax.axis("off")
        display = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
        self._last_output_rgb = display.copy()
        self._image_height, self._image_width = display.shape[:2]
        if display.ndim == 2:
            self._image_artist = self._ax.imshow(display, cmap="gray", vmin=0, vmax=255)
        else:
            self._image_artist = self._ax.imshow(display)
        self._crop_overlay_artist = None
        self._redraw_crop_overlay()
        self._reset_image_view(redraw=False)
        self._mpl_canvas.draw()

    def _reset_image_view(self, redraw: bool = True) -> None:
        if self._image_width <= 0 or self._image_height <= 0:
            return
        self._ax.set_xlim(-0.5, self._image_width - 0.5)
        self._ax.set_ylim(self._image_height - 0.5, -0.5)
        self._pan_start = None
        if redraw:
            self._mpl_canvas.draw_idle()

    def _on_image_scroll(self, event: Any) -> None:
        if (
            self._image_artist is None
            or event.inaxes != self._ax
            or event.xdata is None
            or event.ydata is None
        ):
            return
        if event.button == "up":
            scale = 1.0 / self._zoom_factor
        elif event.button == "down":
            scale = self._zoom_factor
        else:
            return
        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        new_x0 = event.xdata - (event.xdata - x0) * scale
        new_x1 = event.xdata + (x1 - event.xdata) * scale
        new_y0 = event.ydata - (event.ydata - y0) * scale
        new_y1 = event.ydata + (y1 - event.ydata) * scale
        self._ax.set_xlim(*self._clamp_axis_limits(new_x0, new_x1, self._image_width))
        self._ax.set_ylim(*self._clamp_axis_limits(new_y0, new_y1, self._image_height))
        self._mpl_canvas.draw_idle()

    def _zoom_keyboard(self, scale: float) -> None:
        if self._image_artist is None or self._image_width <= 0 or self._image_height <= 0:
            return
        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        new_x0 = cx - (cx - x0) * scale
        new_x1 = cx + (x1 - cx) * scale
        new_y0 = cy - (cy - y0) * scale
        new_y1 = cy + (y1 - cy) * scale
        self._ax.set_xlim(*self._clamp_axis_limits(new_x0, new_x1, self._image_width))
        self._ax.set_ylim(*self._clamp_axis_limits(new_y0, new_y1, self._image_height))
        self._mpl_canvas.draw_idle()

    def _on_image_press(self, event: Any) -> None:
        if (
            self._image_artist is None
            or event.inaxes != self._ax
            or event.button != 1
            or event.xdata is None
            or event.ydata is None
        ):
            return
        self._pan_start = (
            event.xdata,
            event.ydata,
            self._ax.get_xlim(),
            self._ax.get_ylim(),
        )

    def _on_image_release(self, event: Any) -> None:
        del event
        self._pan_start = None

    def _on_image_motion(self, event: Any) -> None:
        if (
            self._pan_start is None
            or self._image_artist is None
            or event.inaxes != self._ax
            or event.xdata is None
            or event.ydata is None
        ):
            return
        start_x, start_y, start_xlim, start_ylim = self._pan_start
        dx = event.xdata - start_x
        dy = event.ydata - start_y
        self._ax.set_xlim(
            *self._clamp_axis_limits(start_xlim[0] - dx, start_xlim[1] - dx, self._image_width)
        )
        self._ax.set_ylim(
            *self._clamp_axis_limits(start_ylim[0] - dy, start_ylim[1] - dy, self._image_height)
        )
        self._mpl_canvas.draw_idle()

    def _redraw_crop_overlay(self) -> None:
        if self._crop_overlay_artist is not None:
            self._crop_overlay_artist.remove()
            self._crop_overlay_artist = None
        rect = self._get_crop_overlay_rect()
        if rect is None:
            self._mpl_canvas.draw_idle()
            return
        x0, y0, w, h = rect
        self._crop_overlay_artist = Rectangle(
            (x0, y0),
            w,
            h,
            fill=False,
            edgecolor="#00cc44",
            linewidth=1.8,
            linestyle="--",
        )
        self._ax.add_patch(self._crop_overlay_artist)
        self._mpl_canvas.draw_idle()

    def _get_crop_overlay_rect(self) -> tuple[float, float, float, float] | None:
        if (
            self.working_config is None
            or self._image_width <= 0
            or self._image_height <= 0
        ):
            return None
        crop = self.working_config.get("crop")
        if not isinstance(crop, dict) or not bool(crop.get("is_enable", False)):
            return None
        sensor = self.working_config.get("sensor_info", {})
        if not isinstance(sensor, dict):
            return None
        sensor_w = int(sensor.get("width", 0) or 0)
        sensor_h = int(sensor.get("height", 0) or 0)
        if sensor_w <= 0 or sensor_h <= 0:
            return None
        x_start = int(crop.get("crop_x_start", 0) or 0)
        y_start = int(crop.get("crop_y_start", 0) or 0)
        new_w = int(crop.get("new_width", sensor_w) or sensor_w)
        new_h = int(crop.get("new_height", sensor_h) or sensor_h)
        x_start = max(0, min(x_start, sensor_w - 1))
        y_start = max(0, min(y_start, sensor_h - 1))
        new_w = max(1, min(new_w, sensor_w - x_start))
        new_h = max(1, min(new_h, sensor_h - y_start))
        sx = self._image_width / float(sensor_w)
        sy = self._image_height / float(sensor_h)
        return (
            x_start * sx - 0.5,
            y_start * sy - 0.5,
            new_w * sx,
            new_h * sy,
        )

    @staticmethod
    def _clamp_axis_limits(v0: float, v1: float, size: int) -> tuple[float, float]:
        if size <= 0:
            return v0, v1
        if v0 > v1:
            lo, hi = v1, v0
            reverse = True
        else:
            lo, hi = v0, v1
            reverse = False
        span = hi - lo
        full_lo, full_hi = -0.5, size - 0.5
        full_span = full_hi - full_lo
        if span >= full_span:
            lo, hi = full_lo, full_hi
        else:
            if lo < full_lo:
                hi += full_lo - lo
                lo = full_lo
            if hi > full_hi:
                lo -= hi - full_hi
                hi = full_hi
        if reverse:
            return hi, lo
        return lo, hi

    # ── About dialog ──────────────────────────────────────────────────────────

    def _show_about(self) -> None:
        about = self.tk.Toplevel(self.root)
        about.title(_ABOUT_TITLE)
        about.transient(self.root)
        about.grab_set()
        about.resizable(True, True)
        about.minsize(460, 340)
        about.columnconfigure(0, weight=1)
        about.rowconfigure(0, weight=1)
        text = self.tk.Text(
            about, wrap=self.tk.WORD, width=60, height=24,
            padx=12, pady=12, font=("TkDefaultFont", 10),
            relief=self.tk.FLAT, borderwidth=0,
        )
        text.insert("1.0", _ABOUT_TEXT)
        text.config(state=self.tk.DISABLED)
        scroll = self.ttk.Scrollbar(about, orient=self.tk.VERTICAL, command=text.yview)
        text.config(yscrollcommand=scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.ttk.Button(about, text="Close", command=about.destroy).grid(
            row=1, column=0, columnspan=2, pady=10
        )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="BrilliantISP Tuning GUI")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Initial YAML config (e.g. config/AD_cam.yml).",
    )
    args = parser.parse_args()

    initial_cfg = args.config
    if initial_cfg is None:
        cand = REPO_ROOT / "config" / "AD_cam.yml"
        if cand.is_file():
            initial_cfg = cand

    import tkinter as tk

    root = tk.Tk()
    ISPTuningApp(root, initial_cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
