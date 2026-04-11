#!/usr/bin/env python3
"""
Basic ISP tuning GUI: open raw + config, preview processed output, toggle pipeline blocks,
save YAML, and save the preview image (File → Save output image…). Click a block name for a
menu to edit that block’s YAML. The merged YAML in memory is what Process uses; checkboxes
only mirror is_enable (toggling a checkbox updates the YAML). After Process, only pipeline
feedback (WB gains, AE gain index, inferred raw size) is merged in. Help → About…
describes the workflow and documentation.

Run from repo root:  python tools/isp_tuning_gui.py
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
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

from brilliant_isp import BrilliantISP
from util.config_merge import load_merged_yaml, pipeline_config_paths

# Pipeline block sections (order); tone-mapper sub-sections (hdr_durand, aces, …) are excluded.
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

_ABOUT_TITLE = "About BrilliantISP Tuning"

_ABOUT_TEXT = (
    "BrilliantISP — Tuning GUI\n\n"
    "BrilliantISP is developed by Brian Deegan; portions of the pipeline derive from "
    "the Infinite-ISP implementation (10xEngineers).\n\n"
    "Preview raw frames through the ISP pipeline, flip pipeline blocks on or off "
    "(YAML is_enable), and save the working configuration or the preview image.\n\n"
    "Workflow\n"
    "  • File → Open config… — loads a YAML; *_cam.yml files merge with config/base_hdr.yml.\n"
    "  • File → Open raw… — choose a frame; the folder is the data path.\n"
    "  • Adjust blocks, then Process — updates the preview from the pipeline.\n"
    "  • Save config / Save output image — writes YAML or PNG (JPEG/TIFF) from the preview.\n\n"
    f"Project: {REPO_ROOT}\n"
    "Docs: docs/ISP_BLOCKS_AND_TUNING.md\n\n"
    "BrilliantISP — Brian Deegan (based in part on 10xEngineers / Infinite-ISP)."
)


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
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def _merge_pipeline_feedback_into(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """
    Copy pipeline-run outputs into dst without replacing user-edited blocks wholesale.
    src is typically isp.c_yaml after execute().
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
        k
        for k in _BLOCK_ORDER
        if k in cfg and isinstance(cfg[k], dict) and "is_enable" in cfg[k]
    ]
    extra = sorted(
        k
        for k in cfg
        if k not in ordered
        and k not in _TONE_MAPPER_PARAM_KEYS
        and k not in ("platform", "sensor_info")
        and isinstance(cfg[k], dict)
        and "is_enable" in cfg[k]
    )
    return ordered + extra


def _label_for_key(key: str) -> str:
    return key.replace("_", " ").strip().title()


class ISPTuningApp:
    def __init__(self, root, default_config: Path | None) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.ttk = ttk

        self.root = root
        root.title("BrilliantISP — tuning")
        root.minsize(900, 600)

        self.raw_path: Path | None = None
        self.config_path: Path | None = None
        self.save_path: Path | None = None
        self.working_config: dict[str, Any] | None = None
        self._block_vars: dict[str, tk.BooleanVar] = {}
        self._processing = False
        self._last_output_rgb: np.ndarray | None = None

        menubar = tk.Menu(root)
        root.config(menu=menubar)
        file_m = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_m)
        file_m.add_command(label="Open raw…", command=self._open_raw)
        file_m.add_command(label="Open config…", command=self._open_config)
        file_m.add_separator()
        file_m.add_command(label="Save config", command=self._save_config)
        file_m.add_command(label="Save config as…", command=self._save_config_as)
        file_m.add_command(label="Save output image…", command=self._save_output_image)
        file_m.add_separator()
        file_m.add_command(label="Exit", command=root.quit)

        help_m = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_m)
        help_m.add_command(label="About…", command=self._show_about)

        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=280)
        main.add(left, weight=0)

        right = ttk.Frame(main)
        main.add(right, weight=1)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, padx=8, pady=8)
        self.process_btn = ttk.Button(btn_row, text="Process", command=self._on_process)
        self.process_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(left, text="Pipeline blocks (is_enable)").pack(anchor=tk.W, padx=8)
        ttk.Label(
            left,
            text="Click a block name for the menu (edit YAML).",
            wraplength=260,
            font=("TkDefaultFont", 9),
        ).pack(anchor=tk.W, padx=8, pady=(0, 4))
        canvas = tk.Canvas(left, highlightthickness=0)
        scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=canvas.yview)
        self._blocks_frame = ttk.Frame(canvas)
        self._blocks_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._blocks_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=4)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=4)

        def _scroll_canvas(delta: int) -> None:
            canvas.yview_scroll(delta, "units")

        canvas.bind("<Enter>", lambda _e: canvas.focus_set())
        canvas.bind("<MouseWheel>", lambda e: _scroll_canvas(int(-1 * (e.delta / 120))))
        canvas.bind("<Button-4>", lambda _e: _scroll_canvas(-3))
        canvas.bind("<Button-5>", lambda _e: _scroll_canvas(3))

        self._status = ttk.Label(left, text="Open a raw file and config.", wraplength=260)
        self._status.pack(fill=tk.X, padx=8, pady=8)

        self._fig = Figure(figsize=(7, 6), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.axis("off")
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if default_config and default_config.is_file():
            self._load_config_path(default_config)

    def _show_about(self) -> None:
        about = self.tk.Toplevel(self.root)
        about.title(_ABOUT_TITLE)
        about.transient(self.root)
        about.grab_set()
        about.resizable(True, True)
        about.minsize(420, 280)
        about.columnconfigure(0, weight=1)
        about.rowconfigure(0, weight=1)
        text = self.tk.Text(
            about,
            wrap=self.tk.WORD,
            width=56,
            height=18,
            padx=12,
            pady=12,
            font=("TkDefaultFont", 10),
            relief=self.tk.FLAT,
            borderwidth=0,
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

    def _load_config_path(self, path: Path) -> None:
        paths = pipeline_config_paths(path)
        self.working_config = load_merged_yaml(paths)
        self.config_path = path.resolve()
        self.save_path = path.resolve()
        self._rebuild_block_checkboxes()
        self._status.config(text=f"Config: {path.name}\nMerged {len(paths)} file(s).")

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
                row,
                variable=var,
                command=lambda k=key: self._sync_block(k),
            ).pack(side=self.tk.LEFT, anchor=self.tk.N)
            name_lbl = self.ttk.Label(
                row,
                text=_label_for_key(key),
                cursor="hand2",
            )
            name_lbl.pack(side=self.tk.LEFT, padx=(2, 0), anchor=self.tk.W)
            for seq in ("<Button-1>", "<Button-3>"):
                name_lbl.bind(
                    seq,
                    lambda e, k=key: self._show_block_menu(e, k),
                )

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
            self.messagebox.showerror("Edit block", f"{key!r} is not a mapping in config.")
            return

        win = self.tk.Toplevel(self.root)
        win.title(f"Edit: {key}")
        win.transient(self.root)
        win.grab_set()
        win.minsize(480, 360)
        win.geometry("640x520")

        self.ttk.Label(
            win,
            text=f"YAML for “{key}” (must stay a mapping).",
            wraplength=600,
        ).pack(anchor=self.tk.W, padx=8, pady=(8, 4))

        outer = self.ttk.Frame(win)
        outer.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=4)
        txt = self.tk.Text(outer, wrap=self.tk.NONE, width=80, height=24, font=("TkFixedFont", 10))
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
                self.messagebox.showerror("YAML error", "Empty or null is not allowed.")
                return
            if not isinstance(loaded, dict):
                self.messagebox.showerror(
                    "YAML error", "Root must be a mapping (dict), e.g. is_enable: true"
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
            win.destroy()

        def cancel() -> None:
            win.destroy()

        self.ttk.Button(btn_row, text="Cancel", command=cancel).pack(side=self.tk.RIGHT, padx=(4, 0))
        self.ttk.Button(btn_row, text="OK", command=apply_ok).pack(side=self.tk.RIGHT)

    def _sync_block(self, key: str) -> None:
        if self.working_config and key in self._block_vars:
            self.working_config[key]["is_enable"] = bool(self._block_vars[key].get())

    def _sync_checkboxes_from_working_config(self) -> None:
        """Match checkbox state to working_config (YAML / block editor is authoritative)."""
        if not self.working_config:
            return
        for key, var in self._block_vars.items():
            sec = self.working_config.get(key)
            if isinstance(sec, dict) and "is_enable" in sec:
                var.set(bool(sec["is_enable"]))

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
            self.working_config.setdefault("platform", {})["filename"] = self.raw_path.name
        self._status.config(
            text=f"Raw: {self.raw_path.name}\nFolder: {self.raw_path.parent}"
        )

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
            self.messagebox.showerror("Config error", f"{e}\n\n{traceback.format_exc()}")

    def _save_config(self) -> None:
        path = self.save_path or self.config_path
        if not path:
            self._save_config_as()
            return
        self._write_yaml(path)

    def _save_config_as(self) -> None:
        p = self.filedialog.asksaveasfilename(
            title="Save ISP YAML",
            defaultextension=".yml",
            filetypes=[("YAML", "*.yml *.yaml"), ("All files", "*.*")],
        )
        if not p:
            return
        self.save_path = Path(p).resolve()
        self._write_yaml(self.save_path)

    def _write_yaml(self, path: Path) -> None:
        if not self.working_config:
            self.messagebox.showwarning("Save", "No configuration loaded.")
            return
        self._sync_checkboxes_from_working_config()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _yaml_safe_value(copy.deepcopy(self.working_config))
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        self._status.config(text=f"Saved: {path}")
        self.messagebox.showinfo("Save", f"Wrote {path}")

    def _save_output_image(self) -> None:
        if self._last_output_rgb is None:
            self.messagebox.showwarning(
                "Save image",
                "No preview yet. Run Process to generate an output image.",
            )
            return
        initialfile = None
        if self.raw_path:
            initialfile = f"{self.raw_path.stem}_isp.png"
        p = self.filedialog.asksaveasfilename(
            title="Save output image",
            initialfile=initialfile or "",
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
        path = Path(p)
        arr_u8 = self._last_output_rgb
        try:
            if arr_u8.ndim == 2:
                to_write = arr_u8
            else:
                to_write = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2BGR)
            ext = path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                ok = cv2.imwrite(
                    str(path), to_write, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
            else:
                ok = cv2.imwrite(str(path), to_write)
            if not ok:
                raise RuntimeError("cv2.imwrite failed (path or format not supported).")
        except Exception as e:
            self.messagebox.showerror(
                "Save image", f"{e}\n\n{traceback.format_exc()}"
            )
            return
        self._status.config(text=f"Saved image: {path.name}")
        self.messagebox.showinfo("Save image", f"Wrote {path}")

    def _prepare_run_config(self) -> dict[str, Any]:
        if not self.working_config:
            raise RuntimeError("No configuration loaded.")
        # Do not push checkboxes → YAML here: stale checkboxes would overwrite is_enable
        # after block-editor changes (e.g. color_saturation_enhancement would stay disabled).
        self._sync_checkboxes_from_working_config()
        cfg = copy.deepcopy(self.working_config)
        plat = cfg.setdefault("platform", {})
        plat["disable_progress_bar"] = True
        plat["debug_enabled"] = False
        plat["plot_histograms"] = False
        plat["render_3a"] = False
        plat["skip_disabled_modules"] = True
        if self.raw_path:
            plat["filename"] = self.raw_path.name
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

        self._processing = True
        self.process_btn.config(state="disabled")
        self._status.config(text="Processing…")

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
                        _yaml_safe_value(cfg),
                        tf,
                        sort_keys=False,
                        allow_unicode=True,
                        default_flow_style=False,
                    )
                    tmp_cfg = tf.name
                try:
                    isp_done = BrilliantISP(
                        str(self.raw_path.parent),
                        tmp_cfg,
                        outFileName="",
                        output_path=str(out_dir),
                    )
                    isp_done.execute(img_path=self.raw_path.name)
                    rgb = isp_done.last_output_rgb
                finally:
                    Path(tmp_cfg).unlink(missing_ok=True)
                if rgb is None and err is None:
                    err = "Pipeline produced no preview image (last_output_rgb is None)."
            except Exception:
                err = traceback.format_exc()

            def finish() -> None:
                self._processing = False
                self.process_btn.config(state="normal")
                if err:
                    self._status.config(text="Error.")
                    self.messagebox.showerror("Pipeline error", err)
                    return
                if (
                    isp_done is not None
                    and isp_done.c_yaml is not None
                    and self.working_config is not None
                ):
                    _merge_pipeline_feedback_into(
                        self.working_config, isp_done.c_yaml
                    )
                if rgb is not None:
                    self._show_rgb(rgb)
                self._status.config(text=f"Done — {self.raw_path.name}")

            self.root.after(0, finish)

        import threading

        threading.Thread(target=work, daemon=True).start()

    def _show_rgb(self, rgb: np.ndarray) -> None:
        self._ax.clear()
        self._ax.axis("off")
        if rgb.ndim == 2:
            display = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
            self._last_output_rgb = display.copy()
            self._ax.imshow(display, cmap="gray", vmin=0, vmax=255)
        else:
            display = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
            self._last_output_rgb = display.copy()
            self._ax.imshow(display)
        self._fig.tight_layout()
        self._canvas.draw()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Initial YAML config (e.g. config/AD_cam.yml). Default: config/AD_cam.yml if present.",
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
