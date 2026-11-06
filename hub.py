"""IRL Voice Changer: unified desktop launcher for five upstream AI VC engines.

This hub does not pretend that incompatible models share one inference API.
Each button starts real third-party software in a dedicated local environment.
"""
from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

from engines import (
    ENGINES, HOME, engine_is_installed, launch_args, load_settings,
    open_upstream, save_settings, setup_engine, source_dir, stop_process,
)

BG = "#090f19"
PANEL = "#111c2b"
CARD = "#182639"
BORDER = "#28394d"
TEXT = "#eef5fd"
MUTED = "#9eb0c4"
CYAN = "#42d8d5"
BLUE = "#56a8ff"
RED = "#ff8394"
YELLOW = "#eacb75"

class Hub:
    def __init__(self, root):
        self.root = root
        self.root.title("IRL Voice Changer · Engine Studio")
        self.root.geometry("1130x770")
        self.root.minsize(910, 630)
        self.root.configure(bg=BG)
        self.settings = load_settings()
        self.events = queue.Queue()
        self.selected = tk.StringVar(value="beatrice")
        self.cuda = tk.BooleanVar(value=bool(self.settings.get("gpu", False)))
        self.cpu = tk.BooleanVar(value=bool(self.settings.get("cpu", True)))
        self.mean_model = tk.StringVar(value=self.settings.get("mean_model", "40ms"))
        self.reference = tk.StringVar(value=self.settings.get("reference", ""))
        self.source_file = tk.StringVar(value=self.settings.get("source_file", ""))
        self.output = tk.StringVar(value=self.settings.get("output", str(HOME / "outputs")))
        self.status = tk.StringVar(value="Choose an engine, then set it up or launch it.")
        self.process = None
        self.installing = False
        self.buttons = {}
        self.engine_rows = {}
        self._theme()
        self._build()
        self.select_engine("beatrice")
        self.root.after(125, self.pump)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _theme(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", PANEL)], foreground=[("active", TEXT)])
        style.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=TEXT,
                        arrowcolor=TEXT, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)

    def label(self, parent, value, size=10, color=TEXT, weight="normal", bg=None, **kwargs):
        return tk.Label(parent, text=value, fg=color, bg=bg or BG,
                        font=("Segoe UI", size, weight), anchor="w", **kwargs)

    def button(self, parent, value, command, *, accent=False, small=False):
        b = tk.Button(parent, text=value, command=command, relief="flat", borderwidth=0,
                      bg=CYAN if accent else CARD, fg=BG if accent else TEXT,
                      activebackground=BLUE if accent else BORDER, activeforeground=BG if accent else TEXT,
                      font=("Segoe UI Semibold", 10 if small else 11),
                      padx=16 if not small else 11, pady=11 if not small else 8,
                      cursor="hand2")
        return b

    def _build(self):
        title = tk.Frame(self.root, bg=BG)
        title.pack(fill="x", padx=28, pady=(22, 16))
        self.label(title, "◉  IRL / VOICE CHANGER", 23, TEXT, "bold").pack(side="left")
        self.label(title, "FIVE ENGINES  /  ONE CONTROL CENTER", 10, CYAN, "bold").pack(side="right", pady=10)
        subtitle = self.label(self.root, "Run the original model engines locally · Separate dependencies · No mock audio", 11, MUTED)
        subtitle.pack(anchor="w", padx=30, pady=(0, 16))

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=22)
        nav = tk.Frame(main, bg=PANEL, width=238, highlightbackground=BORDER, highlightthickness=1)
        nav.pack(side="left", fill="y", padx=(0, 14))
        nav.pack_propagate(False)
        self.label(nav, "VOICE MODELS", 10, MUTED, "bold", bg=PANEL).pack(anchor="w", padx=17, pady=(22, 13))
        for key, spec in ENGINES.items():
            holder = tk.Frame(nav, bg=PANEL)
            holder.pack(fill="x", padx=9, pady=4)
            b = tk.Button(holder, text=spec.title + "\n" + spec.mode,
                          font=("Segoe UI", 10, "bold"), anchor="w", justify="left",
                          padx=14, pady=12, wraplength=194, relief="flat", bd=0,
                          bg=PANEL, fg=TEXT, activebackground=CARD, activeforeground=CYAN,
                          cursor="hand2", command=lambda k=key: self.select_engine(k))
            b.pack(fill="x")
            self.engine_rows[key] = b
        self.label(nav, "RUNS ON YOUR COMPUTER", 9, MUTED, "bold", bg=PANEL).pack(side="bottom", padx=14, pady=18)

        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        panel = tk.Frame(right, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        panel.pack(fill="x", pady=(0, 12))
        self.headline = self.label(panel, "", 21, TEXT, "bold", bg=PANEL)
        self.headline.pack(anchor="w", padx=23, pady=(20, 2))
        self.tagline = self.label(panel, "", 11, CYAN, bg=PANEL)
        self.tagline.pack(anchor="w", padx=23)
        self.description = self.label(panel, "", 10, MUTED, bg=PANEL, wraplength=750, justify="left")
        self.description.pack(anchor="w", fill="x", padx=23, pady=(11, 9))
        self.mode_label = self.label(panel, "", 10, YELLOW, bg=PANEL)
        self.mode_label.pack(anchor="w", padx=23, pady=(0, 17))

        controls = tk.Frame(right, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        controls.pack(fill="x", pady=(0, 12))
        row = tk.Frame(controls, bg=PANEL)
        row.pack(fill="x", padx=18, pady=(17, 11))
        self.buttons["setup"] = self.button(row, "Set up engine", self.setup, accent=True)
        self.buttons["setup"].pack(side="left", padx=(0, 9))
        self.buttons["start"] = self.button(row, "▶  Start engine", self.start)
        self.buttons["start"].pack(side="left", padx=(0, 9))
        self.buttons["stop"] = self.button(row, "■  Stop", self.stop, small=True)
        self.buttons["stop"].pack(side="left")
        self.button(row, "Upstream / Help ↗", self.upstream, small=True).pack(side="right")
        self.buttons["beatrice"] = self.button(controls, "Select official VCClient EXE…", self.browse_client, small=True)
        self.buttons["beatrice"].pack(anchor="w", padx=18, pady=(0, 11))
        options = tk.Frame(controls, bg=PANEL)
        options.pack(fill="x", padx=18, pady=(0, 12))
        ttk.Checkbutton(options, text="Install NVIDIA/CUDA dependencies", variable=self.cuda,
                        command=self.persist).pack(side="left")
        ttk.Checkbutton(options, text="Run MeanVC2 on CPU", variable=self.cpu,
                        command=self.persist).pack(side="left", padx=(19, 0))
        self.label(options, "MeanVC2:", bg=PANEL, color=MUTED).pack(side="left", padx=(18, 5))
        self.modelbox = ttk.Combobox(options, textvariable=self.mean_model,
                                     values=["40ms", "120ms"], width=8, state="readonly")
        self.modelbox.pack(side="left")
        self.modelbox.bind("<<ComboboxSelected>>", lambda _: self.persist())

        paths = tk.Frame(right, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        paths.pack(fill="x", pady=(0, 12))
        self.reference_row = self.file_row(paths, "Target voice WAV (MeanVC2)", self.reference, self.browse_reference)
        self.source_row = self.file_row(paths, "Input audio to convert (LLVC)", self.source_file, self.browse_audio)
        self.output_row = self.file_row(paths, "Converted file output folder", self.output, self.browse_output)
        self.label(paths, "For other live engines, choose microphone, target voice and output device inside the upstream program.",
                   9, MUTED, bg=PANEL, wraplength=690, justify="left").pack(anchor="w", padx=18, pady=(4, 14))

        log_head = tk.Frame(right, bg=BG)
        log_head.pack(fill="x")
        self.label(log_head, "ACTIVITY / INSTALL LOG", 10, MUTED, "bold").pack(side="left")
        self.button(log_head, "Open data folder", self.open_data, small=True).pack(side="right")
        logs = tk.Frame(right, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        logs.pack(fill="both", expand=True, pady=(8, 8))
        self.log = tk.Text(logs, bg=CARD, fg=TEXT, insertbackground=TEXT, relief="flat",
                           font=("Consolas", 9), padx=12, pady=10, wrap="word", height=7,
                           state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(logs, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)
        self.label(self.root, "Pocket TTS microphone demo (separate browser example): use the Pocket Demo page in this repository.",
                   9, MUTED).pack(anchor="w", padx=27, pady=(10, 2))
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=27, pady=(0, 15))
        self.label(bottom, "STATUS  •  ", 9, CYAN, "bold").pack(side="left")
        self.label(bottom, "", 10, TEXT, bg=BG, textvariable=self.status).pack(side="left")
        self.button(bottom, "Pocket Demo ↗", lambda: webbrowser.open(
            "https://tbenitz.github.io/IRL-Voice-Changer/pocket-demo/"), small=True).pack(side="right")

    def file_row(self, parent, heading, variable, command):
        line = tk.Frame(parent, bg=PANEL)
        line.pack(fill="x", padx=18, pady=4)
        self.label(line, heading, 9, MUTED, bg=PANEL).pack(anchor="w")
        inner = tk.Frame(line, bg=PANEL)
        inner.pack(fill="x")
        entry = tk.Entry(inner, textvariable=variable, bg=CARD, fg=TEXT, insertbackground=TEXT,
                         relief="flat", font=("Segoe UI", 10))
        entry.pack(side="left", fill="x", expand=True, ipady=7)
        self.button(inner, "Browse…", command, small=True).pack(side="left", padx=(8, 0))
        return line

    def select_engine(self, key):
        self.selected.set(key)
        spec = ENGINES[key]
        self.headline.configure(text=spec.title)
        self.tagline.configure(text=spec.tagline)
        self.description.configure(text=spec.information)
        self.mode_label.configure(text=spec.mode)
        for name, row in self.engine_rows.items():
            row.configure(bg=CARD if name == key else PANEL, fg=CYAN if name == key else TEXT)
        self.reference_row.pack_forget()
        self.source_row.pack_forget()
        self.output_row.pack_forget()
        if key == "meanvc2":
            self.reference_row.pack(fill="x", padx=18, pady=4)
        if key == "llvc":
            self.source_row.pack(fill="x", padx=18, pady=4)
            self.output_row.pack(fill="x", padx=18, pady=4)
        if key == "beatrice":
            self.buttons["beatrice"].pack(anchor="w", padx=18, pady=(0, 11))
        else:
            self.buttons["beatrice"].pack_forget()
        self.status.set(("Ready to launch" if engine_is_installed(key, self.settings) else "Setup needed") +
                        " · " + spec.title)
        self.refresh()

    def write_log(self, line):
        self.events.put(("log", str(line)))

    def append_log(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 1300:
            self.log.delete("1.0", "250.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def refresh(self):
        busy = self.installing or self.process is not None and self.process.poll() is None
        key = self.selected.get()
        self.buttons["setup"].configure(state="disabled" if busy else "normal",
                                        text="Official download ↗" if key == "beatrice" else "Set up engine")
        self.buttons["start"].configure(state="normal" if not busy and engine_is_installed(key, self.settings) else "disabled")
        self.buttons["stop"].configure(state="normal" if self.process is not None and self.process.poll() is None else "disabled")

    def pump(self):
        try:
            for _ in range(65):
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.append_log(payload)
                elif kind == "done":
                    self.installing = False
                    self.status.set(payload)
                    self.refresh()
                elif kind == "started":
                    self.status.set(payload)
                    self.refresh()
        except queue.Empty:
            pass
        if self.process and self.process.poll() is not None:
            code = self.process.returncode
            self.append_log(f"Engine process exited: {code}")
            self.status.set("Engine stopped" if code == 0 else f"Engine exited with status {code}; see log.")
            self.process = None
            self.refresh()
        self.root.after(125, self.pump)

    def persist(self):
        self.settings.update(gpu=bool(self.cuda.get()), cpu=bool(self.cpu.get()),
                             mean_model=self.mean_model.get(),
                             reference=self.reference.get(), source_file=self.source_file.get(),
                             output=self.output.get())
        save_settings(self.settings)

    def setup(self):
        key = self.selected.get()
        if key == "beatrice":
            open_upstream(key)
            return
        self.persist()
        self.installing = True
        self.status.set("Setting up " + ENGINES[key].title + " ...")
        self.refresh()
        def task():
            try:
                setup_engine(key, self.write_log, bool(self.cuda.get()))
                message = "Setup complete · " + ENGINES[key].title
            except Exception as exc:
                self.write_log(f"SETUP ERROR: {exc}")
                message = "Setup failed · " + str(exc)
            self.events.put(("done", message))
        threading.Thread(target=task, daemon=True).start()

    def start(self):
        self.persist()
        key = self.selected.get()
        try:
            args, cwd = launch_args(key, self.settings, reference=self.reference.get(),
                                   input_file=self.source_file.get(), cpu=self.cpu.get(),
                                   mean_model=self.mean_model.get(), output_dir=self.output.get())
            self.write_log("$ " + " ".join(args))
            self.process = subprocess.Popen(args, cwd=str(cwd), stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            errors="replace", bufsize=1)
            process = self.process
            def reader():
                try:
                    for line in process.stdout:
                        self.write_log(line.rstrip())
                except Exception as e:
                    self.write_log("Output reader: " + str(e))
            threading.Thread(target=reader, daemon=True).start()
            self.status.set("Running " + ENGINES[key].title + " · choose voice/audio devices in upstream GUI")
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Engine could not start", str(exc))
            self.write_log("START ERROR: " + str(exc))

    def stop(self):
        if self.process and self.process.poll() is None:
            self.write_log("Stopping engine and child processes…")
            stop_process(self.process)

    def upstream(self):
        open_upstream(self.selected.get())

    def browse_client(self):
        path = filedialog.askopenfilename(title="Select downloaded official VCClient EXE",
                                          filetypes=[("Windows executables", "*.exe"), ("All files", "*.*")])
        if path:
            self.settings["beatrice_exe"] = path
            save_settings(self.settings)
            self.status.set("VCClient EXE selected")
            self.refresh()

    def browse_reference(self):
        value = filedialog.askopenfilename(title="Reference voice WAV",
                                           filetypes=[("Wave files", "*.wav"), ("Audio files", "*.*")])
        if value:
            self.reference.set(value)
            self.persist()

    def browse_audio(self):
        value = filedialog.askopenfilename(title="LLVC input audio",
                                           filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")])
        if value:
            self.source_file.set(value)
            self.persist()

    def browse_output(self):
        value = filedialog.askdirectory(title="Converted file output folder")
        if value:
            self.output.set(value)
            self.persist()

    def open_data(self):
        HOME.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(HOME)
        else:
            webbrowser.open(HOME.as_uri())

    def close(self):
        if self.installing and not messagebox.askyesno("Setup running", "An install is running. Close anyway?"):
            return
        self.persist()
        self.stop()
        self.root.destroy()

def main():
    root = tk.Tk()
    Hub(root)
    root.mainloop()

if __name__ == "__main__":
    main()
