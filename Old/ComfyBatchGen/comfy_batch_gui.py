#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import threading
import queue
import time
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import requests
import copy

# ---------------- Dark theme palette ----------------
BG          = "#1e1e1e"
PANEL_BG    = "#1e1e1e"
INPUT_BG    = "#2d2d2d"
FG          = "#e6e6e6"
MUTED       = "#9aa0a6"
ACCENT      = "#0a84ff"   # Run
BORDER      = "#3c3c3c"

# ---------------- Network helpers ----------------
def normalize_host(s: str) -> str:
    s = (s or "").strip()
    if not s:
        raise ValueError("Comfy host is empty.")
    if not (s.startswith("http://") or s.startswith("https://")):
        s = "http://" + s
    return s.rstrip("/")

def check_server_or_raise(host_text: str) -> str:
    host = normalize_host(host_text)
    r = requests.get(f"{host}/object_info", timeout=5)
    r.raise_for_status()
    return host

# ---------------- Workflow IO (API format only) ----------------
def load_api_workflow(path: str) -> dict:
    """
    Load a ComfyUI workflow in API format.
    - Accepts a dict at root (API) or under top-level key 'workflow' (already API).
    - If it looks like an editor graph (contains 'nodes'/'links'), we bail out.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    wf = data.get("workflow") if isinstance(data, dict) else None
    wf = wf if isinstance(wf, dict) else data

    if isinstance(wf, dict) and ("nodes" in wf or "links" in wf):
        raise RuntimeError(
            "This file is an *editor* graph. Please export API JSON (Manager → Save (API)) "
            "or use a file that already contains the API prompt graph."
        )

    if not isinstance(wf, dict) or not wf:
        raise RuntimeError("Workflow must be a non-empty dict in API format.")

    # Ensure all keys are strings (Comfy expects string node ids)
    wf = {str(k): v for k, v in wf.items()}
    # Minimal sanity check for API shape
    for nid, node in wf.items():
        if not isinstance(node, dict) or "class_type" not in node or "inputs" not in node:
            raise RuntimeError("Invalid API graph: each node must have 'class_type' and 'inputs'.")
    return wf

def get_lines_from_node(api_graph: dict, node_id: str, key: str = "text"):
    node_id = str(node_id)
    if node_id not in api_graph:
        raise KeyError(f"Node id '{node_id}' not found in API graph.")
    node = api_graph[node_id]
    inputs = node.get("inputs", {})
    if key not in inputs:
        raise KeyError(f"Input key '{key}' not found on node {node_id}. Available: {list(inputs.keys())}")
    raw = inputs[key]
    if not isinstance(raw, str):
        raise TypeError(
            f"Expected string for inputs['{key}'] on node {node_id}, got {type(raw)}.\n"
            "If that input is *linked*, use a Prompts TXT file instead."
        )
    lines = [ln.strip() for ln in raw.splitlines()]
    return [ln for ln in lines if ln and not ln.lstrip().startswith('#')]

def get_lines_from_txt(path: str):
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln and not ln.lstrip().startswith('#')]

def queue_prompt(host: str, prompt_graph: dict, client_id: str) -> dict:
    url = f"{host}/prompt"
    payload = {"prompt": prompt_graph, "client_id": client_id}
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} from {url}\n{r.text}")
    return r.json()

# ---------------- GUI ----------------
class BatchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ComfyUI Batch GUI — queue prompts per line")
        self.geometry("780x560")
        self.minsize(700, 520)
        self.configure(bg=BG)

        # ttk styling
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except tk.TclError: pass

        style.configure("TFrame", background=PANEL_BG)
        style.configure("TLabel", background=PANEL_BG, foreground=FG)
        style.configure("TEntry", fieldbackground=INPUT_BG, foreground=FG,
                        background=PANEL_BG, bordercolor=BORDER)
        style.map("TEntry", fieldbackground=[("disabled", "#1f1f1f"), ("!disabled", INPUT_BG)])
        style.configure("TSpinbox", fieldbackground=INPUT_BG, foreground=FG, arrowsize=12)
        style.map("TSpinbox", fieldbackground=[("disabled", "#1f1f1f"), ("!disabled", INPUT_BG)])

        # Dark button style (for Browse/…)
        style.configure("Dark.TButton",
                        background=INPUT_BG,
                        foreground=FG,
                        bordercolor=BORDER,
                        relief="flat",
                        padding=(12,6))
        style.map("Dark.TButton",
                  background=[("active","#3a3a3a"), ("pressed","#444444")],
                  foreground=[("disabled","#666666")])

        # Accent button (Run only)
        style.configure("Accent.TButton", background=ACCENT, foreground="#0b0b0b",
                        bordercolor=ACCENT, relief="flat", padding=(12,6))
        style.map("Accent.TButton", background=[("active","#1a8dff"), ("pressed","#0f6fd6")])

        pad = {"padx": 10, "pady": 6}

        # --- Top frame (4 columns, neat alignment) ---
        top = ttk.Frame(self, style="TFrame")
        top.pack(fill="x", **pad)

        top.grid_columnconfigure(0, weight=0)
        top.grid_columnconfigure(1, weight=1)   # stretch
        top.grid_columnconfigure(2, weight=0)
        top.grid_columnconfigure(3, weight=0)

        # Row 0: Comfy Host
        ttk.Label(top, text="Comfy Host:").grid(row=0, column=0, sticky="w", pady=(0,6))
        self.host_var = tk.StringVar(value="http://127.0.0.1:8188")
        ttk.Entry(top, textvariable=self.host_var).grid(row=0, column=1, columnspan=3,
                                                        sticky="ew", padx=(6,0), pady=(0,6))

        # Row 1: Workflow JSON
        ttk.Label(top, text="Workflow JSON:").grid(row=1, column=0, sticky="w")
        self.json_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.json_var).grid(row=1, column=1, columnspan=2,
                                                        sticky="ew", padx=(6,6))
        ttk.Button(top, text="Browse…", style="Dark.TButton",
                   command=self._pick_json).grid(row=1, column=3, sticky="e")

        # Row 2: Prompts TXT (optional)
        ttk.Label(top, text="Prompts TXT (optional):").grid(row=2, column=0, sticky="w")
        self.txt_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.txt_var).grid(row=2, column=1, columnspan=2,
                                                       sticky="ew", padx=(6,6))
        ttk.Button(top, text="…", width=3, style="Dark.TButton",
                   command=self._pick_txt).grid(row=2, column=3, sticky="e")

        # Row 3: Text Node ID  |  Text Input Key
        ttk.Label(top, text="Text Node ID:").grid(row=3, column=0, sticky="w", pady=(6,0))
        self.node_var = tk.StringVar(value="1")
        ttk.Entry(top, textvariable=self.node_var, width=10).grid(row=3, column=1,
                                                                  sticky="w", padx=(6,0), pady=(6,0))

        ttk.Label(top, text="Text Input Key:").grid(row=3, column=2, sticky="e", pady=(6,0))
        self.key_var = tk.StringVar(value="text")
        ttk.Entry(top, textvariable=self.key_var, width=12).grid(row=3, column=3,
                                                                 sticky="w", padx=(6,0), pady=(6,0))

        # Row 4: Repeats
        ttk.Label(top, text="Repeats per line:").grid(row=4, column=0, sticky="w")
        self.repeats_var = tk.IntVar(value=1)
        ttk.Spinbox(top, from_=1, to=10000, textvariable=self.repeats_var, width=8)\
            .grid(row=4, column=1, sticky="w", padx=(6,0))

        # Buttons row (Run only)
        btns = ttk.Frame(self, style="TFrame")
        btns.pack(fill="x", **pad)
        self.run_btn = ttk.Button(btns, text="Run", style="Accent.TButton", command=self._on_run)
        self.run_btn.pack(side="right")

        # Log area
        self.log = tk.Text(self, wrap="word", height=18,
                           bg=INPUT_BG, fg=FG, insertbackground=FG,
                           highlightthickness=1, highlightbackground=BORDER,
                           relief="flat")
        self.log.pack(fill="both", expand=True, **pad)
        self._log("Ready. Select a workflow JSON (API format).")

        # queues/timers
        self._log_queue = queue.Queue()
        self._poll_log_queue()

        # single client id for the app session (helps identify runs in server logs)
        self._client_id = f"batch_gui:{uuid.uuid4()}"

    # --- File pickers
    def _pick_json(self):
        p = filedialog.askopenfilename(title="Select Workflow JSON",
                                       filetypes=[("JSON files","*.json"), ("All files","*.*")])
        if p:
            self.json_var.set(p)

    def _pick_txt(self):
        p = filedialog.askopenfilename(title="Select Prompts TXT",
                                       filetypes=[("Text files","*.txt"), ("All files","*.*")])
        if p:
            self.txt_var.set(p)

    # --- Logging pump
    def _log(self, msg):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def _enqueue_log(self, msg):
        self._log_queue.put(msg)

    def _poll_log_queue(self):
        while True:
            try:
                msg = self._log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._log(msg)
        self.after(100, self._poll_log_queue)

    # --- Helpers
    def _get_repeats(self) -> int:
        try:
            val = int(str(self.repeats_var.get()).strip())
        except Exception:
            val = 1
        return max(1, val)

    def _set_unique_seed(self, graph: dict, job_index: int) -> int:
        """
        Force a unique seed in the prompt graph so ComfyUI cache won't hit.
        Tries direct sampler 'seed' fields and common upstream numeric fields.
        Returns how many places were changed.
        """
        seed_val = (int(time.time() * 1000) ^ (job_index * 2654435761)) & 0x7fffffff
        changed = 0
        for nid, node in graph.items():
            ins = node.get("inputs", {})
            if "seed" in ins:
                v = ins["seed"]
                if isinstance(v, (int, float)):
                    ins["seed"] = int(seed_val)
                    changed += 1
                elif isinstance(v, list) and v and isinstance(v[0], (str, int)):
                    src_id = str(v[0])
                    src = graph.get(src_id)
                    if isinstance(src, dict):
                        src_ins = src.get("inputs", {})
                        for key in ("seed", "value", "val", "number", "int", "integer"):
                            if key in src_ins and isinstance(src_ins[key], (int, float)):
                                src_ins[key] = int(seed_val)
                                changed += 1
                                break
        return changed

    # --- Controls
    def _on_run(self):
        # Preflight
        try:
            host = check_server_or_raise(self.host_var.get())
        except Exception as e:
            messagebox.showerror("Connection error", str(e))
            return

        jp = self.json_var.get().strip()
        if not jp:
            messagebox.showerror("Missing file", "Please select a Workflow JSON (API format).")
            return
        if not Path(jp).is_file():
            messagebox.showerror("File not found", f"JSON file not found:\n{jp}")
            return

        # Disable Run while working
        self.run_btn.config(state="disabled")
        threading.Thread(target=lambda: self._worker(host), daemon=True).start()

    # --- Worker
    def _worker(self, host: str):
        json_p   = self.json_var.get().strip()
        txt_p    = self.txt_var.get().strip()
        node_id  = self.node_var.get().strip()
        key      = self.key_var.get().strip() or "text"

        try:
            self._enqueue_log("Loading workflow (API format)...")
            graph = load_api_workflow(json_p)

            # Collect prompts
            if txt_p:
                lines = get_lines_from_txt(txt_p)
                self._enqueue_log(f"Loaded {len(lines)} line(s) from TXT.")
            else:
                lines = get_lines_from_node(graph, node_id, key=key)
                self._enqueue_log(f"Loaded {len(lines)} line(s) from node {node_id}.{key}.")

            if not lines:
                raise RuntimeError("No prompts found after filtering (empty and '#' lines ignored).")

            repeats = self._get_repeats()
            jobs = [(idx, line, rep) for idx, line in enumerate(lines, 1) for rep in range(1, repeats + 1)]
            total = len(jobs)
            self._enqueue_log(f"Preflight: {len(lines)} line(s) × repeats {repeats} = {total} job(s).")

            force_unique = repeats > 1  # only randomize when repeats > 1
            if not force_unique:
                self._enqueue_log("Repeats = 1 → respecting workflow seed (no forced randomization).")

            # Queue jobs
            done = 0
            for idx, line, rep in jobs:
                patched = copy.deepcopy(graph)
                patched[str(node_id)]["inputs"][key] = line

                if force_unique:
                    changed = self._set_unique_seed(patched, job_index=(idx - 1) * repeats + rep)
                    if changed:
                        self._enqueue_log(f"Set unique seed on {changed} node(s) for line#{idx} rep#{rep}")

                data = queue_prompt(host, patched, client_id=self._client_id)
                done += 1
                self._enqueue_log(f"[{done}/{total}] queued prompt_id={data.get('prompt_id')}  line#{idx}  rep#{rep}")
                time.sleep(0.05)

            self._enqueue_log("Done.")
        except Exception as e:
            self._enqueue_log(f"ERROR: {e}")
        finally:
            self.run_btn.config(state="normal")

def main():
    app = BatchGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
