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
DANGER      = "#ff3b30"   # Stop
BORDER      = "#3c3c3c"

# ---------------- Core helpers (unchanged) ----------------
def is_editor_json(d: dict) -> bool:
    return isinstance(d, dict) and "nodes" in d and "links" in d

def convert_editor_to_api(editor: dict) -> dict:
    def widget_map(n):
        names = [w.get("name") for w in n.get("widgets", []) if isinstance(w.get("name"), str)]
        vals = n.get("widgets_values", [])
        return { (names[i] if i < len(names) else f"w{i}") : v for i, v in enumerate(vals) }

    def index_links(links):
        idx = {}
        for it in links:
            if isinstance(it, dict):
                lid = int(it["id"]); idx[lid] = (int(it["from_node"]), int(it["from_slot"]), int(it["to_node"]), int(it["to_slot"]))
            elif isinstance(it, list) and len(it) >= 5:
                lid = int(it[0]); idx[lid] = (int(it[1]), int(it[2]), int(it[3]), int(it[4]))
        return idx

    nodes, links = editor.get("nodes", []), editor.get("links", [])
    link_idx = index_links(links)
    api = {}
    for n in nodes:
        nid = str(n.get("id"))
        ctype = n.get("type")
        if not nid or not ctype:
            continue
        ins = {}
        wmap = widget_map(n)
        for inp in n.get("inputs", []):
            name, lnk = inp.get("name"), inp.get("link")
            if not isinstance(name, str):
                continue
            if lnk is None or lnk == -1:
                if name in wmap:
                    ins[name] = wmap[name]
            else:
                if int(lnk) in link_idx:
                    frm, frm_slot, to, to_slot = link_idx[int(lnk)]
                    if int(nid) == to:
                        ins[name] = [str(frm), frm_slot]
        api[nid] = {"class_type": ctype, "inputs": ins}
    return api

def load_any_workflow(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        wf = json.load(f)
    wf = wf.get("workflow") or wf
    if is_editor_json(wf):
        wf = convert_editor_to_api(wf)
    if not isinstance(wf, dict) or not wf:
        raise RuntimeError("Workflow must be a non-empty dict (API format).")
    return {str(k): v for k, v in wf.items()}

def fetch_object_info(host: str) -> dict:
    r = requests.get(f"{host}/object_info", timeout=20)
    r.raise_for_status()
    return r.json()

def validate_against_server(api_graph: dict, object_info: dict):
    classes = object_info.get("classes") or object_info
    missing = []
    for nid, node in api_graph.items():
        ctype = node.get("class_type")
        if ctype not in classes:
            missing.append((nid, ctype))
    if missing:
        msg = "\n".join([f"- node {nid}: '{ctype}'" for nid, ctype in missing])
        raise RuntimeError("Unknown node class on server:\n" + msg)

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
        raise TypeError(f"Expected string for inputs['{key}'] on node {node_id}, got {type(raw)}.\n"
                        "If it's linked, provide a TXT file instead.")
    lines = [ln.strip() for ln in raw.splitlines()]
    return [ln for ln in lines if ln and not ln.lstrip().startswith('#')]

def get_lines_from_txt(path: str):
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln and not ln.lstrip().startswith('#')]

def queue_prompt(host: str, prompt_graph: dict) -> dict:
    url = f"{host}/prompt"
    payload = {"prompt": prompt_graph, "client_id": str(uuid.uuid4())}
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} from {url}\n{r.text}")
    return r.json()

# ---------------- GUI ----------------
class BatchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ComfyUI Batch GUI — queue prompts per line")
        self.geometry("760x520")
        self.minsize(680, 480)

        # Apply dark window background
        self.configure(bg=BG)

        # ttk styling
        style = ttk.Style(self)
        try:
            style.theme_use("clam")  # respects custom colors across platforms
        except tk.TclError:
            pass

        style.configure("TFrame", background=PANEL_BG)
        style.configure("TLabel", background=PANEL_BG, foreground=FG)

        style.configure("TEntry",
                        fieldbackground=INPUT_BG,
                        foreground=FG,
                        background=PANEL_BG,
                        bordercolor=BORDER)
        style.map("TEntry",
                  fieldbackground=[("disabled", "#1f1f1f"), ("!disabled", INPUT_BG)])

        style.configure("TSpinbox",
                        fieldbackground=INPUT_BG,
                        foreground=FG,
                        arrowsize=12)
        style.map("TSpinbox",
                  fieldbackground=[("disabled", "#1f1f1f"), ("!disabled", INPUT_BG)])

        style.configure("Dark.TButton",
                        background=INPUT_BG,
                        foreground=FG,
                        bordercolor=BORDER,
                        relief="flat",
                        padding=(10,4))
        style.map("Dark.TButton",
                  background=[("active","#3a3a3a"),("pressed","#444444")],
                  foreground=[("disabled","#666666")])

        style.configure("Accent.TButton",
                        background=ACCENT,
                        foreground="#0b0b0b",
                        bordercolor=ACCENT,
                        relief="flat",
                        padding=(10,4))
        style.map("Accent.TButton",
                  background=[("active","#1a8dff"), ("pressed","#0f6fd6")])

        style.configure("Danger.TButton",
                        background=DANGER,
                        foreground="#0b0b0b",
                        bordercolor=DANGER,
                        relief="flat",
                        padding=(10,4))
        style.map("Danger.TButton",
                  background=[("active","#ff5349"), ("pressed","#d62a20")],
                  foreground=[("disabled","#444444")])

        pad = {"padx": 10, "pady": 6}

        # Top frame (grid inside)
        top = ttk.Frame(self, style="TFrame")
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Comfy Host:").grid(row=0, column=0, sticky="w")
        self.host_var = tk.StringVar(value="http://127.0.0.1:8188")
        ttk.Entry(top, textvariable=self.host_var, width=40).grid(row=0, column=1, sticky="ew", padx=(6,12))

        ttk.Label(top, text="Workflow JSON:").grid(row=1, column=0, sticky="w")
        self.json_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.json_var).grid(row=1, column=1, sticky="ew", padx=(6,6))
        ttk.Button(top, text="Browse…", style="Dark.TButton", command=self._pick_json)\
            .grid(row=1, column=2, sticky="e")

        ttk.Label(top, text="Prompts TXT (optional):").grid(row=2, column=0, sticky="w")
        self.txt_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.txt_var).grid(row=2, column=1, sticky="ew", padx=(6,6))
        ttk.Button(top, text="…", width=3, style="Dark.TButton", command=self._pick_txt)\
            .grid(row=2, column=2, sticky="e")

        ttk.Label(top, text="Text Node ID:").grid(row=3, column=0, sticky="w")
        self.node_var = tk.StringVar(value="1")
        ttk.Entry(top, textvariable=self.node_var, width=8).grid(row=3, column=1, sticky="w", padx=(6,0))

        ttk.Label(top, text="Text Input Key:").grid(row=3, column=1, sticky="e", padx=(120,0))
        self.key_var = tk.StringVar(value="text")
        ttk.Entry(top, textvariable=self.key_var, width=12).grid(row=3, column=1, sticky="w", padx=(220,0))

        ttk.Label(top, text="Repeats per line:").grid(row=4, column=0, sticky="w")
        self.repeats_var = tk.IntVar(value=1)
        ttk.Spinbox(top, from_=1, to=10000, textvariable=self.repeats_var, width=8)\
            .grid(row=4, column=1, sticky="w", padx=(6,0))

        # Buttons row (pack)
        btns = ttk.Frame(self, style="TFrame")
        btns.pack(fill="x", **pad)
        # Align to the right for a more "app-like" look
        self.run_btn = ttk.Button(btns, text="Run", style="Accent.TButton", command=self._on_run)
        self.run_btn.pack(side="right")
        self.stop_btn = ttk.Button(btns, text="Stop", style="Danger.TButton", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="right", padx=(0,8))

        # Log area (tk.Text needs manual colors)
        self.log = tk.Text(self, wrap="word", height=18,
                           bg=INPUT_BG, fg=FG,
                           insertbackground=FG,  # caret color
                           highlightthickness=1, highlightbackground=BORDER,
                           relief="flat")
        self.log.pack(fill="both", expand=True, **pad)
        self._log("Ready. Select a workflow JSON.")

        # Grid weights for the top frame
        for c in range(3):
            top.grid_columnconfigure(c, weight=1 if c == 1 else 0)

        # Runtime flags/queues
        self._stop_flag = threading.Event()
        self._log_queue = queue.Queue()
        self._poll_log_queue()

    # --- File pickers
    def _pick_json(self):
        p = filedialog.askopenfilename(title="Select Workflow JSON",
                                       filetypes=[("JSON files","*.json"),("All","*.*")])
        if p:
            self.json_var.set(p)

    def _pick_txt(self):
        p = filedialog.askopenfilename(title="Select Prompts TXT",
                                       filetypes=[("Text files","*.txt"),("All","*.*")])
        if p:
            self.txt_var.set(p)

    # --- Logging
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

    # --- Controls
    def _on_run(self):
        if not self.json_var.get().strip():
            messagebox.showerror("Missing file", "Please select a Workflow JSON.")
            return
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._stop_flag.clear()
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _on_stop(self):
        # Signals the worker loop to stop *after* the current in-flight request
        self._stop_flag.set()
        self._enqueue_log("Stop requested...")

    # --- Worker
    def _worker(self):
        host     = self.host_var.get().strip()
        json_p   = self.json_var.get().strip()
        txt_p    = self.txt_var.get().strip()
        node_id  = self.node_var.get().strip()
        key      = self.key_var.get().strip() or "text"
        repeats  = max(1, int(self.repeats_var.get()))

        try:
            self._enqueue_log("Loading workflow...")
            graph = load_any_workflow(json_p)

            self._enqueue_log("Validating node classes on server...")
            objinfo = fetch_object_info(host)
            validate_against_server(graph, objinfo)

            # Collect prompts
            if txt_p:
                lines = get_lines_from_txt(txt_p)
                self._enqueue_log(f"Loaded {len(lines)} lines from TXT.")
            else:
                lines = get_lines_from_node(graph, node_id, key=key)
                self._enqueue_log(f"Loaded {len(lines)} lines from node {node_id}.{key}.")

            if not lines:
                raise RuntimeError("No prompts found after filtering (empty and '#' lines ignored).")

            total = len(lines) * repeats
            count = 0
            for idx, line in enumerate(lines, 1):
                if self._stop_flag.is_set():
                    break
                for r in range(repeats):
                    if self._stop_flag.is_set():
                        break
                    patched = copy.deepcopy(graph)
                    patched[str(node_id)]["inputs"][key] = line
                    data = queue_prompt(host, patched)  # may block briefly
                    count += 1
                    self._enqueue_log(f"[{count}/{total}] queued prompt_id={data.get('prompt_id')} line#{idx} rep#{r+1}")
                    time.sleep(0.05)

            self._enqueue_log("Done." if not self._stop_flag.is_set() else "Stopped.")
        except Exception as e:
            self._enqueue_log(f"ERROR: {e}")
        finally:
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

def main():
    app = BatchGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
