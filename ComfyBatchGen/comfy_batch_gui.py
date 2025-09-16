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
    try:
        r = requests.get(f"{host}/object_info", timeout=5)
        r.raise_for_status()
        return host
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Cannot reach ComfyUI at {host}.\n{e}")

# ---------------- Workflow conversion & IO ----------------
def is_editor_json(d: dict) -> bool:
    return isinstance(d, dict) and "nodes" in d and "links" in d

def convert_editor_to_api(editor: dict) -> dict:
    # Minimal editor->API conversion (covers common cases).
    def widget_map(n):
        names = [w.get("name") for w in n.get("widgets", []) if isinstance(w.get("name"), str)]
        vals = n.get("widgets_values", [])
        return { (names[i] if i < len(names) else f"w{i}") : v for i, v in enumerate(vals) }

    def index_links(links):
        idx = {}
        for it in links:
            if isinstance(it, dict):
                lid = int(it["id"])
                idx[lid] = (int(it["from_node"]), int(it["from_slot"]), int(it["to_node"]), int(it["to_slot"]))
            elif isinstance(it, list) and len(it) >= 5:
                lid = int(it[0])
                idx[lid] = (int(it[1]), int(it[2]), int(it[3]), int(it[4]))
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
    # Ensure string keys
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

        # Dark window background
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
        self.run_btn = ttk.Button(btns, text="Run", style="Accent.TButton", command=self._on_run)
        self.run_btn.pack(side="right")
        self.stop_btn = ttk.Button(btns, text="Stop", style="Danger.TButton", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="right", padx=(0,8))

        # Log area (tk.Text manual colors)
        self.log = tk.Text(self, wrap="word", height=18,
                           bg=INPUT_BG, fg=FG,
                           insertbackground=FG,  # caret
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

    # --- Small helpers (logic only)
    def _get_repeats(self) -> int:
        """Robustly read repeats (Spinbox may return '' or strings)."""
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
        try:
            host = check_server_or_raise(self.host_var.get())
        except Exception as e:
            messagebox.showerror("Connection error", str(e))
            return

        jp = self.json_var.get().strip()
        if not jp:
            messagebox.showerror("Missing file", "Please select a Workflow JSON.")
            return
        if not Path(jp).is_file():
            messagebox.showerror("File not found", f"JSON file not found:\n{jp}")
            return

        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._stop_flag.clear()
        t = threading.Thread(target=lambda: self._worker(host), daemon=True)
        t.start()

    def _on_stop(self):
        self._stop_flag.set()
        self._enqueue_log("Stop requested...")

    # --- Worker
    def _worker(self, host: str):
        json_p   = self.json_var.get().strip()
        txt_p    = self.txt_var.get().strip()
        node_id  = self.node_var.get().strip()
        key      = self.key_var.get().strip() or "text"

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

            repeats = self._get_repeats()
            jobs = [(idx, line, rep) for idx, line in enumerate(lines, 1) for rep in range(1, repeats + 1)]
            total = len(jobs)
            self._enqueue_log(f"Preflight: {len(lines)} line(s) × repeats {repeats} = {total} job(s).")

            done = 0
            for idx, line, rep in jobs:
                if self._stop_flag.is_set():
                    break

                patched = copy.deepcopy(graph)
                patched[str(node_id)]["inputs"][key] = line

                # Make prompt hash unique so ComfyUI won't serve from cache
                changed = self._set_unique_seed(patched, job_index=(idx - 1) * repeats + rep)
                if changed:
                    self._enqueue_log(f"Set unique seed on {changed} node(s) for line#{idx} rep#{rep}")

                data = queue_prompt(host, patched)
                done += 1
                self._enqueue_log(f"[{done}/{total}] queued prompt_id={data.get('prompt_id')}  line#{idx}  rep#{rep}")
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
