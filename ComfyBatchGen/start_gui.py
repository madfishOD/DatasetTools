import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

# --- Dark palette
BG="#1e1e1e"; INPUT_BG="#2d2d2d"; FG="#e6e6e6"; ACCENT="#0a84ff"; BORDER="#3c3c3c"; MUTED="#9aa0a6"

class JsonPathPicker(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Workflow JSON Picker")
        self.minsize(560, 160)
        self.configure(bg=BG)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # ttk dark styling
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except tk.TclError: pass
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TEntry", fieldbackground=INPUT_BG, foreground=FG, background=BG, bordercolor=BORDER)
        style.map("TEntry", fieldbackground=[("disabled", "#1f1f1f"), ("!disabled", INPUT_BG)])
        style.configure("Dark.TButton", background=INPUT_BG, foreground=FG, bordercolor=BORDER, padding=(10,4))
        style.map("Dark.TButton", background=[("active","#3a3a3a"),("pressed","#444444")],
                  foreground=[("disabled","#666666")])


        main = ttk.Frame(self, padding=12, style="TFrame")
        main.grid(sticky="nsew")
        main.columnconfigure(1, weight=1)  # stretch the path fields

        # ---- Row 0: Workflow JSON
        ttk.Label(main, text="Workflow JSON").grid(row=0, column=0, padx=(0,8), pady=(0,6), sticky="w")
        self.jsonPath_var = tk.StringVar()
        json_entry = ttk.Entry(main, textvariable=self.jsonPath_var)
        json_entry.grid(row=0, column=1, sticky="ew", pady=(0,6))
        ttk.Button(main, text="Browse…", style="Dark.TButton",
                   command=self.pick_json).grid(row=0, column=2, padx=(8,0), pady=(0,6), sticky="e")

        # ---- Row 1: Prompts TXT
        ttk.Label(main, text="Prompts TXT").grid(row=1, column=0, padx=(0,8), sticky="w")
        self.txtPrompt_var = tk.StringVar()
        txt_entry = ttk.Entry(main, textvariable=self.txtPrompt_var)
        txt_entry.grid(row=1, column=1, sticky="ew")
        ttk.Button(main, text="…", style="Dark.TButton",
                   command=self.pick_txt).grid(row=1, column=2, padx=(8,0), sticky="e")

        # ---- Row 2: Prompts Node ID
        ttk.Label(main, text="Prompts Node ID").grid(row=2, column=0, padx=(0,8), sticky="w")
        self.promptId_var = tk.IntVar(value=0)
        promptId_entry = ttk.Entry(main, textvariable=self.promptId_var, width=6)
        promptId_entry.grid(row=2, column=1, sticky="w")

        # ---- Row 3: Prompts Node ID
        ttk.Label(main, text="Repeats").grid(row=3, column=0, padx=(0,8), sticky="w")
        self.repeats_var = tk.IntVar(value=1)
        repeats_entry = ttk.Entry(main, textvariable=self.repeats_var, width=6)
        repeats_entry.grid(row=3, column=1, sticky="w")

        # ---- Row 4: Status line (optional but avoids NameError in validate_path)
        self.status_var = tk.StringVar(value="Select a .json workflow file.")
        ttk.Label(main, textvariable=self.status_var, foreground=MUTED)\
            .grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8,0))

        # ---- Row 5: Run Button
        ttk.Button(main, text="Run Workflow", style="Dark.TButton",
                   command=self.run_workflow).grid(row=5, column=2, padx=(8,0), sticky="e")

        # validate JSON path live
        self.jsonPath_var.trace_add("write", self.validate_path)

    def pick_json(self):
        p = filedialog.askopenfilename(
            title="Select Workflow JSON",
            filetypes=[("JSON files","*.json"), ("All files","*.*")],
            initialdir=str(Path.home())
        )
        if p: self.jsonPath_var.set(p)

    def pick_txt(self):
        p = filedialog.askopenfilename(
            title="Select Prompts TXT",
            filetypes=[("Text files","*.txt"), ("All files","*.*")],
            initialdir=str(Path.home())
        )
        if p: self.txtPrompt_var.set(p)

    def validate_path(self, *_):
        text = self.jsonPath_var.get().strip()
        if not text:
            self.status_var.set("Select a .json workflow file.")
            return
        p = Path(text)
        if p.suffix.lower() != ".json":
            self.status_var.set("⚠ Not a .json file.")
        elif p.exists() and p.is_file():
            self.status_var.set("✓ JSON file selected.")
        else:
            self.status_var.set("⚠ Path does not exist.")

    def run_workflow(self):
        # Placeholder for running the workflow
        json_path = self.jsonPath_var.get().strip()
        txt_path = self.txtPrompt_var.get().strip()
        prompt_id = self.promptId_var.get()
        repeats = self.repeats_var.get()
        print(f"Running workflow with:\n JSON: {json_path}\n TXT: {txt_path}\n Prompt ID: {prompt_id}\n Repeats: {repeats}")
        # Here you would add the logic to actually run the workflow

if __name__ == "__main__":
    JsonPathPicker().mainloop()
