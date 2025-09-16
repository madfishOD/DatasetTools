import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

# --- Dark palette
BG          = "#1e1e1e"
PANEL_BG    = "#252526"
INPUT_BG    = "#2d2d2d"
FG          = "#e6e6e6"
ACCENT      = "#0a84ff"
BORDER      = "#3c3c3c"
MUTED       = "#9aa0a6"

class JsonPathPicker(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window basics
        self.title("Workflow JSON Picker")
        self.minsize(520, 120)          # scalable window
        self.configure(bg=BG)           # outer background
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # --- ttk dark styling
        style = ttk.Style(self)
        try:
            style.theme_use("clam")     # 'clam' respects colors cross-platform
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TEntry",
                        fieldbackground=INPUT_BG,
                        foreground=FG,
                        background=BG,
                        bordercolor=BORDER)
        style.map("TEntry",
                  fieldbackground=[("disabled", "#1f1f1f"), ("!disabled", INPUT_BG)])

        style.configure("Dark.TButton",
                        background=INPUT_BG,
                        foreground=FG,
                        bordercolor=BORDER,
                        focusthickness=1,
                        focuscolor=ACCENT,
                        padding=(10, 4))
        style.map("Dark.TButton",
                  background=[("active", "#3a3a3a"), ("pressed", "#444444")],
                  foreground=[("disabled", "#666666")])

        # --- Main frame
        main = ttk.Frame(self, padding=12, style="TFrame")
        main.grid(sticky="nsew")
        main.columnconfigure(1, weight=1)  # make the path field stretch

        # --- Label
        ttk.Label(main, text="Workflow JSON").grid(row=0, column=0, padx=(0, 8), sticky="w")

        # --- Path field (Entry)
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(main, textvariable=self.path_var)
        path_entry.grid(row=0, column=1, sticky="ew")  # expands horizontally
        # Optional: make the caret a bit thicker (platform dependent)
        try:
            path_entry.configure(insertwidth=2)
        except tk.TclError:
            pass

        # --- Button
        ttk.Button(main, text="Browse…", command=self.pick_file, style="Dark.TButton")\
            .grid(row=0, column=2, padx=(8, 0), sticky="e")

        # # --- Status (optional—but prevents NameError in validate_path)
        # self.status_var = tk.StringVar(value="Select a .json workflow file.")
        # ttk.Label(main, textvariable=self.status_var, foreground=MUTED)\
        #     .grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        # Bind validation when user edits the entry
        self.path_var.trace_add("write", self.validate_path)

    def pick_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Workflow JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(Path.home())
        )
        if file_path:
            self.path_var.set(file_path)

    def validate_path(self, *_):
        text = self.path_var.get().strip()
        if not text:
            self.status_var.set("Select a .json workflow file.")
            return

        p = Path(text)
        if p.suffix.lower() != ".json":
            self.status_var.set("⚠ Not a .json file.")
            return

        if p.exists() and p.is_file():
            self.status_var.set("✓ JSON file selected.")
        else:
            self.status_var.set("⚠ Path does not exist.")

if __name__ == "__main__":
    app = JsonPathPicker()
    app.mainloop()
