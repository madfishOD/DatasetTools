import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

class JsonPathPicker(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window basics
        self.title("Workflow JSON Picker")
        self.minsize(520, 100)          # scalable window
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # --- Main frame (adds padding & makes scaling easier)
        main = ttk.Frame(self, padding=12)
        main.grid(sticky="nsew")
        main.columnconfigure(1, weight=1)  # make the path field stretch
        # Row 0 will hold the whole horizontal group

        # --- Label
        ttk.Label(main, text="Workflow JSON").grid(row=0, column=0, padx=(0, 8), sticky="w")

        # --- Path field (Entry)
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(main, textvariable=self.path_var)
        path_entry.grid(row=0, column=1, sticky="ew")  # expands horizontally

        # --- Button
        ttk.Button(main, text="Browse…", command=self.pick_file).grid(row=0, column=2, padx=(8, 0), sticky="e")

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
        p = Path(self.path_var.get())
        if not p:
            self.status_var.set("Select a .json workflow file.")
            return

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
