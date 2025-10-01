
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tkinter
from typing import List
import customtkinter

# ---------- parse early flags so DB dir is set BEFORE importing fiftyone ----------
os.environ["FIFTYONE_DATABASE_DIR"] = r"D:\FiftyOneDB"

import fiftyone

# System settings
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "dark-blue", "green"

class LabelBig(customtkinter.CTkLabel):
    def __init__(self, master, text):
        super().__init__(master, text=text)
        
        self.configure(font=customtkinter.CTkFont(size=12, weight="bold"))

class ConfirmDialog(customtkinter.CTkToplevel):
    """
    Modal Yes/No dialog. Use ask_confirm(...) to get a bool quickly.
    """
    def __init__(
        self,
        master,
        title="Confirm",
        message="Are you sure?",
        yes_text="Yes",
        no_text="No",
        width=380,
        height=160,
    ):
        super().__init__(master)
        self.result: bool = False

        # basic window setup
        self.title(title)
        self.resizable(False, False)
        self.transient(master)          # stay on top of parent
        self.grab_set()                 # modal
        self.protocol("WM_DELETE_WINDOW", self._on_no)

        # center on parent
        self.update_idletasks()
        try:
            mx = master.winfo_rootx()
            my = master.winfo_rooty()
            mw = master.winfo_width()
            mh = master.winfo_height()
        except Exception:
            mx, my, mw, mh = 100, 100, 800, 600
        x = mx + (mw - width) // 2
        y = my + (mh - height) // 2
        self.geometry(f"{width}x{height}+{max(0,x)}+{max(0,y)}")

        # layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        msg = customtkinter.CTkLabel(self, text=message, wraplength=width-40, justify="center")
        msg.grid(row=0, column=0, padx=16, pady=(18, 10), sticky="nsew")

        btns = customtkinter.CTkFrame(self)
        btns.grid(row=1, column=0, pady=(6, 14))
        yes = customtkinter.CTkButton(btns, text=yes_text, width=90, command=self._on_yes)
        no  = customtkinter.CTkButton(btns, text=no_text,  width=90, command=self._on_no)
        yes.grid(row=0, column=0, padx=6)
        no.grid(row=0, column=1, padx=6)

        # keyboard shortcuts
        self.bind("<Return>", lambda e: self._on_yes())
        self.bind("<Escape>", lambda e: self._on_no())

        # focus
        self.after(50, yes.focus_set)

    def _on_yes(self):
        self.result = True
        self.destroy()

    def _on_no(self):
        self.result = False
        self.destroy()

@dataclass
class TextMappingRule:
    enabled: bool
    sample_field: str          # e.g. "caption", "meta.notes"
    text_source_prefix: str    # e.g. "cap_", "desc_", "tags_"

class TextMappingsConfig:
    def __init__(self, rules: List[TextMappingRule] | None = None):
        self.rules: List[TextMappingRule] = rules or []

    def to_dict(self) -> dict:
        return {"rules": [asdict(r) for r in self.rules]}

    @classmethod
    def from_dict(cls, d: dict) -> "TextMappingsConfig":
        rules = [TextMappingRule(**r) for r in d.get("rules", [])]
        return cls(rules=rules)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TextMappingsConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(d)

def ask_confirm(master, title="Confirm", message="Are you sure?", **kwargs) -> bool:
    """
    Opens a modal confirm and returns True/False.
    kwargs are passed to ConfirmDialog (yes_text, no_text, width, height).
    """
    dlg = ConfirmDialog(master, title=title, message=message, **kwargs)
    master.wait_window(dlg)
    return dlg.result


class MediaSamplesImportSettingsFrame(customtkinter.CTkFrame):
    def __init__(self, master, config: TextMappingsConfig | None = None):
        super().__init__(master)
        self._rows: list[dict] = []

        # Layout
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=0)

        # Header
        customtkinter.CTkLabel(self, text="Enabled", width=80).grid(row=0, column=0, padx=8, pady=(8,4))
        customtkinter.CTkLabel(self, text="Sample field", anchor="w").grid(row=0, column=1, padx=8, pady=(8,4), sticky="w")
        customtkinter.CTkLabel(self, text="Text prefix", anchor="w").grid(row=0, column=2, padx=8, pady=(8,4), sticky="w")

        # Scrollable list
        self.list_frame = customtkinter.CTkScrollableFrame(self, height=220)
        self.list_frame.grid(row=1, column=0, columnspan=4, padx=8, pady=(0,8), sticky="nsew")
        self.grid_rowconfigure(1, weight=1)

        # Buttons
        self.add_btn = customtkinter.CTkButton(self, text="Add", width=90, command=self._add_blank_row)
        self.add_btn.grid(row=2, column=0, padx=8, pady=(4,10))

        self.remove_btn = customtkinter.CTkButton(self, text="Remove Selected", width=150, command=self._remove_selected)
        self.remove_btn.grid(row=2, column=1, padx=8, pady=(4,10), sticky="w")

        # Load initial
        self.set_config(config or TextMappingsConfig())

    def _add_blank_row(self, init: TextMappingRule | None = None):
        r = init or TextMappingRule(True, "caption", "cap_")

        row_index = len(self._rows)
        # Each row has: select checkbox (for remove), enabled checkbox, field entry, prefix entry
        # A small “select” checkbox to mark for removal:
        sel_var = tkinter.BooleanVar(value=False)
        sel_cb = customtkinter.CTkCheckBox(self.list_frame, text="Selected", variable=sel_var, width=20)
        sel_cb.grid(row=row_index, column=0, padx=4, pady=4, sticky="w")

        en_var = tkinter.BooleanVar(value=r.enabled)
        en_cb = customtkinter.CTkCheckBox(self.list_frame, text="", variable=en_var, width=20)
        en_cb.grid(row=row_index, column=1, padx=8, pady=4, sticky="w")

        sf_var = tkinter.StringVar(value=r.sample_field)
        sf_entry = customtkinter.CTkEntry(self.list_frame, textvariable=sf_var, width=220)
        sf_entry.grid(row=row_index, column=2, padx=8, pady=4, sticky="ew")

        tp_var = tkinter.StringVar(value=r.text_source_prefix)
        tp_entry = customtkinter.CTkEntry(self.list_frame, textvariable=tp_var, width=160)
        tp_entry.grid(row=row_index, column=3, padx=8, pady=4, sticky="ew")

        self.list_frame.grid_columnconfigure(2, weight=1)
        self.list_frame.grid_columnconfigure(3, weight=0)

        self._rows.append({
            "sel": sel_var,
            "enabled": en_var,
            "sample_field": sf_var,
            "prefix": tp_var,
            "widgets": (sel_cb, en_cb, sf_entry, tp_entry),
        })

    def _remove_selected(self):
        keep = []
        for rd in self._rows:
            if rd["sel"].get():
                # destroy widgets
                for w in rd["widgets"]:
                    try: w.destroy()
                    except: pass
            else:
                keep.append(rd)
        self._rows = keep
        # re-pack to compact rows
        for i, rd in enumerate(self._rows):
            for j, w in enumerate(rd["widgets"]):
                w.grid_configure(row=i)

    def get_config(self) -> TextMappingsConfig:
        rules: list[TextMappingRule] = []
        for rd in self._rows:
            rule = TextMappingRule(
                enabled=bool(rd["enabled"].get()),
                sample_field=rd["sample_field"].get().strip(),
                text_source_prefix=rd["prefix"].get().strip(),
            )
            if rule.sample_field and rule.text_source_prefix:
                rules.append(rule)
        return TextMappingsConfig(rules=rules)

    def set_config(self, cfg: TextMappingsConfig):
        # clear current
        for rd in self._rows:
            for w in rd["widgets"]:
                try: w.destroy()
                except: pass
        self._rows = []
        # add rows
        for r in cfg.rules:
            self._add_blank_row(r)


class SelectedDatasetFrame(customtkinter.CTkFrame):
    def __init__(self, master, on_open, on_delete):
        super().__init__(master)  
        
        self.on_open = on_open
        self.on_delete = on_delete

        # Layout
        self.grid_columnconfigure(1, weight=1, minsize=200)  # dropdown grows
        self.grid_columnconfigure(2, weight=0)  # refresh button
    
        #self.refresh_datasets()
        #self.dataset_list = fiftyone.list_datasets()
        self.dataset_list = None

        self.label0 = LabelBig(self, text="Selected Dataset:")
        self.label0.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        self.selected_dataset = tkinter.StringVar(value=self.dataset_list[0] if self.dataset_list else "")
        self.dataset_selector = customtkinter.CTkOptionMenu(self, values=self.dataset_list, variable=self.selected_dataset)
        self.dataset_selector.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.label_button = LabelBig(self, "Refresh datasets list:")
        self.label_button.grid(row=0, column=2, padx=10, pady=10, sticky="e")

        self.button_refresh = customtkinter.CTkButton(self, text="Refresh", command=self.refresh_datasets)
        self.button_refresh.grid(row=0, column=3, padx=10, pady=10, sticky="w")

        is_valid_dataset = bool(self.selected_dataset.get() in fiftyone.list_datasets())

        self.button_open = customtkinter.CTkButton(self, text="Open", command=self._open_clicked)  
        self.button_open.configure(state="normal" if is_valid_dataset else "disabled", width=100)
        self.button_open.grid(row=0, column=4, padx=10, pady=10, sticky="w")

        self.button_delete = customtkinter.CTkButton(self, text="Delete", command=self._delete_clicked)
        self.button_delete.configure(state="normal" if is_valid_dataset else "disabled")
        self.button_delete.grid(row=0, column=5, padx=10, pady=10, sticky="w")
        self.button_delete.configure(fg_color="#FF5C5C", hover_color="#FF1E1E", width=50)

        self.smples_path_label = LabelBig(self, "Media Path:")
        self.smples_path_label.grid(row=1, column=0, padx=10, pady=(0,10), sticky="w")
        self.samples_path = customtkinter.StringVar(value="")
        self.samples_path_entry = customtkinter.CTkEntry(self, textvariable=self.samples_path)
        self.samples_path_entry.grid(row=1, column=1, columnspan=2, padx=10, pady=(10,10), sticky="ew")

        self.button_browse = customtkinter.CTkButton(self, text="Browse", command=self._browse_clicked)
        self.button_browse.grid(row=1, column=3, padx=10, pady=10, sticky="e")

        self.refresh_datasets()

    def refresh_datasets(self):
        print("Refreshing dataset list")
        names = fiftyone.list_datasets()

        # 1) Update dropdown items
        if names:
            self.dataset_selector.configure(values=names)
            # keep current if still valid; else set first
            cur = self.selected_dataset.get()
            self.selected_dataset.set(cur if cur in names else names[0])
            is_valid_dataset = True
        else:
            placeholder = ["(no datasets found)"]
            self.dataset_selector.configure(values=placeholder, state="disabled")
            self.selected_dataset.set(placeholder[0])
            is_valid_dataset = False

        # 2) Update buttons
        self.button_open.configure(state="normal" if is_valid_dataset else "disabled")
        self.button_delete.configure(state="normal" if is_valid_dataset else "disabled")

    def _open_clicked(self):
            name = self.selected_dataset.get()
            if name and name != "(no datasets found)":
                self.on_open(name)

    def _delete_clicked(self):
            name = self.selected_dataset.get()
            if name and name != "(no datasets found)":
                self.on_delete(name)

    def _browse_clicked(self):
        dir_path = tkinter.filedialog.askdirectory()
        if dir_path:  # Check if a directory was actually selected
            self.samples_path.set(dir_path)
            print(f"Media path selected: {dir_path}")
        else:
            print("No directory selected.")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # The shared session handle (None until first open)
        self.session: fiftyone.Session | None = None
        self.current_dataset: fiftyone.Dataset | None = None

        self.title("FiftyOne Tools GUI")
        self.geometry("840x480")

        # layout
        self.grid_columnconfigure(0, weight=1)

        self.selected_dataset_frame = SelectedDatasetFrame(self, on_open=self.open_dataset, on_delete=self.delete_dataset)
        self.selected_dataset_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.media_import_settings = MediaSamplesImportSettingsFrame(self)
        self.media_import_settings.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        self.refresh_button = customtkinter.CTkButton(self, text="Refresh Session", command=self.refresh_session)
        self.refresh_button.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
  
    def refresh_session(self):
        try:
            self.session.refresh()
           
        except Exception as e:
            print(f"Error refreshing session: {e}")
            return

    def open_dataset(self, name: str):
        try:
            ds = fiftyone.load_dataset(name)

        except Exception as e:
            print(f"Error loading dataset '{name}': {e}")
            return

        self.current_dataset = ds
        if self.session is None:
            # first time: create a new App session
            self.session = fiftyone.launch_app(ds)
            self.current_dataset = ds

    def delete_dataset(self, name: str):
        ok = ask_confirm(
            self,
            title="Delete dataset?",
            message=f"Delete '{name}' from the FiftyOne DB?\n\n"
                    "Note: this removes the dataset from the DB only,\n"
                    "original image files are not deleted."
        )
        if not ok:
            return

        try:
            fiftyone.delete_dataset(name)
            print(f"Deleted dataset '{name}'")
            # refresh your dropdown after deletion
            self.selected_dataset_frame.refresh_datasets()
        except Exception as e:
            print(f"Error deleting dataset '{name}': {e}")
   
        
app = App()
app.mainloop()