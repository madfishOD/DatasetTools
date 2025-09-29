
import os
import tkinter
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


def ask_confirm(master, title="Confirm", message="Are you sure?", **kwargs) -> bool:
    """
    Opens a modal confirm and returns True/False.
    kwargs are passed to ConfirmDialog (yes_text, no_text, width, height).
    """
    dlg = ConfirmDialog(master, title=title, message=message, **kwargs)
    master.wait_window(dlg)
    return dlg.result

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
        print("BROWSE")

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

        self.refresh_button = customtkinter.CTkButton(self, text="Refresh Session", command=self.refresh_session)
        self.refresh_button.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
  
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