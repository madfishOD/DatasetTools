
import os
import tkinter
import customtkinter

# ---------- parse early flags so DB dir is set BEFORE importing fiftyone ----------
os.environ["FIFTYONE_DATABASE_DIR"] = r"D:\FiftyOneDB"

import fiftyone

# GUI methods
def refresh_datasets():
    dataset_list = fiftyone.list_datasets() 

# System settings
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "dark-blue", "green"

class LabelBig(customtkinter.CTkLabel):
    def __init__(self, master, text):
        super().__init__(master, text=text)
        
        self.configure(font=customtkinter.CTkFont(size=12, weight="bold"))

class SelectedDatasetFrame(customtkinter.CTkFrame):
    def __init__(self, master, on_open, on_delete):
        super().__init__(master)  
        
        self.on_open = on_open
        self.on_delete = on_delete

        # Layout
        self.grid_columnconfigure(1, weight=1, minsize=200)  # dropdown grows
        self.grid_columnconfigure(2, weight=0)  # refresh button
    
        self.refresh_datasets()
        dataset_list = fiftyone.list_datasets()

        self.label0 = LabelBig(self, text="Selected Dataset:")
        self.label0.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        self.selected_dataset = tkinter.StringVar(value=dataset_list[0] if dataset_list else "")
        self.dataset_selector = customtkinter.CTkOptionMenu(self, values=dataset_list, variable=self.selected_dataset)
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

    def refresh_datasets(self):
        dataset_list = fiftyone.list_datasets()

    def _open_clicked(self):
            name = self.selected_dataset.get()
            if name and name != "(no datasets found)":
                self.on_open(name)

    def _delete_clicked(self):
            name = self.selected_dataset.get()
            if name and name != "(no datasets found)":
                self.on_delete(name)

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
        try:
            fiftyone.delete_dataset(name)
        except Exception as e:
            print(f"Error deleting dataset '{name}': {e}")
            return
   
        
app = App()
app.mainloop()