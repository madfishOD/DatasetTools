
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
    def __init__(self, master):
        super().__init__(master)

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

        self.button = customtkinter.CTkButton(self, text="Refresh", command=self.refresh_datasets)
        self.button.grid(row=0, column=3, padx=10, pady=10, sticky="w")

        self.button = customtkinter.CTkButton(self, text="Open", command=self.open_dataset)
        is_valid_to_open = bool(self.selected_dataset.get() in fiftyone.list_datasets())
        self.button.configure(state="normal" if is_valid_to_open else "disabled")
        self.button.grid(row=0, column=4, padx=10, pady=10, sticky="w")

    def refresh_datasets(self):
        dataset_list = fiftyone.list_datasets()

    def open_dataset(self):
        dataset_name = self.selected_dataset.get()
        if dataset_name:
            try:
                dataset = fiftyone.load_dataset(dataset_name)
                fiftyone.launch_app(dataset)
            except Exception as e:
                print(f"Error loading dataset '{dataset_name}': {e}")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("FiftyOne Tools GUI")
        self.geometry("840x480")

        # layout
        self.grid_columnconfigure(0, weight=1)

        self.SelectedDatasetFrame = SelectedDatasetFrame(self)
        self.SelectedDatasetFrame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
app = App()
app.mainloop()