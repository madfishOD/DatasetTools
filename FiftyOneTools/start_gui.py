
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

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("FiftyOne Tools GUI")
        self.geometry("740x480")
        #self.grid_columnconfigure(0, weight=1)
        #self.grid_rowconfigure((0, 1), weight=1)
        
        self.refresh_datasets()
        dataset_list = fiftyone.list_datasets()

        self.label0 = LabelBig(self, text="Select Dataset:")
        self.label0.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        self.selected_dataset = tkinter.StringVar(value=dataset_list[0] if dataset_list else "")
        self.dataset_selector = customtkinter.CTkOptionMenu(self, values=dataset_list, variable=self.selected_dataset)
        self.dataset_selector.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.label_button = LabelBig(self, "Refresh datasets list:")
        self.label_button.grid(row=0, column=2, padx=10, pady=10, sticky="e")

        self.button = customtkinter.CTkButton(self, text="Refresh", command=self.refresh_datasets)
        self.button.grid(row=0, column=3, padx=10, pady=10, sticky="w")

    def refresh_callback(self):
        print("button pressed")

    def refresh_datasets(self):
        dataset_list = fiftyone.list_datasets()


app = App()
app.mainloop()