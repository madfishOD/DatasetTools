
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

# App frame
app = customtkinter.CTk()
app.geometry("740x480")
app.title("FiftyOne Tools GUI")
app.iconbitmap(None)

# Start the GUI
refresh_datasets()
dataset_list = fiftyone.list_datasets()

dataset_label = customtkinter.CTkLabel(app, text="Dataset:")
dataset_label.pack(padx=10, pady=10)
selected_dataset = tkinter.StringVar(value=dataset_list[0] if dataset_list else "")
dataset_selector = customtkinter.CTkOptionMenu(app, values=dataset_list, variable=selected_dataset)
dataset_selector.pack(padx=10, pady=10)

app.mainloop()