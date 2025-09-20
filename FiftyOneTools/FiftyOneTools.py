# -*- coding: utf-8 -*-
import os

import fo_utils
os.environ["FIFTYONE_DATABASE_DIR"] = r"D:\FiftyOneDB"  # root .fiftyone folder, not ...\var\lib\mongo

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pathlib import Path

import sys, getpass, fiftyone as fo
print(
    "\n=== FO ENV CHECK ===",
    f"\nUSER      : {getpass.getuser()}",
    f"\nPYTHON    : {sys.executable}",
    f"\nFiftyOne  : {fo.__version__}",
    f"\nDB ROOT   : {fo.config.database_dir}",  # should END with .fiftyone (root), NOT ...\var\lib\mongo
    f"\nDATASETS  : {fo.list_datasets()}",
    "\n====================\n",
)

#print(Path(fo.config.database_dir).expanduser().resolve())



def main():
    root = tk.Tk()
    root.title("FiftyOne Tools")
    root.geometry("820x780")
    root.resizable(True, True)

    # ---------- vars ----------
    selected_ds = tk.StringVar(value="")

    # ---------- layout ----------
    pad = {"padx": 10, "pady": 6}

    # Existing
    f1 = ttk.LabelFrame(root, text="Open / manage existing dataset")
    f1.pack(fill="x", padx=10, pady=6)

    row1 = tk.Frame(f1); row1.pack(fill="x", **pad)
    ttk.Label(row1, text="Dataset:").pack(side="left")
    cb = ttk.Combobox(row1, textvariable=selected_ds, values=tuple(sorted(fo.list_datasets())), width=42, state="readonly")
    cb.pack(side="left", padx=6)

    ttk.Button(row1, text="Open", command=lambda:open_fo_database(selected_ds.get())).pack(side="right")

    f2 = ttk.LabelFrame(root, text="Create new dataset")
    f2.pack(fill="x", padx=10, pady=6)

    row2 = tk.Frame(f2); row2.pack(fill="x", **pad)
    ttk.Label(row2, text="Name:").pack(side="left")

    root.mainloop()  # <-- keep the window open

def open_fo_database(name: str):
    if not name:
       messagebox.showwarning("Select dataset", "Please choose a dataset from the dropdown.")
       return
    if name in fo.list_datasets():
        #messagebox.showinfo("Launching", f"Launching FiftyOne App for '{ds.name}'")
        ds = fo.load_dataset(name)
        session = fo.launch_app(ds)
        session.wait()

if __name__ == "__main__":
    main()

