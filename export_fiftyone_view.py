#!/usr/bin/env python
"""
Tk GUI: export a FiftyOne dataset/view to an ImageDirectory with
per-image .txt sidecars (selected sample fields/tags), and optionally
register the export as a new dataset in the FiftyOne DB.
"""

import os
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import fiftyone as fo
from fiftyone import types as fot
from fiftyone.core.view import DatasetView  # <-- correct view type


# ---------------------- helpers ----------------------
def list_datasets():
    try:
        return sorted(fo.list_datasets())
    except Exception as e:
        messagebox.showerror("FiftyOne", f"Could not list datasets:\n{e}")
        return []


def list_saved_views(ds_name: str):
    try:
        ds = fo.load_dataset(ds_name)
        names = ["< Entire dataset >"]
        try:
            sv = ds.list_saved_views() or []
            names.extend(sorted(sv))
        except Exception:
            pass
        return names
    except Exception as e:
        messagebox.showerror("FiftyOne", f"Could not list saved views for '{ds_name}':\n{e}")
        return ["< Entire dataset >"]


def get_primitive_sample_fields(ds: fo.Dataset):
    """
    Return sample-level primitive fields suitable for writing to text.
    (String, Int, Float, Bool, and lists of those). Excludes filepath/id/metadata.
    """
    schema = ds.get_field_schema()
    fields = []
    for name, f in schema.items():
        if name in {"id", "filepath", "metadata"}:
            continue
        ftype = f.__class__.__name__
        if ftype in {"StringField", "IntField", "FloatField", "BooleanField"}:
            fields.append(name)
        elif ftype == "ListField":
            sub = getattr(f, "field", None)
            subname = sub.__class__.__name__ if sub else ""
            if subname in {"StringField", "IntField", "FloatField", "BooleanField"}:
                fields.append(name)
    return sorted(fields)


def choose_dir(var: tk.StringVar):
    path = filedialog.askdirectory(title="Choose export folder")
    if path:
        var.set(path)


def values_to_text(sample: fo.Sample, fields, include_tags, as_one_line, delimiter, include_keys):
    """
    Build the text content for a sample given selected fields/tags.
    """
    parts = []

    for name in fields:
        val = sample.get(name, None)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            val_str = ", ".join(str(v) for v in val)
        else:
            val_str = str(val)
        if as_one_line:
            parts.append(f"{name}: {val_str}" if include_keys else val_str)
        else:
            parts.append(f"{name}: {val_str}")

    if include_tags:
        if sample.tags:
            tag_val = " ".join(sample.tags)
            if as_one_line:
                parts.append(f"tags: {tag_val}" if include_keys else tag_val)
            else:
                parts.append(f"tags: {tag_val}")

    return (delimiter.join(parts) if as_one_line else "\n".join(parts)) + ("\n" if parts else "")


def export_view_with_sidecars(
    view: DatasetView,
    export_dir: str,
    fields_to_txt,
    include_tags: bool,
    overwrite: bool,
    one_line: bool,
    delimiter: str,
    include_keys: bool,
):
    """
    Copy images in `view` to export_dir, and write a .txt sidecar per image
    with the selected fields/tags.
    Returns (num_images, num_txt).
    """
    export_path = Path(export_dir)
    if export_path.exists():
        if not overwrite and any(export_path.iterdir()):
            raise RuntimeError(
                f"Export directory exists and is not empty: {export_dir}\n"
                "Enable 'Overwrite' or choose an empty folder."
            )
    export_path.mkdir(parents=True, exist_ok=True)

    n_img = 0
    n_txt = 0
    for s in view.iter_samples(progress=True):
        src = Path(s.filepath)
        if not src.exists():
            continue

        dst_img = export_path / src.name
        if overwrite and dst_img.exists():
            try:
                dst_img.unlink()
            except Exception:
                pass
        shutil.copy2(src, dst_img)
        n_img += 1

        if fields_to_txt or include_tags:
            txt = values_to_text(
                s,
                fields=fields_to_txt,
                include_tags=include_tags,
                as_one_line=one_line,
                delimiter=delimiter,
                include_keys=include_keys,
            )
            if txt:
                with open(dst_img.with_suffix(".txt"), "w", encoding="utf-8") as f:
                    f.write(txt)
                n_txt += 1

    return n_img, n_txt


def register_export_as_dataset(export_dir: str, ds_name: str, overwrite: bool):
    """Add exported ImageDirectory back into FO DB as a dataset."""
    if overwrite and fo.dataset_exists(ds_name):
        fo.delete_dataset(ds_name)
    ds = fo.Dataset.from_dir(
        dataset_dir=export_dir,
        dataset_type=fot.ImageDirectory,
        name=ds_name,
    )
    return ds


# ---------------------- GUI ----------------------
root = tk.Tk()
root.title("FiftyOne — Export with sidecar TXT + register")
root.geometry("780x560")
root.resizable(False, False)

dataset_var = tk.StringVar()
view_var = tk.StringVar(value="< Entire dataset >")
export_dir_var = tk.StringVar()
overwrite_var = tk.BooleanVar(value=True)
include_tags_var = tk.BooleanVar(value=True)
one_line_var = tk.BooleanVar(value=True)
include_keys_var = tk.BooleanVar(value=False)
delimiter_var = tk.StringVar(value=", ")

register_var = tk.BooleanVar(value=True)
register_name_var = tk.StringVar(value="exported_dataset")

fields_listbox = None
available_fields = []


def on_dataset_change(_evt=None):
    ds_name = dataset_var.get().strip()
    views = list_saved_views(ds_name)
    view_cb["values"] = views
    view_var.set(views[0] if views else "< Entire dataset >")
    try:
        ds = fo.load_dataset(ds_name)
        global available_fields
        available_fields = get_primitive_sample_fields(ds)
        fields_listbox.delete(0, tk.END)
        for f in available_fields:
            fields_listbox.insert(tk.END, f)
    except Exception as e:
        messagebox.showerror("FiftyOne", f"Failed to load fields for '{ds_name}':\n{e}")


def choose_export_dir():
    choose_dir(export_dir_var)


def do_export():
    ds_name = dataset_var.get().strip()
    if not ds_name:
        messagebox.showwarning("Missing dataset", "Select a dataset.")
        return

    export_dir = export_dir_var.get().strip()
    if not export_dir:
        messagebox.showwarning("Missing folder", "Choose an export folder.")
        return
    os.makedirs(export_dir, exist_ok=True)

    try:
        ds = fo.load_dataset(ds_name)
        vname = view_var.get().strip()
        if vname == "< Entire dataset >" or not vname:
            view = ds.view()
            shown_view = "Entire dataset"
        else:
            view = ds.load_saved_view(vname)
            shown_view = f"Saved view: {vname}"
    except Exception as e:
        messagebox.showerror("FiftyOne", f"Failed to resolve view:\n{e}")
        return

    sel_indices = fields_listbox.curselection()
    selected_fields = [available_fields[i] for i in sel_indices]

    try:
        n_img, n_txt = export_view_with_sidecars(
            view=view,
            export_dir=export_dir,
            fields_to_txt=selected_fields,
            include_tags=bool(include_tags_var.get()),
            overwrite=bool(overwrite_var.get()),
            one_line=bool(one_line_var.get()),
            delimiter=delimiter_var.get(),
            include_keys=bool(include_keys_var.get()),
        )
    except Exception as e:
        messagebox.showerror("Export failed", f"{e}")
        return

    reg_msg = ""
    if register_var.get():
        name = register_name_var.get().strip()
        if not name:
            messagebox.showwarning("Missing name", "Provide a dataset name to register.")
            return
        try:
            ds2 = register_export_as_dataset(
                export_dir=export_dir,
                ds_name=name,
                overwrite=bool(overwrite_var.get()),
            )
            reg_msg = f"\nRegistered as FO dataset: {ds2.name}"
        except Exception as e:
            messagebox.showerror("Register failed", f"Export succeeded but registering failed:\n{e}")
            reg_msg = "\n(Registration failed)"

    messagebox.showinfo(
        "Done",
        f"Exported {n_img} images and wrote {n_txt} sidecars to:\n{export_dir}\n"
        f"View: {shown_view}{reg_msg}"
    )


# ---- UI layout ----
padx = 12
pady = 8

row1 = ttk.Frame(root, padding=(padx, pady)); row1.pack(fill="x")
ttk.Label(row1, text="Dataset:").pack(side="left")
dataset_cb = ttk.Combobox(row1, width=46, textvariable=dataset_var, state="readonly")
dataset_cb["values"] = list_datasets()
dataset_cb.pack(side="left", padx=8)
dataset_cb.bind("<<ComboboxSelected>>", on_dataset_change)
ttk.Button(row1, text="Refresh", command=lambda: dataset_cb.configure(values=list_datasets())).pack(side="left")

row2 = ttk.Frame(root, padding=(padx, 0, padx, pady)); row2.pack(fill="x")
ttk.Label(row2, text="View:").pack(side="left")
view_cb = ttk.Combobox(row2, width=46, textvariable=view_var, state="readonly")
view_cb["values"] = ["< Entire dataset >"]
view_cb.pack(side="left", padx=8)

field_frame = ttk.LabelFrame(root, text="Fields to include in TXT sidecars", padding=(padx, pady))
field_frame.pack(fill="both", expand=False, padx=10, pady=6)

fields_listbox = tk.Listbox(field_frame, selectmode=tk.MULTIPLE, width=48, height=10, exportselection=False)
fields_listbox.pack(side="left", padx=6, pady=4)

opts_col = ttk.Frame(field_frame); opts_col.pack(side="left", fill="y", padx=10)
ttk.Checkbutton(opts_col, text="Include sample tags", variable=include_tags_var).pack(anchor="w", pady=2)
ttk.Checkbutton(opts_col, text="Write as one line", variable=one_line_var).pack(anchor="w", pady=2)
ttk.Checkbutton(opts_col, text="Include field names (key: val)", variable=include_keys_var).pack(anchor="w", pady=2)
dl_row = ttk.Frame(opts_col); dl_row.pack(anchor="w", pady=2)
ttk.Label(dl_row, text="Delimiter:").pack(side="left")
ttk.Entry(dl_row, width=12, textvariable=delimiter_var).pack(side="left", padx=6)

row3 = ttk.LabelFrame(root, text="Export target", padding=(padx, pady))
row3.pack(fill="x", padx=10, pady=6)
ttk.Label(row3, text="Folder:").pack(side="left")
ttk.Entry(row3, width=54, textvariable=export_dir_var).pack(side="left", padx=8)
ttk.Button(row3, text="Browse…", command=choose_export_dir).pack(side="left")
ttk.Checkbutton(row3, text="Overwrite if exists", variable=overwrite_var).pack(side="left", padx=12)

row4 = ttk.LabelFrame(root, text="Register export as new dataset (optional)", padding=(padx, pady))
row4.pack(fill="x", padx=10, pady=6)
ttk.Checkbutton(row4, text="Add to FiftyOne DB", variable=register_var).pack(side="left")
ttk.Label(row4, text="Name:").pack(side="left", padx=(12, 4))
ttk.Entry(row4, width=28, textvariable=register_name_var).pack(side="left")

row5 = ttk.Frame(root, padding=(padx, pady))
row5.pack(fill="x")
ttk.Button(row5, text="Export", command=do_export).pack(side="right")
ttk.Button(row5, text="Close", command=root.destroy).pack(side="right", padx=8)

vals = dataset_cb["values"]
if vals:
    dataset_var.set(vals[0])
    on_dataset_change()

root.mainloop()
