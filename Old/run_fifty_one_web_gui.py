#!/usr/bin/env python
# Tk GUI to open/create a FiftyOne dataset, auto-tag by percentiles,
# tag near-duplicates with CLIP similarity, compute similarity density,
# compute CLIP embeddings + UMAP for the Embeddings panel, launch App,
# or delete a dataset.

# ---------- parse early flags so DB dir is set BEFORE importing fiftyone ----------
import os, argparse
ap = argparse.ArgumentParser("FiftyOne GUI launcher (Tk)", add_help=False)
ap.add_argument("--db_dir", default="", help="Optional: custom FiftyOne database dir")
ap.add_argument("--address", default="127.0.0.1", help="App bind address (default 127.0.0.1)")
ap.add_argument("--caption_ext", default=".txt", help="Caption sidecar extension (default .txt)")
early, _ = ap.parse_known_args()

# if early.db_dir:
#     os.environ["FIFTYONE_DATABASE_DIR"] = early.db_dir
# os.environ.setdefault("FIFTYONE_APP_ADDRESS", early.address)

os.environ["FIFTYONE_DATABASE_DIR"] = r"D:\FiftyOneDB"  # root .fiftyone folder, not ...\var\lib\mongo


# ---------- now safe to import fiftyone ----------
import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.brain as fob
import fiftyone.zoo as foz

from pathlib import Path
import numpy as np
from typing import Optional

# ---------- GUI ----------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CAPTION_EXT = early.caption_ext

# Auto-tagging by uniqueness
AUTO_LOW_DEFAULT = "keep_core,low_unique"
AUTO_HIGH_DEFAULT = "review_keep,outlier"
UNIQUENESS_FIELD = "uniqueness"   # where we store/read uniqueness

# Similarity / near-dup defaults
SIM_BRAIN_KEY_DEFAULT = "clip_sim"
SIM_MODEL_DEFAULT = "clip-vit-base32-torch"

# Embeddings defaults
EMB_FIELD_DEFAULT = "clip_emb"
UMAP_BRAIN_KEY_DEFAULT = "umap_all"

# ------------------------------ helpers ------------------------------ #
def gather_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def ensure_fields(ds: fo.Dataset):
    schema = ds.get_field_schema()
    if "caption" not in schema:
        ds.add_sample_field("caption", fo.StringField)
    if "relpath" not in schema:
        ds.add_sample_field("relpath", fo.StringField)
    if "topdir" not in schema:
        ds.add_sample_field("topdir", fo.StringField)

def _make_sample_from_path(img: Path, root_dir: Path) -> fo.Sample:
    img_abs = str(img.resolve())
    rel = img.relative_to(root_dir)
    topdir = rel.parts[0] if len(rel.parts) > 1 else ""
    s = fo.Sample(filepath=img_abs, relpath=str(rel).replace("\\", "/"), topdir=topdir)
    cap = img.with_suffix(CAPTION_EXT)
    if cap.exists():
        try:
            s["caption"] = cap.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            s["caption"] = cap.read_text(errors="ignore").strip()
    return s

def create_or_update_dataset(name: str, root_dir: Path, overwrite: bool) -> fo.Dataset:
    if fo.dataset_exists(name):
        if overwrite:
            fo.delete_dataset(name)
            ds = fo.Dataset(name)
            ds.persistent = True
        else:
            return fo.load_dataset(name)
    else:
        ds = fo.Dataset(name)
        ds.persistent = True
    

    ensure_fields(ds)
    imgs = gather_images(root_dir)
    samples = [_make_sample_from_path(img, root_dir) for img in imgs]
    if samples:
        ds.add_samples(samples)
    ds.compute_metadata()
    return ds

def upsert_from_folder(ds: fo.Dataset, root_dir: Path):
    """Add any new images from root_dir (recursive); update captions for existing files.
    Returns (added_count, updated_caption_count)."""
    ensure_fields(ds)
    root_dir = Path(root_dir)
    existing = set(ds.values("filepath"))  # absolute paths stored by FO
    added, updated = 0, 0

    imgs = gather_images(root_dir)
    new_samples = []

    for img in imgs:
        img_abs = str(Path(img).resolve())
        if img_abs not in existing:
            new_samples.append(_make_sample_from_path(img, root_dir))
            added += 1
        else:
            cap = Path(img).with_suffix(CAPTION_EXT)
            if cap.exists():
                try:
                    txt = cap.read_text(encoding="utf-8").strip()
                except UnicodeDecodeError:
                    txt = cap.read_text(errors="ignore").strip()
                s = ds.match(F("filepath") == img_abs).first()
                if s is not None and s.get("caption", "") != txt:
                    s["caption"] = txt
                    s.save()
                    updated += 1

    if new_samples:
        ds.add_samples(new_samples)

    ds.compute_metadata()
    return added, updated

def ensure_uniqueness(ds: fo.Dataset, field: str = UNIQUENESS_FIELD):
    # compute only if field missing or partially missing
    if field not in ds.get_field_schema() or ds.exists(field).count() < ds.count():
        fob.compute_uniqueness(ds, uniqueness_field=field)

def auto_tag_by_percentiles(
    ds: fo.Dataset,
    field: str = UNIQUENESS_FIELD,
    p_low: float = 10.0,
    p_high: float = 90.0,
    low_tags=("keep_core","low_unique"),
    high_tags=("review_keep","outlier"),
    clear_previous=True,
):
    """Tag low/high percentile samples by a numeric field (default: 'uniqueness')."""
    if ds.count() == 0:
        return 0, 0

    ensure_uniqueness(ds, field=field)

    vals = np.array(ds.values(field), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise RuntimeError(f"No numeric values found in field '{field}'")

    q_low, q_high = np.quantile(vals, [p_low / 100.0, p_high / 100.0])

    if clear_previous:
        to_clear = list(set(low_tags) | set(high_tags))
        if to_clear:
            ds.match_tags(to_clear).untag_samples(to_clear)

    lo = ds.match(F(field) <= float(q_low))
    hi = ds.match(F(field) >= float(q_high))

    if low_tags:
        lo.tag_samples(list(low_tags))
    if high_tags:
        hi.tag_samples(list(high_tags))
    ds.save()
    return lo.count(), hi.count()

# -------- similarity / near-duplicate -------- #
def ensure_similarity_index(ds: fo.Dataset, brain_key: str, model_name: str):
    """Compute similarity index if missing."""
    if brain_key not in ds.list_brain_runs():
        fob.compute_similarity(ds, brain_key=brain_key, model=model_name)

def tag_near_duplicates(
    ds: fo.Dataset,
    brain_key: str = SIM_BRAIN_KEY_DEFAULT,
    model_name: str = SIM_MODEL_DEFAULT,
    k: int = 6,
    max_dist: float = 0.20,
    seed_tag: str = "seed",
    dup_tag: str = "near_dup",
    clear_previous: bool = True,
):
    """Tags one representative per group as `seed` and close neighbors as `near_dup`."""
    if ds.count() == 0:
        return (0, 0)

    ensure_similarity_index(ds, brain_key, model_name)

    if clear_previous:
        ds.match_tags([seed_tag, dup_tag]).untag_samples([seed_tag, dup_tag])

    seen = set()
    seeds = 0
    dups = 0
    dist_field = "simdist"  # valid field name (cannot start with '_')

    for s in ds:
        if s.id in seen:
            continue

        view = ds.sort_by_similarity(
            s.id, brain_key=brain_key, k=int(k) + 1, dist_field=dist_field
        )

        neighbors = [
            r for r in view[1:]
            if getattr(r, dist_field, None) is not None
            and float(getattr(r, dist_field)) <= float(max_dist)
        ]

        if neighbors:
            if seed_tag not in s.tags:
                s.tags = list(set(s.tags + [seed_tag]))
                s.save()
            seeds += 1

            for r in neighbors:
                if dup_tag not in r.tags:
                    r.tags = list(set(r.tags + [dup_tag]))
                    r.save()
                    dups += 1
                seen.add(r.id)

    # remove the temporary distance field
    if dist_field in ds.get_field_schema():
        ds.delete_sample_field(dist_field)

    return (seeds, dups)

def prune_near_dups(ds: fo.Dataset, dup_tag: str = "near_dup"):
    """Delete all samples tagged as near-duplicates (does NOT delete files on disk)."""
    view = ds.match_tags([dup_tag])
    n = view.count()
    if n > 0:
        ds.delete_samples(view)
    return n

# -------- similarity density (kNN) -------- #
def compute_sim_density(
    ds: fo.Dataset,
    brain_key: str,
    model_name: str,
    k: int = 12,
    max_dist: Optional[float] = None,
    out_field: str = "sim_density",
):
    """
    Writes a density score per sample based on CLIP similarity:
      score = 1 / (eps + mean distance to k nearest neighbors)
      (optionally only neighbors within max_dist are used)

    Then you can Embeddings -> Color by -> this field.
    """
    ensure_similarity_index(ds, brain_key, model_name)

    temp = "simdist_tmp"
    ids, scores = [], []

    for s in ds:
        v = ds.sort_by_similarity(s.id, brain_key=brain_key, k=int(k) + 1, dist_field=temp)
        dists = [getattr(r, temp) for r in v[1:] if getattr(r, temp, None) is not None]
        if max_dist is not None:
            dists = [d for d in dists if d <= float(max_dist)]
        score = 0.0 if not dists else 1.0 / (1e-6 + (sum(dists) / len(dists)))
        ids.append(s.id)
        scores.append(float(score))

    # create field if missing
    if out_field not in ds.get_field_schema():
        ds.add_sample_field(out_field, fo.FloatField)

    # values first, ids second
    ds.set_values(out_field, scores, ids)

    if temp in ds.get_field_schema():
        ds.delete_sample_field(temp)

    return len(scores)

# -------- embeddings + UMAP visualization -------- #
def compute_clip_embeddings_and_umap(
    ds_name: str,
    emb_field: str = EMB_FIELD_DEFAULT,
    umap_brain_key: str = UMAP_BRAIN_KEY_DEFAULT,
    model_name: str = SIM_MODEL_DEFAULT,
):
    """
    Computes CLIP embeddings into `emb_field`, then a UMAP visualization
    with brain key `umap_brain_key` so the OSS App Embeddings panel works.
    """
    ds = fo.load_dataset(ds_name)

    # Only compute embeddings if field missing or incomplete
    need_emb = (emb_field not in ds.get_field_schema()) or (ds.exists(emb_field).count() < ds.count())
    if need_emb:
        model = foz.load_zoo_model(model_name)   # CPU or GPU automatically
        ds.compute_embeddings(model, embeddings_field=emb_field)

    # Create/update the 2D visualization (UMAP)
    fob.compute_visualization(ds, embeddings=emb_field, brain_key=umap_brain_key)
    return emb_field, umap_brain_key

# ------------------------------ GUI ------------------------------ #
def main():
    root = tk.Tk()
    root.title("FiftyOne — Open/Create, Auto-tag, Near-dups, Density, Embeddings, Launch")
    root.geometry("820x780")
    root.resizable(False, False)

    # --- vars ---
    db_dir = fo.config.database_dir or ""
    ds_names = tk.Variable(value=tuple(sorted(fo.list_datasets())))
    selected_ds = tk.StringVar(value="")
    new_name = tk.StringVar(value="")
    new_root = tk.StringVar(value="")
    overwrite = tk.BooleanVar(value=False)
    address = tk.StringVar(value=early.address)

    # auto-tag controls
    at_enable_after_create = tk.BooleanVar(value=False)
    at_enable_after_add = tk.BooleanVar(value=True)
    at_low = tk.DoubleVar(value=10.0)
    at_high = tk.DoubleVar(value=90.0)
    at_low_tags = tk.StringVar(value=AUTO_LOW_DEFAULT)
    at_high_tags = tk.StringVar(value=AUTO_HIGH_DEFAULT)
    at_clear = tk.BooleanVar(value=True)

    # Add-to-existing controls
    add_root = tk.StringVar(value="")

    # near-dup controls
    nd_brain_key = tk.StringVar(value=SIM_BRAIN_KEY_DEFAULT)
    nd_model = tk.StringVar(value=SIM_MODEL_DEFAULT)
    nd_k = tk.IntVar(value=6)
    nd_maxdist = tk.DoubleVar(value=0.20)
    nd_seed_tag = tk.StringVar(value="seed")
    nd_dup_tag = tk.StringVar(value="near_dup")
    nd_clear = tk.BooleanVar(value=True)

    # density controls
    dens_k = tk.IntVar(value=12)
    dens_maxdist = tk.StringVar(value="")      # empty = no threshold
    dens_field = tk.StringVar(value="sim_density")

    # embeddings controls
    emb_field = tk.StringVar(value=EMB_FIELD_DEFAULT)
    umap_key = tk.StringVar(value=UMAP_BRAIN_KEY_DEFAULT)
    emb_model = tk.StringVar(value=SIM_MODEL_DEFAULT)

    # --- callbacks ---
    def refresh_list():
        ds_names.set(tuple(sorted(fo.list_datasets())))
        if selected_ds.get() not in ds_names.get():
            selected_ds.set(ds_names.get()[0] if ds_names.get() else "")

    def choose_root_new():
        path = filedialog.askdirectory(title="Choose dataset root folder")
        if path:
            new_root.set(path)

    def choose_root_add():
        path = filedialog.askdirectory(title="Choose folder to add/refresh into selected dataset")
        if path:
            add_root.set(path)

    def launch_for_dataset(ds: fo.Dataset):
        try:
            messagebox.showinfo("Launching", f"Launching FiftyOne App for '{ds.name}'")
            root.withdraw()
            session = fo.launch_app(ds, address=address.get() or "127.0.0.1")
            session.wait()
        finally:
            try:
                root.destroy()
            except:
                pass

    def open_existing():
        name = (selected_ds.get() or "").strip()
        if not name:
            messagebox.showwarning("Select dataset", "Please choose a dataset from the dropdown.")
            return
        try:
            ds = fo.load_dataset(name)
            launch_for_dataset(ds)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open dataset:\n{e}")

    def delete_existing():
        name = (selected_ds.get() or "").strip()
        if not name:
            messagebox.showwarning("Select dataset", "Please choose a dataset to delete.")
            return
        if not fo.dataset_exists(name):
            messagebox.showwarning("Not found", f"Dataset '{name}' does not exist.")
            return
        if not messagebox.askyesno("Confirm delete",
                                   f"Delete dataset '{name}'?\nThis removes its index/DB entries (not your image files)."):
            return
        try:
            fo.delete_dataset(name)
            messagebox.showinfo("Deleted", f"Dataset '{name}' deleted.")
            refresh_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete dataset:\n{e}")

    def add_to_existing():
        name = (selected_ds.get() or "").strip()
        folder = (add_root.get() or "").strip().strip('"')
        if not name:
            messagebox.showwarning("Select dataset", "Please choose a dataset from the dropdown.")
            return
        if not folder:
            messagebox.showwarning("Folder required", "Choose a folder to add/refresh.")
            return
        p = Path(folder)
        if not p.exists():
            messagebox.showerror("Folder not found", f"Path does not exist:\n{p}")
            return
        try:
            ds = fo.load_dataset(name)
            added, updated = upsert_from_folder(ds, p)

            tag_msg = ""
            if at_enable_after_add.get():
                lo_tags = tuple(t.strip() for t in at_low_tags.get().split(",") if t.strip())
                hi_tags = tuple(t.strip() for t in at_high_tags.get().split(",") if t.strip())
                lo, hi = auto_tag_by_percentiles(
                    ds,
                    field=UNIQUENESS_FIELD,
                    p_low=float(at_low.get()),
                    p_high=float(at_high.get()),
                    low_tags=lo_tags,
                    high_tags=hi_tags,
                    clear_previous=bool(at_clear.get()),
                )
                tag_msg = f"\nAuto-tagged: low {lo} / high {hi}"

            messagebox.showinfo("Add/refresh complete",
                                f"Dataset '{name}':\nAdded {added}, Updated captions {updated}.{tag_msg}")
        except Exception as e:
            messagebox.showerror("Error", f"Add/refresh failed:\n{e}")

    def create_new():
        name = (new_name.get() or "").strip()
        folder = (new_root.get() or "").strip().strip('"')
        if not name:
            messagebox.showwarning("Name required", "Please enter a dataset name.")
            return
        if not folder:
            messagebox.showwarning("Folder required", "Please choose a root folder to ingest.")
            return
        p = Path(folder)
        if not p.exists():
            messagebox.showerror("Folder not found", f"Path does not exist:\n{p}")
            return
        try:
            ds = create_or_update_dataset(name, p, overwrite.get())

            if at_enable_after_create.get():
                lo_tags = tuple(t.strip() for t in at_low_tags.get().split(",") if t.strip())
                hi_tags = tuple(t.strip() for t in at_high_tags.get().split(",") if t.strip())
                lo, hi = auto_tag_by_percentiles(
                    ds,
                    field=UNIQUENESS_FIELD,
                    p_low=float(at_low.get()),
                    p_high=float(at_high.get()),
                    low_tags=lo_tags,
                    high_tags=hi_tags,
                    clear_previous=bool(at_clear.get()),
                )
                messagebox.showinfo(
                    "Auto-tag complete",
                    f"Tagged {lo} low-percentile and {hi} high-percentile samples."
                )

            launch_for_dataset(ds)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create/open dataset:\n{e}")

    def autotag_existing():
        name = (selected_ds.get() or "").strip()
        if not name:
            messagebox.showwarning("Select dataset", "Please choose a dataset from the dropdown.")
            return
        try:
            ds = fo.load_dataset(name)
            lo_tags = tuple(t.strip() for t in at_low_tags.get().split(",") if t.strip())
            hi_tags = tuple(t.strip() for t in at_high_tags.get().split(",") if t.strip())
            lo, hi = auto_tag_by_percentiles(
                ds,
                field=UNIQUENESS_FIELD,
                p_low=float(at_low.get()),
                p_high=float(at_high.get()),
                low_tags=lo_tags,
                high_tags=hi_tags,
                clear_previous=bool(at_clear.get()),
            )
            messagebox.showinfo("Auto-tag complete",
                                f"Dataset '{name}':\n"
                                f"  low ≤ {at_low.get():.1f}%% → {', '.join(lo_tags)} [{lo}]\n"
                                f"  high ≥ {at_high.get():.1f}%% → {', '.join(hi_tags)} [{hi}]")
        except Exception as e:
            messagebox.showerror("Error", f"Auto-tag failed:\n{e}")

    def run_near_dups():
        name = (selected_ds.get() or "").strip()
        if not name:
            messagebox.showwarning("Select dataset", "Please choose a dataset from the dropdown.")
            return
        try:
            ds = fo.load_dataset(name)
            seeds, dups = tag_near_duplicates(
                ds,
                brain_key=nd_brain_key.get().strip() or SIM_BRAIN_KEY_DEFAULT,
                model_name=nd_model.get().strip() or SIM_MODEL_DEFAULT,
                k=int(nd_k.get()),
                max_dist=float(nd_maxdist.get()),
                seed_tag=nd_seed_tag.get().strip() or "seed",
                dup_tag=nd_dup_tag.get().strip() or "near_dup",
                clear_previous=bool(nd_clear.get()),
            )
            messagebox.showinfo("Near-duplicate tagging",
                                f"Tagged seeds: {seeds}\nTagged near_dups: {dups}\n"
                                f"(Use Filters → sample tags to view)")
        except Exception as e:
            messagebox.showerror("Error", f"Near-dup tagging failed:\n{e}")

    def run_prune_dups():
        name = (selected_ds.get() or "").strip()
        if not name:
            messagebox.showwarning("Select dataset", "Please choose a dataset from the dropdown.")
            return
        try:
            ds = fo.load_dataset(name)
            tag = nd_dup_tag.get().strip() or "near_dup"
            v = ds.match_tags([tag]); n = v.count()
            if n == 0:
                messagebox.showinfo("Prune near-dups", "No samples with tag to delete.")
                return
            if not messagebox.askyesno("Confirm prune",
                f"Delete {n} samples tagged '{tag}' from dataset '{name}'?\n"
                "This removes them from the dataset (image files on disk are NOT deleted)."):
                return
            deleted = prune_near_dups(ds, dup_tag=tag)
            messagebox.showinfo("Prune near-dups", f"Deleted {deleted} samples with tag '{tag}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Prune failed:\n{e}")

    def run_density():
        name = (selected_ds.get() or "").strip()
        if not name:
            messagebox.showwarning("Select dataset", "Choose a dataset first.")
            return
        try:
            ds = fo.load_dataset(name)
            k = int(dens_k.get())
            md = dens_maxdist.get().strip()
            md_val = None if md == "" else float(md)
            field = dens_field.get().strip() or "sim_density"
            n = compute_sim_density(
                ds,
                brain_key=nd_brain_key.get().strip() or SIM_BRAIN_KEY_DEFAULT,
                model_name=nd_model.get().strip() or SIM_MODEL_DEFAULT,
                k=k,
                max_dist=md_val,
                out_field=field,
            )
            messagebox.showinfo(
                "Density",
                f"Computed density for {n} samples → field '{field}'.\n"
                "Tip: Embeddings → Color by → that field."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Density computation failed:\n{e}")

    def run_compute_embeddings():
        name = (selected_ds.get() or "").strip()
        if not name:
            messagebox.showwarning("Select dataset", "Choose a dataset first.")
            return
        try:
            ef, uk = compute_clip_embeddings_and_umap(
                name,
                emb_field=emb_field.get().strip() or EMB_FIELD_DEFAULT,
                umap_brain_key=umap_key.get().strip() or UMAP_BRAIN_KEY_DEFAULT,
                model_name=emb_model.get().strip() or SIM_MODEL_DEFAULT,
            )
            messagebox.showinfo(
                "Embeddings ready",
                f"Embeddings field: {ef}\nUMAP key: {uk}\n\n"
                "Open the App → Curate ▸ Embeddings → choose that key."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Embedding computation failed:\n{e}")

    # ---------- layout ----------
    pad = {"padx": 10, "pady": 6}

    top = tk.Frame(root, padx=10, pady=10)
    top.pack(fill="x")
    tk.Label(top, text=f"DB: {db_dir}", fg="#888").pack(anchor="w")

    # Existing
    f1 = ttk.LabelFrame(root, text="Open / manage existing dataset")
    f1.pack(fill="x", padx=10, pady=6)

    row1 = tk.Frame(f1); row1.pack(fill="x", **pad)
    ttk.Label(row1, text="Dataset:").pack(side="left")
    cb = ttk.Combobox(row1, textvariable=selected_ds, values=tuple(sorted(fo.list_datasets())), width=42, state="readonly")
    cb.pack(side="left", padx=6)
    ttk.Button(row1, text="Refresh", command=refresh_list).pack(side="left", padx=4)
    ttk.Button(row1, text="Open", command=open_existing).pack(side="right")
    ttk.Button(row1, text="Delete", command=delete_existing).pack(side="right", padx=4)

    # Add/refresh into existing
    row1b = tk.Frame(f1); row1b.pack(fill="x", **pad)
    ttk.Label(row1b, text="Add/refresh from folder:").pack(side="left")
    ttk.Entry(row1b, textvariable=add_root, width=38).pack(side="left", padx=6)
    ttk.Button(row1b, text="Browse…", command=choose_root_add).pack(side="left")
    ttk.Checkbutton(row1b, text="Auto-tag after add", variable=at_enable_after_add).pack(side="left", padx=10)
    ttk.Button(row1b, text="Run add/refresh", command=add_to_existing).pack(side="right")

    # Auto-tag controls
    f_at = ttk.LabelFrame(root, text="Auto-tag by percentiles (uses 'uniqueness'; computes if missing)")
    f_at.pack(fill="x", padx=10, pady=6)

    rowA = tk.Frame(f_at); rowA.pack(fill="x", **pad)
    ttk.Label(rowA, text="Low ≤ %").pack(side="left")
    ttk.Entry(rowA, textvariable=at_low, width=6).pack(side="left", padx=4)
    ttk.Label(rowA, text="→ Tags:").pack(side="left")
    ttk.Entry(rowA, textvariable=at_low_tags, width=28).pack(side="left", padx=6)

    ttk.Label(rowA, text="High ≥ %").pack(side="left", padx=(20,0))
    ttk.Entry(rowA, textvariable=at_high, width=6).pack(side="left", padx=4)
    ttk.Label(rowA, text="→ Tags:").pack(side="left")
    ttk.Entry(rowA, textvariable=at_high_tags, width=28).pack(side="left", padx=6)

    rowB = tk.Frame(f_at); rowB.pack(fill="x", **pad)
    ttk.Checkbutton(rowB, text="Clear these tags before re-tagging", variable=at_clear).pack(side="left")
    ttk.Button(rowB, text="Run on selected dataset", command=autotag_existing).pack(side="right")

    # Near-duplicate section
    f_nd = ttk.LabelFrame(root, text="Near-duplicate tagging (CLIP similarity)")
    f_nd.pack(fill="x", padx=10, pady=6)

    r1 = tk.Frame(f_nd); r1.pack(fill="x", **pad)
    ttk.Label(r1, text="Brain key:").pack(side="left")
    ttk.Entry(r1, textvariable=nd_brain_key, width=18).pack(side="left", padx=6)
    ttk.Label(r1, text="Model:").pack(side="left")
    ttk.Entry(r1, textvariable=nd_model, width=28).pack(side="left", padx=6)

    r2 = tk.Frame(f_nd); r2.pack(fill="x", **pad)
    ttk.Label(r2, text="Neighbors K:").pack(side="left")
    ttk.Entry(r2, textvariable=nd_k, width=6).pack(side="left", padx=6)
    ttk.Label(r2, text="Max distance:").pack(side="left")
    ttk.Entry(r2, textvariable=nd_maxdist, width=8).pack(side="left", padx=6)
    ttk.Checkbutton(r2, text="Clear previous seed/near_dup tags", variable=nd_clear).pack(side="left", padx=10)
    ttk.Button(r2, text="Tag near-duplicates", command=run_near_dups).pack(side="right")

    r3 = tk.Frame(f_nd); r3.pack(fill="x", **pad)
    ttk.Label(r3, text="Seed tag:").pack(side="left")
    ttk.Entry(r3, textvariable=nd_seed_tag, width=18).pack(side="left", padx=6)
    ttk.Label(r3, text="Near-dup tag:").pack(side="left")
    ttk.Entry(r3, textvariable=nd_dup_tag, width=18).pack(side="left", padx=6)
    ttk.Button(r3, text="Prune: delete all near_dups", command=run_prune_dups).pack(side="right")

    # Density section
    f_den = ttk.LabelFrame(root, text="Similarity density (kNN)")
    f_den.pack(fill="x", padx=10, pady=6)

    dr1 = tk.Frame(f_den); dr1.pack(fill="x", **pad)
    ttk.Label(dr1, text="k:").pack(side="left")
    ttk.Entry(dr1, textvariable=dens_k, width=6).pack(side="left", padx=6)
    ttk.Label(dr1, text="max_dist (optional):").pack(side="left")
    ttk.Entry(dr1, textvariable=dens_maxdist, width=10).pack(side="left", padx=6)
    ttk.Label(dr1, text="Field:").pack(side="left")
    ttk.Entry(dr1, textvariable=dens_field, width=16).pack(side="left", padx=6)
    ttk.Button(dr1, text="Compute density", command=run_density).pack(side="right")

    # Embeddings section
    f_emb = ttk.LabelFrame(root, text="Embeddings (CLIP) + UMAP visualization for App Embeddings panel")
    f_emb.pack(fill="x", padx=10, pady=6)

    e1 = tk.Frame(f_emb); e1.pack(fill="x", **pad)
    ttk.Label(e1, text="Embeddings field:").pack(side="left")
    ttk.Entry(e1, textvariable=emb_field, width=16).pack(side="left", padx=6)
    ttk.Label(e1, text="UMAP brain key:").pack(side="left")
    ttk.Entry(e1, textvariable=umap_key, width=16).pack(side="left", padx=6)
    ttk.Label(e1, text="Model:").pack(side="left")
    ttk.Entry(e1, textvariable=emb_model, width=24).pack(side="left", padx=6)
    ttk.Button(e1, text="Compute CLIP embeddings + UMAP", command=run_compute_embeddings).pack(side="right")

    # Create new
    f2 = ttk.LabelFrame(root, text="Create new dataset")
    f2.pack(fill="x", padx=10, pady=6)

    row2 = tk.Frame(f2); row2.pack(fill="x", **pad)
    ttk.Label(row2, text="Name:").pack(side="left")
    ttk.Entry(row2, textvariable=new_name, width=38).pack(side="left", padx=6)
    ttk.Checkbutton(row2, text="Overwrite if exists", variable=overwrite).pack(side="right")

    row3 = tk.Frame(f2); row3.pack(fill="x", **pad)
    ttk.Label(row3, text="Root folder:").pack(side="left")
    ttk.Entry(row3, textvariable=new_root, width=38).pack(side="left", padx=6)
    ttk.Button(row3, text="Browse…", command=choose_root_new).pack(side="left")

    row4 = tk.Frame(f2); row4.pack(fill="x", **pad)
    ttk.Label(row4, text="App address:").pack(side="left")
    ttk.Entry(row4, textvariable=address, width=18).pack(side="left", padx=6)
    ttk.Checkbutton(row4, text="Auto-tag after create", variable=at_enable_after_create).pack(side="left", padx=12)
    ttk.Button(row4, text="Create + Open", command=create_new).pack(side="right")

    # init
    refresh_list()
    root.mainloop()

if __name__ == "__main__":
    main()
