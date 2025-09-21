IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DB_DIR = r"D:\FiftyOneDB"
DATASET_DIR = r"E:\Work\Lorna_daaset_extension"
DATASET_NAME = "lorna_ext"

import os
import time
from pathlib import Path

from fiftyone.core import session

# If you set DB dir in this file, do it BEFORE importing fiftyone:
# os.environ["FIFTYONE_DATABASE_DIR"] = DB_DIR
# import fiftyone as fo
os.environ["FIFTYONE_DATABASE_DIR"] = DB_DIR  # root .fiftyone folder, not ...\var\lib\mongo

import fiftyone as fo

# --- Vars ---
# dataset: fo.Dataset
# session: fo.Session

# ---------- Console helpers ----------
def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def ask_yn(prompt: str, default: bool | None = None) -> bool:
    """
    Ask a yes/no question with validation.
    default=True -> [Y/n], default=False -> [Y/N], default=None -> [y/n] (no default)
    """
    if default is True:
        suffix = " [Y/n] "
    elif default is False:
        suffix = " [y/N] "
    else:
        suffix = " [y/n] "

    while True:
        ans = input(prompt + suffix).strip().lower()
        if not ans and default is not None:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")

def count_images(root) -> int:
    root = Path(root)
    return sum(
        1
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

def collect_image_paths(root: str) -> list[str]:
    r = Path(root)
    return [str(p) for p in r.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS]

def create_dataset_from_dir(root: str) -> fo.Dataset:
    ds = fo.Dataset(name=DATASET_NAME)
    paths = collect_image_paths(root)
    ds.add_images(paths)   
    ds.persistent = True
    ds.save()
    return ds

def check_dataset_by_name (name: str) -> fo.Dataset:
    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    else:
        return None

def load_or_create_dataset(name: str, root: str) -> fo.Dataset:
    ds = check_dataset_by_name(name)
    if ds is None:
        ds = create_dataset_from_dir(root, True)
    else:
        sample_count = ds.count()
        images_count = count_images(DATASET_DIR)
        import_images = ask_yn(
                        f"\nDataset '{DATASET_NAME}' already exists in DB -> {DB_DIR}\n"
                        f"Dataset samples:{sample_count} images in dir:{images_count}\n"
                        f"Reimport samples from {DATASET_DIR}?",
                        False
                        )
        if import_images:
            paths = collect_image_paths(root)
            ds.add_images(paths)
            ds.save()
            print(f"Dataset samples imported: {ds.count()}")
    return ds

def main():

    dataset_exists = fo.dataset_exists(DATASET_NAME)
    print(f"dataset_exists = {dataset_exists}")

    image_paths = collect_image_paths(DATASET_DIR)
    if(len(image_paths)>0):
        new_samples = []
        for p in collect_image_paths(DATASET_DIR):
            new_samples.append(fo.Sample(filepath=p))

    if(dataset_exists):
        dataset = fo.load_dataset(DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' loaded from DB:{DB_DIR}"
              f"\nSamples = {dataset.count()}")

        if(dataset.count()<1 and len(new_samples)>0):
            dataset.add_samples(new_samples)
            dataset.save()
            print(f"Samples added: {dataset.count()}")

        else:
            print(f"Existing dataset {dataset}")
    else:
        dataset = fo.Dataset(name=DATASET_NAME, persistent=True)
        dataset.media_type = "image"
        if(len(new_samples)>0):
            dataset.add_samples(new_samples)
            dataset.compute_metadata()
            dataset.save()

    # session = fo.launch_app(dataset)
    # session.wait()

if __name__ == "__main__":
    main()