IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DB_DIR = r"D:\FiftyOneDB"
DATASET_DIR = r"E:\Work\Lorna_daaset_extension"
DATASET_NAME = "lorna_ext"

import os
import time
from pathlib import Path

from fiftyone.core import session
import fiftyone.core.session.events as foe

# If you set DB dir in this file, do it BEFORE importing fiftyone:
# os.environ["FIFTYONE_DATABASE_DIR"] = DB_DIR
# import fiftyone as fo
os.environ["FIFTYONE_DATABASE_DIR"] = DB_DIR  # root .fiftyone folder, not ...\var\lib\mongo

import fiftyone as fo

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
    ds.compute_metadata()
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
        if(sample_count != images_count):
            reimport_images = ask_yn(
                            f"\nDataset '{DATASET_NAME}' already exists in DB -> {DB_DIR}\n"
                            f"Dataset samples:{sample_count} images in dir:{images_count}\n"
                            f"Reimport samples from {DATASET_DIR}?",
                            False
                            )
            if reimport_images:
                fo.delete_dataset(name)
                ds = create_dataset_from_dir(root)
                print(f"Dataset samples imported: {ds.count()}")
    return ds

def main():

    dataset = load_or_create_dataset(DATASET_NAME, DATASET_DIR)

    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()