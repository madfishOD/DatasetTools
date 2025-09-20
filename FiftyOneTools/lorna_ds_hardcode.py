IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DB_DIR = r"D:\FiftyOneDB"
DATASET_DIR = r"E:\Work\Lorna_daaset_extension"
DATASET_NAME = "lorna_ext"

import os
import time
from pathlib import Path

# If you set DB dir in this file, do it BEFORE importing fiftyone:
# os.environ["FIFTYONE_DATABASE_DIR"] = DB_DIR
# import fiftyone as fo
os.environ["FIFTYONE_DATABASE_DIR"] = DB_DIR  # root .fiftyone folder, not ...\var\lib\mongo

import fiftyone as fo

# --- Vars ---
dataset: fo.Dataset
session: fo.Session

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

def create_dataset_from_dir():
     dataset = fo.Dataset.from_dir(
        dataset_dir=DATASET_DIR,
        dataset_type=fo.types.ImageDirectory,
        name=DATASET_NAME,)
     dataset.persistent = True
     return dataset

if DATASET_NAME in fo.list_datasets():
    dataset = fo.load_dataset(DATASET_NAME)
    sample_count = dataset.count()
    images_count = count_images(DATASET_DIR)
    answer = ask_yn(
                    f"\nDataset '{DATASET_NAME}' already exists in DB -> {DB_DIR}\n"
                    f"Dataset samples:{sample_count} images in dir:{images_count}\n"
                    f"Reimport samples from {DATASET_DIR}?",
                    False
                    )
    if answer:
        fo.delete_dataset(DATASET_NAME)
        dataset = create_dataset_from_dir()
    else:
        create_dataset_from_dir()

# sample_count = dataset.count()
# print(f"Samples in dataset {sample_count}")


#session = fo.launch_app(dataset)
#session.wait()