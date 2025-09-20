IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DB_DIR = r"D:\FiftyOneDB"
DATASET_DIR = r"E:\Work\Lorna_daaset_extension"
DATASET_NAME = "lorna_ext"

import os

# If you set DB dir in this file, do it BEFORE importing fiftyone:
# os.environ["FIFTYONE_DATABASE_DIR"] = DB_DIR
# import fiftyone as fo
os.environ["FIFTYONE_DATABASE_DIR"] = DB_DIR  # root .fiftyone folder, not ...\var\lib\mongo

import fiftyone as fo

# --- Vars ---
dataset: fo.Dataset
session: fo.Session

def create_dataset_from_dir():
     dataset = fo.Dataset.from_dir(
        dataset_dir=DATASET_DIR,
        dataset_type=fo.types.ImageDirectory,
        name=DATASET_NAME,)
     dataset.persistent = True
     return dataset

if DATASET_NAME in fo.list_datasets():
    dataset = fo.load_dataset(DATASET_NAME)
    print(f"\nDataset '{DATASET_NAME}' already exist in DB: {DB_DIR}")
    answer = input(f"\nReimport semples from {DATASET_DIR}? y/n:\n")

    if answer == "y":
        fo.delete_dataset(DATASET_NAME)
        dataset = create_dataset_from_dir()
else:
    create_dataset_from_dir()

sample_count = dataset.count()
print(f"Samples in dataset {sample_count}")


#session = fo.launch_app(dataset)
#session.wait()