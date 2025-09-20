import os

os.environ["FIFTYONE_DATABASE_DIR"] = r"D:\FiftyOneDB"  # root .fiftyone folder, not ...\var\lib\mongo

import fiftyone as fo

# --- Vars ---
dataset_dir = r"E:\Work\Lorna_daaset_extension"
name = "lorna_ext"
dataset: fo.Dataset
session: fo.Session

if name in fo.list_datasets():
    dataset = fo.load_dataset(name)
else:
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.ImageDirectory,
        name=name,
    )
dataset.persistent = True

session = fo.launch_app(dataset)
session.wait()