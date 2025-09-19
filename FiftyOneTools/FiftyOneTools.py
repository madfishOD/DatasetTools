# -*- coding: utf-8 -*-
# import os
# import sys
# from pathlib import Path

# import fiftyone as fo
# import fiftyone.zoo as foz

# # Load a dataset (or create your own)
# dataset = foz.load_zoo_dataset("quickstart")

# # Launch the FiftyOne App to visualize the dataset
# session = fo.launch_app(dataset)

# # If running in a script and you want the app to remain open
# # until you manually close it, use session.wait()
# session.wait()

import os.path
from pathlib import Path
import fo_utils as foUtils

image_paths = foUtils.collect_image_paths(Path(r"E:\Work\Lorna_daaset_extension\Test"))

plan = foUtils.rename_files(image_paths, name="Lorna", fmt="000", start=1, dry_run=True)
for old_path, new_path in plan:
    print(f"{old_path} -> {new_path}")