IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DB_DIR = r"D:\FiftyOneDB"
DATASET_DIR = r"E:\Work\Lorna_daaset_extension"
DATASET_NAME = "lorna_ext"
MODEL_ID = "microsoft/Florence-2-large"  # or "microsoft/Florence-2-large"
TASK_TOKEN = "<DETAILED_CAPTION>"   # Florence-2 task for captions
BATCH_SIZE = 1                      # 1 is safest; you can try >1 on big GPUs
MAX_NEW_TOKENS = 96                 # caption length budget
CAPTION_FIELD = "florence2_caption"

import os
import time
from pathlib import Path
from typing import Optional

from fiftyone.core import session
import fiftyone.core.session.events as foe

# If you set DB dir in this file, do it BEFORE importing fiftyone:
# os.environ["FIFTYONE_DATABASE_DIR"] = DB_DIR
# import fiftyone as fo
os.environ["FIFTYONE_DATABASE_DIR"] = DB_DIR  # root .fiftyone folder, not ...\var\lib\mongo

import fiftyone as fo
import torch
from fiftyone import ViewField as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

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

# -------------- Florence ------------
def load_model(model_id: str):
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # use half precision only on GPU
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # IMPORTANT: no device_map here; just load and then .to(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,            # fp16 on GPU, fp32 on CPU
        trust_remote_code=True,
        low_cpu_mem_usage=True,       # optional, helps during load
    )

    # Move to device explicitly and set eval()
    model.to(device)
    model.eval()

    return processor, model, device

def caption_with_florence(dataset: fo.Dataset) -> None:
    """Caption all samples that don't yet have CAPTION_FIELD."""
    if not ask_yn("Run Florence2 captioning?", default=False):
        print("Skipped captioning.")
        return

    # samples where CAPTION_FIELD does not exist
    # use either line below; both are equivalent
    # view = dataset.match(fo.ViewField(CAPTION_FIELD).exists() == False)
    view = dataset.match(~F(CAPTION_FIELD).exists())

    if len(view) == 0:
        print("Nothing to caption (all samples already have captions, or view is empty).")
        return

    print(f"Preparing to caption {len(view)} images with Florence-2: {MODEL_ID}")

    # load once
    processor, model, device = load_model(MODEL_ID)
    print(f"Model loaded on {device}")

    updated = 0
    for sample in view.iter_samples(progress=True):
        if sample.media_type != "image":
            continue
        if getattr(sample, CAPTION_FIELD, None):  # already has a caption
            continue

        cap = caption_image(sample.filepath, processor, model)
        if not cap:
            continue

        sample[CAPTION_FIELD] = cap
        sample.save()
        updated += 1

    print(f"Done. Captions written: {updated}")

@torch.inference_mode()
def caption_image(img_path: str, processor, model) -> Optional[str]:
    image = Image.open(img_path).convert("RGB")

    # Build inputs and move to the SAME device/dtype as the model
    inputs = processor(images=image, text=TASK_TOKEN, return_tensors="pt")
    inputs = {
        k: (v.to(device=model.device, dtype=model.dtype) if v.is_floating_point()
            else v.to(device=model.device))
        for k, v in inputs.items()
    }

    # Optional: with torch.autocast("cuda"):
    generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    # Florence sometimes prefixes "CAPTION:" — strip if present
    if text.upper().startswith("CAPTION:"):
        text = text.split(":", 1)[1].strip()

    return text or None

def main():

    # Load the model once
    processor, model, device = load_model(MODEL_ID)
    print(f"Model loaded on {device}")

    dataset = load_or_create_dataset(DATASET_NAME, DATASET_DIR)

    caption_with_florence(dataset)

    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()