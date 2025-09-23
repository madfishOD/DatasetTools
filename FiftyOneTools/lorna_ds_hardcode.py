IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DB_DIR = r"D:\FiftyOneDB"
DATASET_DIR = r"E:\Work\Lorna_daaset_extension\Lorna_ext_filtered"
DATASET_NAME = "lorna_ext_filtered"
MODEL_ID = "microsoft/Florence-2-large"  # or "microsoft/Florence-2-large"
TASK_TOKEN = "<DETAILED_CAPTION>"   # Florence-2 task for captions
BATCH_SIZE = 1                      # 1 is safest; you can try >1 on big GPUs
MAX_NEW_TOKENS = 96                 # caption length budget
CAPTION_FIELD = "florence2_caption"
UNIQUENESS_FIELD = "uniqueness"   # where we store/read uniqueness


# -------------- Zero Shot Labels ------------
CAPTION_FIELD = "florence2_caption"# your text field
LABELS_FIELD  = "auto_labels"      # new multi-label field to create
TAG_FIELD     = "sample tags"      # optional: also push high-confidence labels to sample tags

EMB_FIELD = "clip_emb"     # where to store vectors
UMAP_KEY  = "umap_all"     # brain key to find in App

# Similarity / near-dup defaults
SIM_BRAIN_KEY_DEFAULT = "clip_sim"
SIM_MODEL_DEFAULT = "clip-vit-base32-torch"

# Provide the label set you want to detect
CANDIDATE_LABELS = [
    "portrait", "full body", "close-up",
    "horns", "antlers",
    "blue hair", "purple hair", "long hair", "short hair",
    "cartoon", "anime", "sketch", "line art", "render",
    "one girl", "two girls", "group",
    "outdoor", "city", "night", "indoor",
]

CONF_THRESH = 0.4  # tweak for your dataset

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
import fiftyone.brain as fob
import fiftyone.zoo as foz
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import pipeline
from fiftyone.core.labels import Classifications, Classification

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
        ds = create_dataset_from_dir(root)
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

# -------------- Zero Shot Labels ------------
def ensure_labels_field(ds: fo.Dataset, field_name: str = LABELS_FIELD) -> str:
    """
    Ensures there is a field to store multi-label results.
    Prefers `fo.Classifications`. If that fails (older API), creates a
    List[String] field instead. Returns the effective field "kind":
    either "classifications" or "string_list".
    """
    schema = ds.get_field_schema()
    if field_name in schema:
        f = schema[field_name]
        # Detect existing type
        if f.__class__.__name__ == "EmbeddedDocumentField" and getattr(f, "document_type", None).__name__ == "Classifications":
            return "classifications"
        if f.__class__.__name__ == "ListField" and getattr(f, "field", None).__class__.__name__ == "StringField":
            return "string_list"
        # If it exists but is a different type, raise to avoid corrupting data
        raise ValueError(f"Field '{field_name}' already exists but is not a Classifications or List[String] field")

    # Try preferred: Classifications
    try:
        ds.add_sample_field(field_name, fo.Classifications)
        ds.save()
        return "classifications"
    except Exception:
        # Fallback: List[String]
        ds.add_sample_field(field_name, fo.ListField, subfield=fo.StringField)
        ds.save()
        return "string_list"


# ---------- zero-shot model ----------
def build_zero_shot():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )

def labels_from_caption(clf, caption: str, candidate_labels, thresh=CONF_THRESH):
    if not caption or not caption.strip():
        return []
    out = clf(caption, candidate_labels, multi_label=True)
    # keep (label, score) pairs over threshold
    return [(lab, float(score)) for lab, score in zip(out["labels"], out["scores"]) if score >= thresh]

# ---------- writers that handle both schemas ----------
def write_labels_to_sample(sample: fo.Sample, lab_scores, field_kind: str):
    """
    Writes to LABELS_FIELD in either of the supported schemas.
    Also mirrors high-confidence labels into sample.tags if TAG_FIELD is set.
    """
    if not lab_scores:
        return

    if field_kind == "classifications":
        # Multi-label with confidences
        from fiftyone.core.labels import Classifications, Classification
        sample[LABELS_FIELD] = Classifications(
            classifications=[Classification(label=l, confidence=s) for l, s in lab_scores]
        )
    else:
        # Simple fallback: just store the strings
        labels = [l for l, _ in lab_scores]
        sample[LABELS_FIELD] = labels

    # Optional: also add as sample tags (keeps existing tags)
    if TAG_FIELD:
        for l, _ in lab_scores:
            if l not in sample.tags:
                sample.tags.append(l)

    sample.save()


# ---------- main entry to label from captions ----------
def tag_dataset_from_captions(ds: fo.Dataset):
    # 1) Make sure the destination field exists and learn its schema
    field_kind = ensure_labels_field(ds, LABELS_FIELD)

    # 2) Only samples that have captions and don't yet have labels
    view = ds.match(
        (fo.ViewField(CAPTION_FIELD).exists()) &
        (~fo.ViewField(LABELS_FIELD).exists())
    )

    clf = build_zero_shot()
    n = 0
    for s in view.iter_samples(progress=True):
        # access the caption without .get()
        try:
            caption = s[CAPTION_FIELD]     # preferred; fast and explicit
        except KeyError:
            caption = getattr(s, CAPTION_FIELD, "")

        labs = labels_from_caption(clf, caption, CANDIDATE_LABELS, CONF_THRESH)
        if labs:
            write_labels_to_sample(s, labs, field_kind)
            n += 1

    print(f"[label] Labeled {n} samples into '{LABELS_FIELD}' (mode: {field_kind}).")

def labels_from_captions(dataset: fo.Dataset):
    run = ask_yn("Generate labels from captions (Zero Shot)?", default=True)
    if run:
        tag_dataset_from_captions(dataset)

# -------------- Embeddings & UMAP ------------
def ensure_media(ds: fo.Dataset) -> None:
    # guarantees media_type is populated so we can filter to images
    if "media_type" not in ds.get_field_schema():
        ds.compute_metadata()

def ensure_embeddings(ds: fo.Dataset, force: bool = False) -> None:
    """
    Computes CLIP embeddings into EMB_FIELD for image samples (skips others).
    Reuses existing vectors unless force=True.
    """
    ensure_media(ds)

    if (not force) and (EMB_FIELD in ds.get_field_schema()):
        print(f"[emb] Reusing embeddings: {EMB_FIELD}")
        return

    print("[emb] Computing CLIP embeddings...")
    model = foz.load_zoo_model("clip-vit-base32-torch")  # CPU or GPU
    img_view = ds.match(F("media_type") == "image")
    if len(img_view) == 0:
        print("[emb] No images to embed.")
        return

    img_view.compute_embeddings(model, embeddings_field=EMB_FIELD)
    ds.save()
    print("[emb] Done.")

def ensure_umap(ds: fo.Dataset, force: bool = False) -> None:
    """
    Computes (or reuses) a 2D visualization from embeddings via UMAP.
    """
    # If we already have this brain run and not forcing, reuse it
    if (UMAP_KEY in ds.list_brain_runs()) and (not force):
        print(f"[umap] Reusing visualization: {UMAP_KEY}")
        return

    print("[umap] Computing UMAP (install 'umap-learn' if prompted)...")
    fob.compute_visualization(
        ds,
        embeddings=EMB_FIELD,
        brain_key=UMAP_KEY,
        method="umap",
        metric="cosine",
        n_neighbors=25,
        min_dist=0.05,
        overwrite=True,
    )
    print("[umap] Done.")

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

# -------------- Main ------------
def main():

    # Load the model once
    processor, model, device = load_model(MODEL_ID)
    print(f"Model loaded on {device}")

    dataset = load_or_create_dataset(DATASET_NAME, DATASET_DIR)

    caption_with_florence(dataset)
    labels_from_captions(dataset)

    # 2) Ensure CLIP embeddings (once)
    ensure_embeddings(dataset, force=False)

    # 3) Build (or reuse) similarity index
    ensure_similarity_index(
        dataset,
        brain_key=SIM_BRAIN_KEY_DEFAULT,
        model_name=SIM_MODEL_DEFAULT,
    )

    # 4) Tag near-duplicate clusters
    if ask_yn("Tag near-duplicates (seed='keep_core', dup='near_dup')?", default=True):
        seeds, dups = tag_near_duplicates(
            dataset,
            brain_key=SIM_BRAIN_KEY_DEFAULT,
            model_name=SIM_MODEL_DEFAULT,
            k=6,                # neighbors to inspect per seed
            max_dist=0.20,      # tighter -> fewer dups
            seed_tag="keep_core",
            dup_tag="near_dup",
            clear_previous=True,
        )
        print(f"[sim] Groups tagged — seeds: {seeds}, near_dups: {dups}")

        # Optional: quick prune prompt
        if dups and ask_yn(f"Prune {dups} samples tagged 'near_dup' (DB only, not files)?", default=False):
            n = prune_near_dups(dataset, dup_tag="near_dup")
            print(f"[sim] Pruned {n} near-dup samples")

    # 5) (Optional) Density score for “crowdedness” coloring in Embeddings
    if ask_yn("Compute similarity density field 'sim_density' for coloring in UMAP?", default=False):
        n = compute_sim_density(
            dataset,
            brain_key=SIM_BRAIN_KEY_DEFAULT,
            model_name=SIM_MODEL_DEFAULT,
            k=12,
            max_dist=None,          # or set a cap like 0.25
            out_field="sim_density"
        )
        print(f"[sim] Wrote density scores for {n} samples -> field 'sim_density'")

    # 6) (Optional) UMAP so you can Curate ▸ Embeddings immediately
    if ask_yn("Compute/reuse UMAP visualization?", default=True):
        ensure_umap(dataset, force=False)

    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()