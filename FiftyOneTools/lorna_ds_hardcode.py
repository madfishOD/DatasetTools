# -*- coding: utf-8 -*-import os

os.environ["FIFTYONE_DATABASE_DIR"] = r"D:\FiftyOneDB"  # ROOT .fiftyone folderimport fiftyone as foimport fiftyone.types as fotimport fiftyone.zoo as fozimport fiftyone.brain as fob# --- Config ---dataset_dir = r"E:\Work\Lorna_daaset_extension"  # ensure this path is correctname = "lorna_ext"EMB_FIELD = "clip_emb"BRAIN_KEY = "img_sim"UNIQUE_TOP_N = 100# --- Load/create dataset ---if fo.dataset_exists(name):
	ds = fo.load_dataset(name)else:
	ds = fo.Dataset.from_dir(
		dataset_dir=dataset_dir,
		dataset_type=fot.ImageDirectory,
		name=name,
	)# Keep it across restartsif not ds.persistent:
	ds.persistent = True# --- Compute (or reuse) CLIP embeddings ---if EMB_FIELD not in ds.get_field_schema():
	print("Computing CLIP embeddings…")
	clip_model = foz.load_zoo_model("clip-vit-base32-torch")
	ds.compute_embeddings(clip_model, embeddings_field=EMB_FIELD)
	ds.save()else:
	print(f"Embeddings already present: {EMB_FIELD}")# --- Compute (or reuse) similarity brain run ---if BRAIN_KEY in ds.list_brain_runs():
    print(f"Reusing brain run: {BRAIN_KEY}")
    results = ds.load_brain_results(BRAIN_KEY)   # <- use the dataset method
else:
    print("Computing similarity (cosine on embeddings)…")
    results = fob.compute_similarity(
        ds,
        embeddings=EMB_FIELD,       # or model=...
        brain_key=BRAIN_KEY,
        metric="cosine",
    )# after you have EMB_FIELD on samples
UMAP_KEY = "umap_img"

if UMAP_KEY in ds.list_brain_runs():
    print(f"Reusing UMAP: {UMAP_KEY}")
else:
    print("Computing UMAP on clip_emb …")
    # need umap-learn installed once:  pip install umap-learn
    fob.compute_visualization(
        ds,
        embeddings=EMB_FIELD,      # your stored "clip_emb"
        brain_key=UMAP_KEY,
        method="umap",             # or "tsne"
        metric="cosine",
        n_neighbors=25,
        min_dist=0.05,
        overwrite=True,            # set True when you want to refresh
    )# --- Find most unique samples ---results.find_unique(UNIQUE_TOP_N)unique_view = ds.select(results.unique_ids)# Optionally tag them so you can filter laterfor s in unique_view:
	if "keep_core" not in s.tags:
		s.tags.append("keep_core")ds.save()# --- Launch App focused on the unique view ---session = fo.launch_app(ds, address="127.0.0.1")session.view = unique_viewprint("URL:", session.url)session.wait()