# Training intent, advice and model selection

Open **Project settings…** after creating or opening a project. Settings are stored in `project.json` and survive Resume and project moves. The UI is in English.

## Training intent

The **Training intent** tab accepts a goal (style, character, concept, pose/interaction or custom), a description of fixed/variable attributes, an optional trigger, training model family and exact base model, method, trainer and training hardware. The training machine may differ from the captioning machine.

**Training advice…** previews guidance for approved, complete captions only. **Export approved** includes `TRAINING_ADVICE.txt` and structured context/statistics in `dataset_manifest.json`. Advice uses declared intent and technical facts: sample count, duplicate image hashes, repeated text, caption word lengths and image header dimensions. No visual or semantic analysis is performed. A custom description is recorded verbatim; this version does not automatically interpret it.

Changing the goal or context changes the export fingerprint and produces a new snapshot. Previous snapshots and captions remain intact. Advice has an explicit rules version. Changes to approved sample versions also invalidate the export fingerprint; repeated export without changes reuses the intact snapshot.

Advice contains goal-specific preparation/evaluation suggestions and conditional actions for small sets, duplicates, missing declared triggers, small dimensions, wide aspect ratios and long captions. These thresholds are project heuristics. It cannot establish semantic diversity, caption accuracy or optimal training settings.

Dataset-only advice deliberately avoids exact LR/rank/step recommendations without a validated recipe. For a OneTrainer export, numerical values are read from the generated training plan and match its configuration; they are existing starting defaults, not optimized for the declared goal. Full fine-tuning or a different trainer requires dataset-only export. Declared training family must agree with an explicitly requested OneTrainer configuration.

Example CLI:

```bash
./captioning_tool/auto_captioning_tool.command --resume projects/my-dataset \
  --export-only --no-training-config --training-goal character \
  --goal-description "Stable identity; variable outfits and backgrounds" \
  --trigger-word my_character --training-family sdxl --training-method lora \
  --trainer kohya --training-hardware "Remote NVIDIA GPU"
```

## Processing models

The **Processing models** tab provides independent choices:

| Stage | Variants |
| --- | --- |
| Caption | Qwen3-VL 2B Instruct, Qwen3-VL 8B Instruct, JoyCaption Beta One |
| Regions | Qwen3-VL 2B Instruct, Qwen3-VL 8B Instruct |
| Masks | SAM 2.1 Tiny, SAM 2.1 Large |

Each entry shows its repository, pinned revision, estimated memory budget and validation status. These are supported model variants, not arbitrary Hugging Face repositories or user-entered revisions. Larger Qwen captioning and MPS paths outside the tested Compact combination are experimental; model selection does not imply every device/model combination has been tested.

The catalog in `model_catalog.py` is shared by CLI, GUI and fingerprints. It can be extended for more powerful systems without redesigning the selector, provided a compatible loader and validation are added. The current Mac is not a hardware ceiling. All catalog variants remain selectable on small and large hosts. There is no silent substitution, quantization or CPU fallback.

A separate subprocess detects local CUDA/MPS/CPU availability and current memory. CUDA free VRAM and shared MPS system RAM are identified separately. Estimates (8/24 GiB for Qwen 2B/8B, 24 GiB for JoyCaption, 4/8 GiB for SAM Tiny/Large) are rough working budgets, not benchmarked minima. Available memory above a budget does not guarantee successful execution; precision, image size, tokens and other processes affect peak use. Stages run sequentially, so their budgets are not summed. Float32 can require more memory. Training capacity is a separate question.

CLI overrides:

```bash
./captioning_tool/auto_captioning_tool.command --resume projects/my-dataset \
  --caption-model qwen3-vl-8b --grounding-model qwen3-vl-2b --sam-model sam2.1-tiny
```

Explicit stage choices override Compact/Quality defaults and persist on resume. Supported IDs are listed by `--help`. Existing imported/edited/approved captions are preserved after a model change; **Regenerate selected** explicitly reruns a chosen sample. Missing or stale unapproved generated results use the new model. The actual model/revision is recorded per result. Model downloads happen only for stages with pending work.

## Verification

38 automated tests cover prior pipeline behavior plus goal differentiation, snapshot invalidation/reuse, export-only scope, configuration/advice consistency, unsupported training-config combinations, independent stage fingerprints, persisted model choice and selectable heavy variants on simulated 4/96 GiB hosts. No larger model weights are downloaded by these tests. A real MPS smoke explicitly overrode the Quality profile with Qwen3-VL 2B: 2/2 captions completed, 0 errors, and records contain the selected repository/revision. The accepted 40-pair dataset was exported with advice without inference; all snapshot hashes matched and repeat export reused the snapshot.

References used for general training context: [Diffusers DreamBooth](https://huggingface.co/docs/diffusers/training/dreambooth), [Diffusers LoRA](https://huggingface.co/docs/diffusers/training/lora). Model loading follows the existing Transformers paths and the [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). No training experiment has been run to validate a goal-specific numerical recipe.
