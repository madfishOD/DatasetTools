"""Transparent training guidance from declared intent and exported sample metadata."""
from collections import Counter
import json
from pathlib import Path
from training_export import atomic_write

RULES_VERSION = 1
CONTEXT_DEFAULTS = dict(training_goal='unspecified', goal_description='', training_family='unknown',
    training_method='unknown', trainer='', training_hardware='', trigger_word='', base_model='')
GOALS = {
    'unspecified': ['Choose a training goal before treating these notes as a recipe.'],
    'style': [
        'Use varied subjects and compositions so the target style is not tied to one subject.',
        'Describe changing content in captions; use a consistent style trigger if your workflow needs one.',
        'Evaluate the learned style on subjects and compositions absent from the training set.'],
    'character': [
        'Keep the identity label consistent; describe changing clothing, backgrounds, viewpoints and actions.',
        'Include varied views and framing. Decide which appearance traits should stay fixed and which should remain controllable.',
        'Evaluate identity on unseen scenes and outfits; a fixed background can become entangled with identity.'],
    'concept': [
        'Define what belongs to the concept and which incidental properties should remain variable.',
        'Use varied contexts and a consistent label; include examples that clarify the concept boundary.',
        'Evaluate the concept in unseen contexts, including combinations with already known concepts.'],
    'pose': [
        'Describe body arrangement, relative positions, viewpoint and interaction explicitly.',
        'Vary identities, clothing and backgrounds while covering the intended pose range.',
        'Check left/right conventions. Do not enable horizontal flips blindly when captions encode direction.',
        'A caption dataset does not provide pose conditioning; explicit pose control needs a compatible conditioning workflow.'],
    'custom': ['Use the stated goal to define what should stay fixed, what should vary, and how held-out results will be assessed.'],
}
SOURCES = [
    'https://huggingface.co/docs/diffusers/training/dreambooth',
    'https://huggingface.co/docs/diffusers/training/lora',
    'https://github.com/Nerogar/OneTrainer/wiki/Training',
]


def context(args):
    result = {key: getattr(args, key, default) for key, default in CONTEXT_DEFAULTS.items()}
    if result['training_goal'] not in GOALS: raise ValueError('Unsupported training goal')
    return result


def statistics(root, records):
    captions = [r.get('caption', '') for r in records]
    sizes = []
    for record in records:
        width, height = record.get('width'), record.get('height')
        if not width or not height:
            try:
                from PIL import Image
                with Image.open(root / 'images' / record['file']) as image:
                    width, height = image.size  # Header metadata only; no content evaluation.
            except (OSError, ValueError):
                continue
        sizes.append((width, height))
    return dict(samples=len(records), unique_image_hashes=len({r['image_sha256'] for r in records}),
        repeated_caption_count=sum(n-1 for n in Counter(captions).values()),
        caption_words_min=min((len(t.split()) for t in captions), default=0),
        caption_words_max=max((len(t.split()) for t in captions), default=0),
        known_dimensions=len(sizes), short_side_min=min((min(w,h) for w,h in sizes), default=None),
        aspect_ratio_min=min((max(w,h)/min(w,h) for w,h in sizes), default=None),
        aspect_ratio_max=max((max(w,h)/min(w,h) for w,h in sizes), default=None))


def render(root, records, args, plan=None):
    intent = context(args); stats = statistics(root, records)
    lines = ['TRAINING ADVICE', f'Rules version: {RULES_VERSION}',
        'Scope: only the approved image/caption versions in this export.',
        'No visual or semantic evaluation was performed. Guidance is a starting point, not an optimal recipe.', '',
        'TRAINING INTENT']
    lines += [f'{key}: {value or "Not specified"}' for key,value in intent.items()]
    lines += ['', 'DATASET FACTS'] + [f'{key}: {value if value is not None else "Unknown"}' for key,value in stats.items()]
    lines += ['', 'GOAL-SPECIFIC GUIDANCE'] + GOALS[intent['training_goal']]
    if intent['training_goal'] == 'custom':
        lines.append('Custom goals currently use general guidance; no automatic interpretation of the description is performed.')
    if intent['trigger_word']:
        count = sum(intent['trigger_word'] in r.get('caption', '') for r in records)
        lines.append(f'Declared trigger appears as a literal substring in {count}/{len(records)} captions. This is not a tokenizer check.')
        if count < len(records): lines.append('Review whether missing triggers are intentional before training; captions were not modified.')
    lines += ['', 'DATASET-DEPENDENT ACTIONS']
    if stats['samples'] < 20:
        lines.append('Small set (<20 samples): high repetition can overfit. Inspect early checkpoints and add variation where the goal needs it.')
    if stats['unique_image_hashes'] < stats['samples']:
        lines.append('Exact image duplicates exist. They increase exposure weight; remove unintended copies or account for weighting.')
    if stats['repeated_caption_count']:
        lines.append('Repeated caption text exists. Check whether it omits changing attributes; repetition alone does not prove an error.')
    if stats['short_side_min'] and stats['short_side_min'] < 512:
        lines.append('Some images have a short side below 512 px. Upscaling will not restore missing detail; avoid choosing resolution solely from model defaults.')
    if stats['aspect_ratio_max'] and stats['aspect_ratio_max'] > 1.5:
        lines.append('The export includes non-square images. Use supported aspect-ratio buckets and check cropping against the learning target.')
    if stats['caption_words_max'] > 75:
        lines.append('Some captions exceed 75 whitespace-separated words. Check truncation with the training model tokenizer; words are not tokens.')
    lines += ['Reserve held-out examples or prompts for evaluation. Keep exact duplicates in the same split.',
              'Masks are not training conditioning in this export. Training images and captions are unchanged.']
    lines += ['', 'TRAINING SETTINGS']
    if plan and plan.get('ready'):
        lines += ['The following values match the accompanying OneTrainer configuration. They are existing starting defaults, not goal-optimized values.']
        lines += [f'{key}: {plan[key]}' for key in ('family','base_model','resolution','learning_rate','lora_rank','epochs','estimated_optimizer_steps')]
        lines += ['Method: LoRA. Batch size: 1. Gradient accumulation: 1. Repeats: 1.', plan['heuristic']]
    else:
        lines += ['No trainer configuration accompanies this dataset. Exact learning rate, rank, steps and resolution are intentionally not inferred from image count.',
                  'Start with the recipe documented for your exact base model, training method and trainer version.']
        if intent['training_method'] == 'lora':
            lines.append('Tune LoRA capacity and learning rate separately; increasing rank is not automatically an improvement. Start with the text encoder frozen unless the selected recipe requires training it.')
        elif intent['training_method'] == 'full':
            lines.append('Full fine-tuning has different memory and learning-rate requirements from LoRA; do not reuse LoRA settings unchanged.')
        else:
            lines.append('Specify LoRA or full fine-tuning before choosing capacity or optimizer settings.')
    lines += ['Use a short pilot run; change one parameter at a time and compare checkpoints using a fixed evaluation setup.',
              'Actual training memory also depends on architecture, resolution, optimizer, precision and offloading. Captioning hardware availability is not a training fit estimate.',
              '', 'ASSUMPTIONS AND REFERENCES',
              'The goal and training context are user declarations. Subject diversity, caption accuracy and suitability remain unmeasured.',
              'Goal-specific and dataset thresholds above are project heuristics, not claims of validated training outcomes.'] + SOURCES
    return '\n'.join(lines) + '\n', {'rules_version': RULES_VERSION, 'context': intent, 'statistics': stats}


def write(root, staging, records, args):
    package = 'dataset' if args.no_training_config else 'onetrainer'
    plan = json.loads((staging / package / 'training_plan.json').read_text())
    text, metadata = render(root, records, args, plan)
    atomic_write(staging / 'TRAINING_ADVICE.txt', text.encode('utf-8'))
    return metadata
