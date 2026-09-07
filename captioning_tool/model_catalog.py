"""Explicit supported model variants; memory budgets are estimates, not fit guarantees."""
CATALOG = {
    'qwen3-vl-2b': dict(label='Qwen3-VL 2B Instruct', repo='Qwen/Qwen3-VL-2B-Instruct',
        revision='89644892e4d85e24eaac8bacfd4f463576704203', backend='qwen', stages=('caption', 'grounding'),
        budget_gib=8, note='Captioning tested on MPS; regions experimental.'),
    'qwen3-vl-8b': dict(label='Qwen3-VL 8B Instruct', repo='Qwen/Qwen3-VL-8B-Instruct',
        revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', backend='qwen', stages=('caption', 'grounding'),
        budget_gib=24, note='Grounding tested on CUDA; captioning and MPS experimental.'),
    'joycaption-beta-one': dict(label='JoyCaption Beta One', repo='fancyfeast/llama-joycaption-beta-one-hf-llava',
        revision='ebf414ea497a020da0f82df3913e5b6cb8e9663a', backend='llava', stages=('caption',),
        budget_gib=24, note='Captioning tested on CUDA; MPS experimental.'),
    'sam2.1-tiny': dict(label='SAM 2.1 Tiny', repo='facebook/sam2.1-hiera-tiny',
        revision='de431c4043854a71d8101e17995dfe596bf101a5', backend='sam', stages=('sam',),
        budget_gib=4, note='Limited MPS smoke passed.'),
    'sam2.1-large': dict(label='SAM 2.1 Large', repo='facebook/sam2.1-hiera-large',
        revision='665f8e2ad61cf5f53d65644ff27c8ee525124610', backend='sam', stages=('sam',),
        budget_gib=8, note='Tested on CUDA; MPS experimental.'),
}
DEFAULTS = {'compact': dict(caption='qwen3-vl-2b', grounding='qwen3-vl-2b', sam='sam2.1-tiny'),
            'quality': dict(caption='joycaption-beta-one', grounding='qwen3-vl-8b', sam='sam2.1-large')}


def selections(args):
    return {stage: getattr(args, stage + '_model', None) for stage in ('caption', 'grounding', 'sam')}


def selected(profile, overrides=None):
    result = DEFAULTS[profile].copy()
    for stage, name in (overrides or {}).items():
        if name:
            if name not in CATALOG or stage not in CATALOG[name]['stages']:
                raise ValueError(f'Unsupported {stage} model: {name}')
            result[stage] = name
    return result


def hardware():
    import psutil
    import torch
    ram = psutil.virtual_memory()
    result = {'ram_total_gib': round(ram.total / 2**30, 1), 'ram_available_gib': round(ram.available / 2**30, 1)}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result.update(device='cuda', name=torch.cuda.get_device_name(), available_gib=round(free / 2**30, 1), total_gib=round(total / 2**30, 1))
    elif torch.backends.mps.is_available():
        result.update(device='mps', name='Apple GPU (shared system memory)', available_gib=result['ram_available_gib'], total_gib=result['ram_total_gib'])
    else:
        result.update(device='cpu', name='CPU (slow inference)', available_gib=result['ram_available_gib'], total_gib=result['ram_total_gib'])
    return result


if __name__ == '__main__':
    import json
    print(json.dumps(hardware()), flush=True)
