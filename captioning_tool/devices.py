"""Small shared device helpers; importing the CLI does not require PyTorch."""
import gc
import os


def resolve_device(requested='auto', precision='auto'):
    import torch
    device = requested
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            raise RuntimeError('No GPU available. Use --device cpu explicitly for slow CPU inference.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment.')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable. Use an arm64 Python and MPS-enabled PyTorch on a supported Mac.')
    if device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Unset PYTORCH_ENABLE_MPS_FALLBACK: this experiment requires explicit MPS execution without hidden CPU fallback.')
    if precision == 'auto':
        precision = 'float32' if device == 'cpu' else (
            'bfloat16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'float16')
    if device == 'cuda' and precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This CUDA GPU does not support bfloat16; use --dtype float16.')
    if device == 'mps' and precision == 'bfloat16':
        raise ValueError('bfloat16 is not validated for Mac Compact; use float16 or float32.')
    return device, precision


def synchronize(device):
    import torch
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def release_memory(device):
    import torch
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


def memory_snapshot(device):
    """Point-in-time values, deliberately not labelled as peak memory."""
    import torch
    import psutil
    values = {'process_rss_bytes': psutil.Process().memory_info().rss,
              'system_available_bytes': psutil.virtual_memory().available}
    if device == 'cuda':
        values['device_allocated_bytes'] = torch.cuda.memory_allocated()
        values['device_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    elif device == 'mps':
        values['device_allocated_bytes'] = torch.mps.current_allocated_memory()
        values['device_driver_bytes'] = torch.mps.driver_allocated_memory()
    return values
