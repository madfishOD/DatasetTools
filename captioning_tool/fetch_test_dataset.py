"""Fetch the small MIT-licensed Beans validation split and make a 12-image smoke set.

Run with the captioning runtime. Downloads are pinned and checksum-verified.
"""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import httpx

ROOT = Path(__file__).resolve().parent.parent
REVISION = '27aa014ce09b193e1a6f58112d4a66e0eddb69c5'
SHA256 = '90b7aa1c26d91d9afff07a30bbc67a5ea34f1f1397f068d8675be09d7d0c602d'
URL = f'https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/{REVISION}/data/validation.zip'


def main():
    cache = ROOT / 'cache/test-datasets'
    destination = ROOT / 'projects/test-datasets/beans'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'beans-validation.zip'
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        partial = archive.with_suffix('.partial')
        with httpx.stream('GET', URL, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open('wb') as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('Beans archive checksum mismatch')
        partial.replace(archive)
    # Only extract expected image members; never follow archive paths outside the dataset.
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unsafe archive path: {relative}')
            if relative.suffix.lower() != '.jpg':
                continue
            target = destination / relative
            data = source.read(member)
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Refusing to overwrite modified dataset image: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    images = sorted((destination / 'validation').rglob('*.jpg'))
    if len(images) != 133:
        raise RuntimeError(f'Expected 133 validation images, found {len(images)}')
    smoke = destination / 'smoke12'
    smoke.mkdir(exist_ok=True)
    selected = []
    for group in sorted((destination / 'validation').iterdir()):
        if group.is_dir():
            for image in sorted(group.glob('*.jpg'))[:4]:
                target = smoke / image.name
                if target.exists() and target.read_bytes() != image.read_bytes():
                    raise ValueError(f'Refusing to overwrite modified smoke image: {target}')
                shutil.copy2(image, target)
                selected.append(image.relative_to(destination).as_posix())
    response = httpx.get('https://raw.githubusercontent.com/AI-Lab-Makerere/ibean/master/LICENSE',
                         follow_redirects=True, timeout=30)
    response.raise_for_status()
    (destination / 'LICENSE.txt').write_bytes(response.content)
    (destination / 'SOURCE.json').write_text(json.dumps({
        'dataset': 'AI-Lab-Makerere/beans', 'revision': REVISION,
        'source': 'https://github.com/AI-Lab-Makerere/ibean', 'url': URL,
        'archive_sha256': SHA256, 'license': 'MIT', 'split': 'validation',
        'count': len(images), 'image_bytes': sum(p.stat().st_size for p in images),
        'smoke_selection': selected,
    }, indent=2), encoding='utf-8')
    print(f'Beans: {len(images)} images; smoke: {len(selected)} images; {destination}')


if __name__ == '__main__':
    main()
