"""Download COCO128 and select 40 diverse photos for local smoke testing.

Run with the captioning runtime. YOLO labels are kept separate from captions.
"""
import hashlib
import json
from pathlib import Path
import zipfile

import httpx

ROOT = Path(__file__).resolve().parent.parent
URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip'
SHA256 = '61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e'


def write_unchanged(path, content):
    if path.is_symlink():
        raise ValueError(f'Refusing to write through symlink: {path}')
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f'Refusing to overwrite modified file: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main():
    archive = ROOT / 'cache/test-datasets/coco128.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        partial = archive.with_suffix('.partial')
        with httpx.stream('GET', URL, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open('wb') as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('COCO128 archive checksum mismatch')
        partial.replace(archive)
    base = ROOT / 'projects/test-datasets'
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'coco128':
                raise ValueError(f'Unexpected archive path: {relative}')
            target = base / relative
            if not target.resolve().is_relative_to(base.resolve()):
                raise ValueError(f'Archive path escapes destination: {target}')
            if not member.is_dir():
                write_unchanged(target, source.read(member))
    root = base / 'coco128'
    images = sorted((root / 'images/train2017').glob('*.jpg'))
    if len(images) != 128:
        raise RuntimeError(f'Expected 128 images, found {len(images)}')
    classes, missing = {}, []
    for image in images:
        label = root / 'labels/train2017' / (image.stem + '.txt')
        if not label.exists():
            missing.append(image.name)
        classes[image] = {int(line.split()[0]) for line in label.read_text().splitlines()
                          if line.strip()} if label.exists() else set()
    selected, covered, remaining = [], set(), images[:]
    while len(selected) < 40:
        image = max(remaining, key=lambda p: (len(classes[p] - covered), len(classes[p]), -int(p.stem)))
        selected.append(image)
        covered.update(classes[image])
        remaining.remove(image)
    for image in selected:
        write_unchanged(root / 'smoke40' / image.name, image.read_bytes())
    people = ['000000000049.jpg', '000000000077.jpg', '000000000110.jpg',
              '000000000165.jpg', '000000000241.jpg', '000000000328.jpg']
    for name in people:
        write_unchanged(root / 'people6' / name, (root / 'images/train2017' / name).read_bytes())
    (root / 'SOURCE.json').write_text(json.dumps({
        'source': 'https://docs.ultralytics.com/datasets/detect/coco128/',
        'download_url': URL, 'archive_sha256': SHA256,
        'archive_bytes': archive.stat().st_size, 'image_count': len(images),
        'sample_count': len(selected), 'sample_class_ids': sorted(covered),
        'selection': 'greedy unseen class coverage, then class count, then smallest numeric image ID',
        'sample_files': [p.name for p in selected], 'missing_label_files': missing,
        'people6_files': people,
        'people6_selection': 'manual: riders, skateboarders, dining, standing and seated people; visual smoke only',
        'terms': 'https://cocodataset.org/#termsofuse',
        'note': 'Detection labels are not captions. Original archive notices retained; '
                'photos retain original licenses. This is not a held-out benchmark.',
    }, indent=2), encoding='utf-8')
    print(f'COCO128: {len(images)} images, {len(selected)} selected, '
          f'{len(covered)} annotated categories; {root}')


if __name__ == '__main__':
    main()
