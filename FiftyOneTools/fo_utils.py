from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Collect all image file paths from a directory and its subdirectories
def collect_image_paths(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def rename_files(
    paths: Iterable[Path],
    name: str = "",
    fmt: str = "000",
    start: int = 1,
    dry_run: bool = False,
) -> List[Tuple[Path, Path]]:
    """
    Rename files in-place with safe incrementing:
      - if name != "":  name_001.jpg, name_002.jpg, ...
      - if name == "":  001.jpg, 002.jpg, ...
    Width comes from len(fmt), e.g. "0000" -> 4.

    Returns a list of (old_path, new_path). If dry_run=True, does not rename.
    """
    width = max(1, len(fmt))
    plan: List[Tuple[Path, Path]] = []
    planned = set()  # (dirpath, lower-filename) to avoid in-batch clashes

    for p in sorted(map(Path, paths)):
        d = p.parent
        i = start
        while True:
            if name:
                newname = f"{name}_{i:0{width}d}{p.suffix}"
            else:
                newname = f"{i:0{width}d}{p.suffix}"
            target = d / newname
            key = (str(d), newname.lower())

            # accept if target is the same file, or doesn't exist on disk and not already planned
            if target == p or (not target.exists() and key not in planned):
                break
            i += 1

        plan.append((p, target))
        planned.add(key)

    if not dry_run:
        for src, dst in plan:
            if src != dst:
                src.rename(dst)

    return plan