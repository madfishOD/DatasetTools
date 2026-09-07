"""Versioned local projects, stage checkpoints and immutable export snapshots."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid

from training_export import atomic_write, save, sha, export_training

VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def local(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Project path escapes its root: {relative}')
    return path


@contextmanager
def project_lock(root):
    """OS locks release after a crash; the remaining lock file is harmless."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.project.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Project is already open by another process.') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_project(root):
    path = root / 'project.json'
    if not path.is_file():
        raise ValueError('No project.json. Legacy outputs cannot be resumed; import their images/TXT into a new project.')
    project = json.loads(path.read_text(encoding='utf-8'))
    if project.get('schema_version') != VERSION:
        raise ValueError('Unsupported project schema version; refusing to modify it.')
    if not isinstance(project.get('sources'), list):
        raise ValueError('Invalid project sources')
    seen = set()
    for row in project['sources']:
        for field in ('image', 'record'):
            local(root, row[field])
        if row['sample_id'] in seen:
            raise ValueError('Duplicate sample ID')
        seen.add(row['sample_id'])
    return project


def create_project(root):
    project = {'schema_version': VERSION, 'project_id': str(uuid.uuid4()),
               'sources': [], 'options': {}, 'prompts': {}, 'latest_export': None}
    for folder in ('images', 'captions', 'regions', 'errors', 'prompts', 'previews', 'grounding_raw'):
        (root / folder).mkdir(exist_ok=True)
    save(root / 'project.json', project)
    return project


def record_path(root, row):
    return local(root, row['record'])


def caption_path(root, record):
    return local(root, 'captions/' + Path(record['file']).stem + '.txt')


def remember_caption(root, record, raw):
    key = hashlib.sha256(raw).hexdigest()
    relative = f"caption_history/{record['sample_id']}/{key}.txt"
    path = local(root, relative)
    if not path.exists():
        atomic_write(path, raw)
    return relative


def import_sources(root, project, source, files):
    """Reserve stable IDs before copying, so interrupted imports can resume."""
    known = {row['source_file']: row for row in project['sources']}
    stems = {}
    for image in files:
        stems.setdefault((image.parent, image.stem.casefold()), []).append(image)
    additions = []
    for image in files:
        relative = image.relative_to(source).as_posix()
        checksum = sha(image)
        if relative in known:
            if known[relative]['sha256'] != checksum:
                raise ValueError(f'Source changed: {relative}. Create a new project for changed originals.')
            continue
        identifier = uuid.uuid5(uuid.UUID(project['project_id']), relative).hex
        filename = identifier + image.suffix.lower()
        txt = image.with_suffix('.txt')
        if txt.exists() and len(stems[(image.parent, image.stem.casefold())]) > 1:
            raise ValueError(f'Ambiguous TXT for multiple images: {txt}')
        caption = None
        if txt.is_file():
            raw = txt.read_bytes()
            raw.decode('utf-8-sig')  # Reject invalid encoding before mutating the project.
            if raw.strip():
                caption = {'source_file': txt.relative_to(source).as_posix(), 'sha256': sha(txt)}
        additions.append({'source_id': identifier, 'sample_id': identifier,
                          'source_file': relative, 'sha256': checksum,
                          'image': 'images/' + filename, 'record': 'regions/' + identifier + '.json',
                          'imported_caption': caption})
    project['sources'].extend(additions)
    project['source_root'] = str(source)
    save(root / 'project.json', project)
    finish_import(root, project, source)


def finish_import(root, project, source=None):
    for row in project['sources']:
        image = local(root, row['image'])
        if not image.exists():
            if source is None:
                raise ValueError('Incomplete import: provide --input pointing to the original folder.')
            original = local(source, row['source_file'])
            raw = original.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError(f"Source changed during import: {row['source_file']}")
            atomic_write(image, raw)
        if sha(image) != row['sha256']:
            raise ValueError(f"Project image modified: {row['image']}")
        path = record_path(root, row)
        if path.exists():
            continue
        record = {'file': image.name, 'source_file': row['source_file'],
                  'source_id': row['source_id'], 'sample_id': row['sample_id'],
                  'image_sha256': row['sha256'], 'caption': '', 'characters': [],
                  'qa_flags': [], 'stages': {}, 'review_status': 'unreviewed'}
        if row.get('imported_caption'):
            info = row['imported_caption']
            history = local(root, f"caption_history/{row['sample_id']}/{info['sha256']}.txt")
            if history.exists():
                raw = history.read_bytes()
            elif source is not None:
                raw = local(source, info['source_file']).read_bytes()
            else:
                raise ValueError('Incomplete TXT import: provide --input with the original folder.')
            if hashlib.sha256(raw).hexdigest() != info['sha256']:
                raise ValueError('Original TXT changed during import')
            record['original_caption_path'] = remember_caption(root, record, raw)
            atomic_write(caption_path(root, record), raw)
            record.update(caption=raw.decode('utf-8-sig'), caption_sha256=info['sha256'], caption_origin='imported')
            record['stages']['caption'] = {'status': 'complete', 'fingerprint': 'imported'}
        save(path, record)


def load_records(root, project):
    records = {}
    for row in project['sources']:
        image = local(root, row['image'])
        if not image.is_file() or sha(image) != row['sha256']:
            raise ValueError(f"Missing or modified project image: {row['image']}")
        record = json.loads(record_path(root, row).read_text(encoding='utf-8'))
        if record['sample_id'] != row['sample_id'] or record['file'] != image.name:
            raise ValueError('Record identity does not match manifest')
        records[image.name] = record
    return records


def sync_edits(root, records, write=True):
    """The editable TXT is the user-facing working copy, immutable text versions stay in history."""
    for record in records.values():
        path = caption_path(root, record)
        if not path.exists():
            if record.get('caption_sha256'):
                record['stages']['caption'] = {'status': 'missing'}
                record['review_status'] = 'unreviewed'
            continue
        raw = path.read_bytes()
        text = raw.decode('utf-8-sig')
        checksum = hashlib.sha256(raw).hexdigest()
        if checksum == record.get('caption_candidate_sha256') and record['stages'].get('caption', {}).get('status') != 'complete':
            # A crash may land between atomic TXT replacement and the completion checkpoint.
            # This is an uncommitted model result, not a newly imported human edit.
            if write:
                record['uncommitted_caption_path'] = remember_caption(root, record, raw)
                save(root / 'regions' / (Path(record['file']).stem + '.json'), record)
            continue
        if checksum != record.get('caption_sha256'):
            if write:
                remember_caption(root, record, raw)
            record.update(caption=text, caption_sha256=checksum, caption_origin='edited', review_status='unreviewed')
            record.pop('caption_candidate_sha256', None)
            record['stages']['caption'] = {'status': 'complete' if text.strip() else 'missing', 'fingerprint': 'edited'}
        if write:
            save(root / 'regions' / (Path(record['file']).stem + '.json'), record)


def fingerprints(record, args, prompts):
    from engine import model_specs
    repos, revisions = model_specs(args.profile)
    base = {'version': 1, 'image': record['image_sha256']}
    region = digest({**base, 'repo': repos['grounding'], 'revision': revisions['grounding'],
                     'prompt': prompts['regions'], 'tokens': args.region_tokens, 'dtype': args.dtype})
    return {'grounding': region,
            'sam': digest({**base, 'grounding': region, 'revision': revisions['sam'],
                           'boxes': [c.get('bbox_xyxy_pixels') for c in record.get('characters', [])]}),
            'caption': digest({**base, 'repo': repos['caption'], 'revision': revisions['caption'],
                               'prompt': prompts['instruction'], 'system': prompts['system'],
                               'tokens': args.caption_tokens, 'side': args.max_image_side, 'dtype': args.dtype})}


def artifact_valid(root, record, stage):
    state = record['stages'].get(stage, {})
    if state.get('status') != 'complete':
        return False
    if stage == 'caption':
        path = caption_path(root, record)
        return path.is_file() and bool(record.get('caption', '').strip()) and sha(path) == record.get('caption_sha256')
    for name, checksum in state.get('artifacts', {}).items():
        path = local(root, name)
        if not path.is_file() or sha(path) != checksum:
            return False
    return bool(record.get('grounding_complete' if stage == 'grounding' else 'segmentation_complete'))


def plan_work(root, records, args, prompts):
    work = {stage: [] for stage in ('grounding', 'sam', 'caption')}
    requested = set(args.select or [])
    matches = {r['sample_id'] for r in records.values()
               if requested.intersection((r['sample_id'], r['file'], r['source_file']))}
    valid_names = {name for r in records.values() for name in (r['sample_id'], r['file'], r['source_file'])}
    if requested - valid_names:
        raise ValueError('Unknown --select sample: ' + ', '.join(sorted(requested - valid_names)))
    if args.caption_policy == 'regenerate' and not requested:
        raise ValueError('--caption-policy regenerate requires explicit --select sample IDs or source filenames.')
    for record in records.values():
        if requested and record['sample_id'] not in matches:
            continue
        expected = fingerprints(record, args, prompts)
        image = root / 'images' / record['file']
        if args.stage != 'captions':
            needs_grounding = not artifact_valid(root, record, 'grounding') or record['stages']['grounding'].get('fingerprint') != expected['grounding']
            if needs_grounding:
                record['grounding_complete'] = False
                record['segmentation_complete'] = False
                record['stages']['sam'] = {'status': 'pending'}
                work['grounding'].append(image)
            if needs_grounding or not artifact_valid(root, record, 'sam') or record['stages']['sam'].get('fingerprint') != expected['sam']:
                work['sam'].append(image)
        if args.stage != 'regions':
            ready = artifact_valid(root, record, 'caption')
            protected = record.get('caption_origin') in ('edited', 'imported') or record.get('review_status') == 'approved'
            stale = record['stages'].get('caption', {}).get('fingerprint') != expected['caption']
            if args.caption_policy == 'regenerate' or (args.caption_policy == 'fill-missing' and (not ready or (stale and not protected))):
                work['caption'].append(image)
                record['review_status'] = 'unreviewed'
                record['stages']['caption'] = {'status': 'pending'}
    return work


def checkpoint(root, records, args, prompts, image, stage, status, error=None):
    record = records[image.name]
    if status == 'running':
        record['stages'][stage] = {'status': status, 'fingerprint': fingerprints(record, args, prompts)[stage]}
        if stage == 'caption':
            record['review_status'] = 'unreviewed'
            path = caption_path(root, record)
            if path.exists():
                remember_caption(root, record, path.read_bytes())
    else:
        state = record['stages'].setdefault(stage, {})
        state['status'] = status
        if error:
            state['error'] = str(error)
        if status == 'complete':
            state.pop('error', None)
            if stage == 'caption':
                raw = caption_path(root, record).read_bytes()
                record['generated_caption_path'] = remember_caption(root, record, raw)
                record['caption_origin'] = 'generated'
                record.pop('caption_candidate_sha256', None)
            names = []
            if stage == 'grounding':
                names = ['grounding_raw/' + image.stem + '.txt']
            elif stage == 'sam':
                names = [c['mask'] for c in record['characters'] if c.get('mask')]
            state['artifacts'] = {name: sha(local(root, name)) for name in names}
    save(root / 'regions' / (image.stem + '.json'), record)


def export_snapshot(root, project, records, args):
    included = [r for r in records.values() if r.get('review_status') == 'approved' and artifact_valid(root, r, 'caption')]
    if not included:
        project['latest_export'] = None
        save(root / 'project.json', project)
        return {'accepted_pairs': 0, 'reason': 'No approved complete captions; use --approve-all after review.'}
    settings = {key: getattr(args, key) for key in ('family', 'base_model', 'resolution', 'rank', 'learning_rate',
                                                  'epochs', 'target_steps', 'no_training_config')}
    key = digest({'samples': [(r['sample_id'], r['image_sha256'], r['caption_sha256']) for r in included],
                  'settings': settings, 'path': None if args.no_training_config else str(root.resolve()), 'version': 1})
    exports = root / 'exports'
    exports.mkdir(exist_ok=True)
    # Recover an export committed just before a crash that prevented updating the latest pointer.
    for path in sorted(exports.glob('*/snapshot.json')):
        if path.parent.name.startswith('.'):
            continue
        old = json.loads(path.read_text())
        if old.get('fingerprint') == key and old.get('status') == 'complete':
            if all(local(path.parent, name).is_file() and sha(local(path.parent, name)) == checksum
                   for name, checksum in old['artifacts'].items()):
                project['latest_export'] = path.parent.relative_to(root).as_posix()
                save(root / 'project.json', project)
                return {'accepted_pairs': len(included), 'path': project['latest_export'], 'reused': True}
    identifier = uuid.uuid4().hex
    staging, final = exports / ('.partial-' + identifier), exports / identifier
    staging.mkdir()
    files = []
    for record in included:
        image = root / 'images' / record['file']
        if sha(image) != record['image_sha256']:
            raise ValueError('Image changed before export')
        raw = caption_path(root, record).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['caption_sha256']:
            raise ValueError('Caption changed during export; retry after editing has finished')
        atomic_write(staging / 'captions' / (image.stem + '.txt'), raw)
        files.append(image)
    export_training(staging, files, args, path_root=final)
    for record, image in zip(included, files):
        if sha(image) != record['image_sha256']:
            raise ValueError('Image changed during export')
    save(staging / 'dataset_manifest.json', {'schema_version': VERSION,
         'samples': [{'source_id': r['source_id'], 'sample_id': r['sample_id'], 'source_file': r['source_file'],
                      'image': ('dataset' if args.no_training_config else 'onetrainer') + '/data/' + r['file'],
                      'caption': ('dataset' if args.no_training_config else 'onetrainer') + '/data/' + Path(r['file']).stem + '.txt',
                      'image_sha256': r['image_sha256'], 'caption_sha256': r['caption_sha256']} for r in included]})
    artifacts = {p.relative_to(staging).as_posix(): sha(p) for p in staging.rglob('*') if p.is_file()}
    save(staging / 'snapshot.json', {'status': 'complete', 'fingerprint': key, 'artifacts': artifacts})
    os.replace(staging, final)
    project['latest_export'] = final.relative_to(root).as_posix()
    save(root / 'project.json', project)
    return {'accepted_pairs': len(included), 'path': project['latest_export'], 'reused': False}


def edit_caption(root, sample_id, text, expected_sha256, approved=False):
    """Save one GUI edit under the CLI lock; reject stale editors instead of losing text."""
    with project_lock(root):
        project = load_project(root)
        records = load_records(root, project)
        record = next((r for r in records.values() if r['sample_id'] == sample_id), None)
        if record is None:
            raise ValueError('Sample no longer exists')
        path = caption_path(root, record)
        actual = sha(path) if path.exists() else None
        if actual != expected_sha256:
            raise ValueError('Caption changed outside this editor. Reload the project before saving.')
        if approved and not text.strip():
            raise ValueError('An empty caption cannot be approved')
        if actual is not None:
            remember_caption(root, record, path.read_bytes())
        # Preserve exact imported line endings/BOM for approval without editing.
        if not path.exists() or path.read_bytes().decode('utf-8-sig') != text:
            atomic_write(path, text.encode('utf-8'))
        sync_edits(root, records)
        if approved and not artifact_valid(root, record, 'caption'):
            raise ValueError('Caption has an unfinished model checkpoint; edit it or resume generation first.')
        record['review_status'] = 'approved' if approved else 'unreviewed'
        save(root / 'regions' / (Path(record['file']).stem + '.json'), record)
        project['latest_export'] = None  # Old snapshots remain immutable; rebuild the current selection on export.
        save(root / 'project.json', project)
        return record
