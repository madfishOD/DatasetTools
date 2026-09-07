"""Filesystem/crash-recovery tests. No model downloads, GPU or visual comparisons."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_captioning_tool as cli
import project as projects
from training_export import atomic_write, save, sha


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'input'; self.source.mkdir()
        self.root = self.base / 'project'
        self.args = cli.parser().parse_args(['--stage', 'captions', '--no-training-config'])
        self.prompts = {'instruction': 'caption', 'system': 'system', 'regions': 'regions'}

    def image(self, name, caption=None):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('image:' + name).encode())
        if caption is not None:
            path.with_suffix('.txt').write_bytes(caption)
        return path

    def imported(self):
        self.root.mkdir()
        project = projects.create_project(self.root)
        projects.import_sources(self.root, project, self.source, cli.discover(self.source))
        return project, projects.load_records(self.root, project)

    def fake_engine(self, out, files, records, args, prompts, work, checkpoint_fn):
        for image in work['caption']:
            checkpoint_fn(image, 'caption', 'running')
            text = b'generated caption'
            records[image.name]['caption_candidate_sha256'] = hashlib.sha256(text).hexdigest()
            checkpoint_fn(image, 'caption', 'writing')
            atomic_write(out / 'captions' / (image.stem + '.txt'), text)
            records[image.name].update(caption=text.decode(), caption_sha256=hashlib.sha256(text).hexdigest())
            checkpoint_fn(image, 'caption', 'complete')

    def run_cli(self, arguments, engine=None):
        with patch.object(cli, 'resolve_device', return_value=('cpu', 'float32')), \
             patch.object(cli, 'ensure_models', return_value=[]), \
             patch.object(cli, 'run_models', side_effect=engine or self.fake_engine) as runner:
            status = cli.main(arguments)
        return status, runner

    def test_import_preserves_bytes_and_stable_ids_when_adding_collisions(self):
        self.image('a.jpg', b'  Caption\r\n')
        project, records = self.imported()
        before = project['sources'][0].copy()
        record = next(iter(records.values()))
        self.assertEqual(projects.caption_path(self.root, record).read_bytes(), b'  Caption\r\n')
        self.assertEqual(projects.local(self.root, record['original_caption_path']).read_bytes(), b'  Caption\r\n')
        self.image('nested/a.jpg')
        projects.import_sources(self.root, project, self.source, cli.discover(self.source))
        self.assertEqual(project['sources'][0], before)
        self.assertEqual(len({s['sample_id'] for s in project['sources']}), 2)

    def test_ambiguous_sidecar_and_changed_source_are_rejected(self):
        self.image('a.jpg', b'Caption'); self.image('a.png')
        self.root.mkdir(); project = projects.create_project(self.root)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            projects.import_sources(self.root, project, self.source, cli.discover(self.source))
        self.assertFalse(project['sources'])
        (self.source / 'a.png').unlink()
        projects.import_sources(self.root, project, self.source, cli.discover(self.source))
        (self.source / 'a.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            projects.import_sources(self.root, project, self.source, cli.discover(self.source))

    def test_interrupted_import_keeps_reserved_ids_and_recovers(self):
        self.image('a.jpg', b'Original'); self.image('b.jpg')
        self.root.mkdir(); project = projects.create_project(self.root)
        calls = 0
        def failing_write(path, data):
            nonlocal calls
            if Path(path).parent.name == 'images':
                calls += 1
                if calls == 2:
                    raise OSError('disk failure')
            atomic_write(path, data)
        with patch.object(projects, 'atomic_write', side_effect=failing_write):
            with self.assertRaises(OSError):
                projects.import_sources(self.root, project, self.source, cli.discover(self.source))
        loaded = projects.load_project(self.root)
        ids = [r['sample_id'] for r in loaded['sources']]
        projects.finish_import(self.root, loaded, self.source)
        self.assertEqual(len(projects.load_records(self.root, loaded)), 2)
        self.assertEqual([r['sample_id'] for r in loaded['sources']], ids)

    def test_resume_only_runs_unfinished_files_after_interrupt_and_move(self):
        self.image('a.jpg'); self.image('b.jpg')
        def interrupted(out, files, records, args, prompts, work, checkpoint_fn):
            first = work['caption'][0]
            self.fake_engine(out, files, records, args, prompts, {'caption':[first]}, checkpoint_fn)
            checkpoint_fn(work['caption'][1], 'caption', 'running')
            raise KeyboardInterrupt()
        code, _ = self.run_cli(['--input',str(self.source),'--output',str(self.root),'--stage','captions','--no-training-config'], interrupted)
        self.assertEqual(code, 130)
        completed = list((self.root / 'captions').glob('*.txt'))[0]
        before = completed.stat().st_mtime_ns
        name = completed.name
        moved = self.base / 'moved'; self.root.rename(moved); self.root = moved
        shutil.rmtree(self.source)
        code, runner = self.run_cli(['--resume',str(moved)])
        self.assertEqual(code, 0)
        self.assertEqual(len(runner.call_args.kwargs['work']['caption']), 1)
        self.assertEqual((moved/'captions'/name).stat().st_mtime_ns, before)
        # Completed resume requires neither PyTorch nor models.
        with patch.object(cli,'resolve_device',side_effect=AssertionError('GPU accessed')), \
             patch.object(cli,'ensure_models',side_effect=AssertionError('Models accessed')), \
             patch.object(cli,'run_models',side_effect=AssertionError('Inference repeated')):
            self.assertEqual(cli.main(['--resume',str(moved)]), 0)

    def test_uncommitted_generated_txt_is_not_mistaken_for_manual_edit(self):
        self.image('a.jpg')
        project, records = self.imported(); record = next(iter(records.values()))
        image = self.root / 'images' / record['file']
        projects.checkpoint(self.root,records,self.args,self.prompts,image,'caption','running')
        raw = b'uncommitted output'
        record['caption_candidate_sha256'] = hashlib.sha256(raw).hexdigest()
        projects.checkpoint(self.root,records,self.args,self.prompts,image,'caption','writing')
        atomic_write(projects.caption_path(self.root,record),raw)
        records = projects.load_records(self.root,project)
        projects.sync_edits(self.root,records)
        self.assertEqual(len(projects.plan_work(self.root,records,self.args,self.prompts)['caption']),1)
        record = next(iter(records.values()))
        self.assertNotEqual(record.get('caption_origin'),'edited')
        atomic_write(projects.caption_path(self.root,record),b'Human correction')
        projects.sync_edits(self.root,records)
        self.assertEqual(record['caption_origin'],'edited')
        self.assertFalse(projects.plan_work(self.root,records,self.args,self.prompts)['caption'])

    def test_manual_edits_preserved_and_snapshots_immutable(self):
        self.image('a.jpg',b'Original\r\n')
        project,records = self.imported(); record = next(iter(records.values()))
        record['review_status'] = 'approved'
        first = projects.export_snapshot(self.root,project,records,self.args)
        old = self.root / first['path'] / 'dataset/data' / (Path(record['file']).stem+'.txt')
        self.assertEqual(old.read_bytes(),b'Original\r\n')
        self.assertTrue(projects.export_snapshot(self.root,project,records,self.args)['reused'])
        atomic_write(projects.caption_path(self.root,record),b'Edited\n')
        projects.sync_edits(self.root,records)
        self.assertEqual(record['review_status'],'unreviewed')
        self.assertFalse(projects.plan_work(self.root,records,self.args,{**self.prompts,'instruction':'new'})['caption'])
        self.assertEqual(projects.export_snapshot(self.root,project,records,self.args)['accepted_pairs'],0)
        record['review_status'] = 'approved'
        second = projects.export_snapshot(self.root,project,records,self.args)
        self.assertNotEqual(first['path'],second['path'])
        self.assertEqual(old.read_bytes(),b'Original\r\n')
        self.assertEqual(projects.local(self.root,record['original_caption_path']).read_bytes(),b'Original\r\n')

    def test_failed_export_and_commit_before_pointer_are_recoverable(self):
        self.image('a.jpg',b'Caption')
        project,records = self.imported()
        next(iter(records.values()))['review_status'] = 'approved'
        with patch.object(projects,'export_training',side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):projects.export_snapshot(self.root,project,records,self.args)
        self.assertFalse(list((self.root/'exports').glob('*/snapshot.json')))
        def fail_pointer(path, data):
            if Path(path).name == 'project.json':raise OSError('crash before latest pointer')
            save(path,data)
        with patch.object(projects,'save',side_effect=fail_pointer):
            with self.assertRaises(OSError):projects.export_snapshot(self.root,project,records,self.args)
        project = projects.load_project(self.root)
        result = projects.export_snapshot(self.root,project,records,self.args)
        self.assertTrue(result['reused'])

    def test_atomic_file_failure_preserves_previous_json(self):
        target = self.base/'state.json';save(target,{'complete':True})
        with patch('training_export.os.replace',side_effect=OSError('replace failure')):
            with self.assertRaises(OSError):save(target,{'complete':False})
        self.assertEqual(json.loads(target.read_text()),{'complete':True})

    def test_stage_fingerprints_and_missing_mask_only_rerun_sam(self):
        self.image('a.jpg',b'Imported')
        project,records = self.imported();record=next(iter(records.values()))
        image=self.root/'images'/record['file'];self.args.stage='regions'
        record['grounding_complete']=True
        atomic_write(self.root/'grounding_raw'/(image.stem+'.txt'),b'[]')
        projects.checkpoint(self.root,records,self.args,self.prompts,image,'grounding','running')
        projects.checkpoint(self.root,records,self.args,self.prompts,image,'grounding','complete')
        mask='masks/'+image.stem+'/person_1.png';atomic_write(self.root/mask,b'mask')
        record.update(characters=[{'mask':mask}],segmentation_complete=True)
        projects.checkpoint(self.root,records,self.args,self.prompts,image,'sam','running')
        projects.checkpoint(self.root,records,self.args,self.prompts,image,'sam','complete')
        self.assertFalse(any(projects.plan_work(self.root,records,self.args,self.prompts).values()))
        (self.root/mask).unlink()
        work=projects.plan_work(self.root,records,self.args,self.prompts)
        self.assertFalse(work['grounding']);self.assertEqual(work['sam'],[image])
        work=projects.plan_work(self.root,records,self.args,{**self.prompts,'regions':'changed'})
        self.assertEqual(work['grounding'],[image]);self.assertEqual(work['sam'],[image])

    def test_regenerate_requires_explicit_selection(self):
        self.image('a.jpg',b'Original');_,records=self.imported()
        self.args.caption_policy='regenerate'
        with self.assertRaisesRegex(ValueError,'requires explicit'):projects.plan_work(self.root,records,self.args,self.prompts)
        self.args.select=['a.jpg']
        self.assertEqual(len(projects.plan_work(self.root,records,self.args,self.prompts)['caption']),1)

    def test_prompt_change_only_requeues_unapproved_generated_caption(self):
        self.image('a.jpg',b'Imported');self.image('b.jpg');self.image('c.jpg')
        project,records=self.imported()
        for record in records.values():
            if record['source_file']=='a.jpg':continue
            image=self.root/'images'/record['file']
            self.fake_engine(self.root,[image],records,self.args,self.prompts,{'caption':[image]},
                             lambda image,stage,status,error=None: projects.checkpoint(self.root,records,self.args,self.prompts,image,stage,status,error))
            if record['source_file']=='c.jpg':record['review_status']='approved'
        work=projects.plan_work(self.root,records,self.args,{**self.prompts,'instruction':'changed'})
        self.assertEqual([records[p.name]['source_file'] for p in work['caption']],['b.jpg'])

    def test_corrupted_snapshot_is_not_reused(self):
        self.image('a.jpg',b'Caption');project,records=self.imported()
        record=next(iter(records.values()));record['review_status']='approved'
        first=projects.export_snapshot(self.root,project,records,self.args)
        path=self.root/first['path']/'dataset/data'/record['file']
        path.write_bytes(b'corrupted export')
        second=projects.export_snapshot(self.root,project,records,self.args)
        self.assertFalse(second['reused']);self.assertNotEqual(first['path'],second['path'])

    def test_import_export_only_and_check_do_not_touch_gpu(self):
        self.image('a.jpg',b'Caption')
        with patch.object(cli,'resolve_device',side_effect=AssertionError('GPU accessed')):
            self.assertEqual(cli.main(['--input',str(self.source),'--output',str(self.root),
                                       '--import-only','--no-training-config','--stage','captions','--approve-all']),0)
            self.assertEqual(cli.main(['--resume',str(self.root),'--export-only']),0)
            before={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
            self.assertEqual(cli.main(['--resume',str(self.root),'--check']),0)
            after={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_concurrent_lock_and_future_schema_rejected(self):
        with projects.project_lock(self.root):
            with self.assertRaises(RuntimeError):
                with projects.project_lock(self.root):pass
        save(self.root/'project.json',{'schema_version':999,'sources':[]})
        with self.assertRaisesRegex(ValueError,'Unsupported'):projects.load_project(self.root)

    def test_trainer_snapshot_paths_point_to_final_directory(self):
        self.image('a.jpg',b'Caption');project,records=self.imported()
        next(iter(records.values()))['review_status']='approved';self.args.no_training_config=False
        result=projects.export_snapshot(self.root,project,records,self.args)
        root=self.root/result['path']/'onetrainer'
        config=json.loads((root/'train.json').read_text())
        self.assertEqual(config['concept_file_name'],str(root/'concepts.json'))
        concept=json.loads((root/'concepts.json').read_text())[0]
        self.assertEqual(concept['path'],str(root/'data'))


if __name__ == '__main__':
    unittest.main()
