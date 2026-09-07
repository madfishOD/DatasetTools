"""Advice snapshots and per-stage model selection, without semantic image review."""
import contextlib
import io
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import auto_captioning_tool as cli
import engine
import project as projects
import training_advice as advice
from model_catalog import CATALOG, selections, selected
from training_export import sha


class AdviceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name); self.source = self.base/'input'; self.source.mkdir()
        self.root = self.base/'project'
        for name,text in [('a','trigger portrait'),('b','second caption')]:
            (self.source/f'{name}.jpg').write_bytes(b'same-image-bytes')
            (self.source/f'{name}.txt').write_text(text)
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(['--input',str(self.source),'--output',str(self.root),'--import-only',
                      '--stage','captions','--no-training-config','--approve-all'])
        self.project = projects.load_project(self.root)
        self.records = projects.load_records(self.root,self.project)
        self.args = cli.parser().parse_args(['--no-training-config'])

    def test_goals_and_unknown_context_are_distinct_without_numeric_recipe(self):
        outputs = []
        for goal in ('style','character','concept','pose','custom'):
            self.args.training_goal = goal
            text,meta = advice.render(self.root,list(self.records.values()),self.args)
            outputs.append(text)
            self.assertEqual(meta['context']['training_goal'],goal)
            self.assertIn('Exact learning rate, rank, steps and resolution are intentionally not inferred',text)
            self.assertNotIn('learning_rate: 0.0001',text)
        self.assertEqual(len(set(outputs)),5)

    def test_statistics_and_trigger_only_use_exported_records(self):
        self.args.trigger_word='trigger'
        text, meta = advice.render(self.root,[next(iter(self.records.values()))],self.args)
        self.assertEqual(meta['statistics']['samples'],1)
        self.assertIn('1/1 captions',text)
        _, both = advice.render(self.root,list(self.records.values()),self.args)
        self.assertEqual(both['statistics']['unique_image_hashes'],1)

    def test_goal_change_creates_new_snapshot_preserving_captions_and_reuses_it(self):
        old=self.project['latest_export']; hashes={p.name:sha(p) for p in (self.root/'captions').glob('*.txt')}
        with patch.object(cli,'ensure_models',side_effect=AssertionError('No model access')), \
             patch.object(cli,'run_models',side_effect=AssertionError('No inference')), \
             contextlib.redirect_stdout(io.StringIO()):
            cli.main(['--resume',str(self.root),'--export-only','--training-goal','style'])
            new=projects.load_project(self.root)['latest_export']
            cli.main(['--resume',str(self.root),'--export-only'])
        self.assertNotEqual(old,new); self.assertTrue((self.root/old/'snapshot.json').exists())
        self.assertEqual(new,projects.load_project(self.root)['latest_export'])
        self.assertEqual(hashes,{p.name:sha(p) for p in (self.root/'captions').glob('*.txt')})
        snapshot=json.loads((self.root/new/'snapshot.json').read_text())
        self.assertIn('TRAINING_ADVICE.txt',snapshot['artifacts'])
        manifest=json.loads((self.root/new/'dataset_manifest.json').read_text())
        self.assertEqual(manifest['training_advice']['context']['training_goal'],'style')

    def test_onetrainer_advice_matches_config_and_rejects_full_recipe(self):
        self.args.no_training_config=False; self.args.training_method='lora'
        export=projects.export_snapshot(self.root,self.project,self.records,self.args)
        root=self.root/export['path']
        config=json.loads((root/'onetrainer/train.json').read_text())
        text=(root/'TRAINING_ADVICE.txt').read_text()
        self.assertIn(f"learning_rate: {config['learning_rate']}",text)
        self.assertIn(f"lora_rank: {config['lora_rank']}",text)
        self.assertIn(f"epochs: {config['epochs']}",text)
        self.args.training_method='full'
        with self.assertRaisesRegex(ValueError,'LoRA only'):
            projects.export_snapshot(self.root,self.project,self.records,self.args)

    def test_independent_model_choice_changes_only_dependent_fingerprints(self):
        record=next(iter(self.records.values())); prompts={'instruction':'x','system':'y','regions':'z'}
        self.args.profile='compact'
        original=projects.fingerprints(record,self.args,prompts)
        self.args.caption_model='qwen3-vl-8b'
        modified=projects.fingerprints(record,self.args,prompts)
        self.assertNotEqual(original['caption'],modified['caption'])
        self.assertEqual(original['grounding'],modified['grounding']); self.assertEqual(original['sam'],modified['sam'])
        repos,revisions=engine.model_specs('compact',selections(self.args))
        self.assertEqual(repos['caption'],'Qwen/Qwen3-VL-8B-Instruct')
        self.assertEqual(revisions['caption'],CATALOG['qwen3-vl-8b']['revision'])
        self.assertEqual(engine.model_path(self.base,'caption','compact',selections(self.args)),engine.model_path(self.base,'grounding','quality'))
        with self.assertRaises(ValueError): selected('compact',{'sam':'qwen3-vl-8b'})

    def test_explicit_model_survives_resume_without_rewriting_imported_caption(self):
        with patch.object(cli,'run_models',side_effect=AssertionError('Imported captions protected')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['--resume',str(self.root),'--caption-model','qwen3-vl-8b']),0)
            self.assertEqual(cli.main(['--resume',str(self.root)]),0)
        self.assertEqual(projects.load_project(self.root)['options']['caption_model'],'qwen3-vl-8b')

    def test_settings_keep_large_variants_selectable_on_small_and_large_hosts(self):
        try:
            from PySide6.QtWidgets import QApplication
            from settings_dialog import SettingsDialog
        except ImportError: self.skipTest('Qt not installed')
        app=QApplication.instance() or QApplication([])
        dialog=SettingsDialog({'profile':'compact'},probe=False)
        for available in (4,96):
            dialog.host={'device':'cuda','available_gib':available}
            box=dialog.model_boxes['caption']; box.setCurrentIndex(box.findData('qwen3-vl-8b')); dialog.show_models()
            self.assertEqual(dialog.values()['caption_model'],'qwen3-vl-8b')
            self.assertTrue(box.model().item(box.currentIndex()).isEnabled())
        self.assertIn('exceeds this estimate',dialog.model_details.toPlainText())
        dialog.close()


if __name__ == '__main__': unittest.main()
