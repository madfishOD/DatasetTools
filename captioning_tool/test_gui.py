"""Headless GUI behavior tests: no image display or semantic caption evaluation."""
import contextlib
import io
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_captioning_tool as cli
import project as projects
from training_export import atomic_write, sha
import test_project

try:
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication, QMessageBox
    from gui import Window
except ImportError:
    Window = None


class EditorStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project'
        self.source = Path(self.temp.name) / 'source'; self.source.mkdir()
        (self.source / 'a.jpg').write_bytes(b'fixture-image')
        (self.source / 'a.txt').write_bytes(b'Caption\r\n')
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(['--input', str(self.source), '--output', str(self.root), '--import-only',
                      '--stage', 'captions', '--no-training-config'])
        self.project = projects.load_project(self.root)
        self.record = next(iter(projects.load_records(self.root, self.project).values()))
        self.path = projects.caption_path(self.root, self.record)

    def test_approval_preserves_bytes_and_conflict_rejects_stale_editor(self):
        checksum = sha(self.path)
        projects.edit_caption(self.root, self.record['sample_id'], 'Caption\r\n', checksum, True)
        self.assertEqual(self.path.read_bytes(), b'Caption\r\n')
        self.path.write_bytes(b'External edit')
        with self.assertRaisesRegex(ValueError, 'outside this editor'):
            projects.edit_caption(self.root, self.record['sample_id'], 'Overwrite', checksum)
        self.assertEqual(self.path.read_bytes(), b'External edit')
        with self.assertRaises(ValueError):
            projects.edit_caption(self.root, self.record['sample_id'], '', sha(self.path), True)

    def test_cooperative_stop_commits_current_caption_and_resume_skips_it(self):
        self.path.unlink()
        (self.source / 'b.jpg').write_bytes(b'fixture-image-b')
        stop = Path(self.temp.name) / 'stop'
        def engine(out, files, records, args, prompts, work, checkpoint_fn):
            first = True
            def checkpoint(image, stage, status, error=None):
                nonlocal first
                if status == 'complete' and first:
                    stop.touch(); first = False
                checkpoint_fn(image, stage, status, error)
            test_project.ProjectTests.fake_engine(self, out, files, records, args, prompts, work, checkpoint)
        options = ['--resume', str(self.root), '--input', str(self.source), '--events', '--stop-file', str(stop)]
        with patch.object(cli, 'resolve_device', return_value=('cpu', 'float32')), \
             patch.object(cli, 'ensure_models'), patch.object(cli, 'run_models', side_effect=engine), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(options), 130)
        records = projects.load_records(self.root, projects.load_project(self.root))
        self.assertEqual(sum(projects.artifact_valid(self.root, r, 'caption') for r in records.values()), 1)
        events = [json.loads(line.removeprefix('DATASET_EVENT ')) for line in output.getvalue().splitlines() if line.startswith('DATASET_EVENT ')]
        self.assertTrue(events[-1]['summary']['interrupted'])
        with patch.object(cli, 'resolve_device', return_value=('cpu', 'float32')), \
             patch.object(cli, 'ensure_models'), patch.object(cli, 'run_models', side_effect=lambda *a, **k: test_project.ProjectTests.fake_engine(self, *a, **k)) as runner, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['--resume', str(self.root)]), 0)
        self.assertEqual(len(runner.call_args.kwargs['work']['caption']), 1)


@unittest.skipIf(Window is None, 'Install requirements-gui.txt to run Qt tests')
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        EditorStorageTests.setUp(self)
        with patch('gui.credentials.load_token', return_value=('', '')):
            self.window = Window(self.root, previews=False)
        self.window.settings = type('NoSettings', (), {'setValue': lambda *args: None})()
        self.addCleanup(self.window.close)

    def wait_worker(self):
        deadline = time.monotonic() + 15
        while self.window.busy and time.monotonic() < deadline:
            self.app.processEvents(); time.sleep(.01)
        if self.window.busy:
            self.window.process.kill(); self.window.process.waitForFinished(2000)
            self.fail('Worker timed out')

    def test_save_approve_export_and_noop_resume(self):
        window = self.window
        self.assertFalse(window.dirty())
        self.assertTrue(window.save(True))
        self.assertEqual(self.path.read_bytes(), b'Caption\r\n')
        window.editor.setPlainText('Edited caption')
        self.assertTrue(window.save(False))
        self.assertEqual(next(iter(window.records.values()))['review_status'], 'unreviewed')
        self.assertTrue(window.save(True))
        window.export(); self.assertTrue(window.busy); self.assertTrue(window.editor.isReadOnly())
        self.wait_worker()
        project = projects.load_project(self.root)
        snapshot = project['latest_export']; self.assertIsNotNone(snapshot)
        self.assertEqual(window.last_summary['export']['accepted_pairs'], 1)
        window.run(); self.wait_worker()
        self.assertEqual(window.last_summary['pending_at_start']['caption'], 0)
        self.assertEqual(projects.load_project(self.root)['latest_export'], snapshot)

    def test_worker_crash_keeps_editor_usable(self):
        self.window.set_busy(True)
        self.window.process.start(sys.executable, ['-c', 'import os; os.abort()'])
        self.wait_worker()
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertIn('crashed', self.window.status.text())
        self.assertTrue(self.window.save(True))

    def test_session_token_environment_and_clear_preserve_existing_auth(self):
        with patch.dict(os.environ, {'HF_TOKEN':'hf_existingFixture', 'HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}, clear=True):
            self.window.hf_token = 'hf_sessionFixture'
            environment = self.window.worker_environment()
            self.assertEqual(environment.value('HF_TOKEN'), 'hf_sessionFixture')
            self.assertFalse(environment.contains('HF_HUB_DISABLE_IMPLICIT_TOKEN'))
            self.assertEqual(os.environ['HF_TOKEN'], 'hf_existingFixture')
            self.window.hf_token = ''
            environment = self.window.worker_environment()
            self.assertEqual(environment.value('HF_TOKEN'), 'hf_existingFixture')
            self.assertEqual(environment.value('HF_HUB_DISABLE_IMPLICIT_TOKEN'), '1')

    def test_token_reaches_worker_but_not_log_or_project(self):
        self.window.hf_token = 'hf_sessionFixture'
        self.window.set_busy(True)
        self.window.process.setProcessEnvironment(self.window.worker_environment())
        self.window.process.start(sys.executable, ['-c',
            "import os; print(os.environ['HF_TOKEN']); print('authenticated environment received')"])
        self.wait_worker()
        self.assertIn('authenticated environment received', self.window.log.toPlainText())
        self.assertIn('[REDACTED]', self.window.log.toPlainText())
        self.assertNotIn('hf_sessionFixture', self.window.log.toPlainText())
        for path in self.root.rglob('*'):
            if path.is_file(): self.assertNotIn(b'hf_sessionFixture', path.read_bytes())

    def test_unsaved_cancel_keeps_current_selection(self):
        self.window.editor.setPlainText('Unsaved')
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Cancel):
            self.assertFalse(self.window.can_leave())
        self.assertEqual(self.window.editor.toPlainText(), 'Unsaved')
        self.window.editor.setPlainText(self.window.loaded_text)


if __name__ == '__main__': unittest.main()
