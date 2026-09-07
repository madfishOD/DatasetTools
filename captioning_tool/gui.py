"""Local Qt caption editor. Inference stays in the existing CLI subprocess."""
import argparse
import codecs
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from PySide6.QtCore import QProcess, QSettings, Qt, QUrl, QSize
from PySide6.QtGui import QDesktopServices, QImageReader, QPixmap, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget, QDialog,
    QDialogButtonBox, QLineEdit)

import project as projects
from training_export import atomic_write

HERE = Path(__file__).resolve().parent


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('New project — input and output')
        self.setMinimumWidth(660)
        layout = QVBoxLayout(self)
        self.source = QLineEdit(); self.target = QLineEdit()
        for field, title, description, button_text in (
            (self.source, '1. Input folder — source images',
             'The existing folder containing your images and optional TXT captions.', 'Browse input…'),
            (self.target, '2. Output folder — project and results',
             'A separate empty or new folder for image copies, captions, and exports.', 'Browse output…'),
        ):
            label = QLabel(title); label.setBuddy(field); layout.addWidget(label)
            note = QLabel(description); note.setWordWrap(True); layout.addWidget(note)
            row = QHBoxLayout(); layout.addLayout(row)
            field.setAccessibleName(title); field.setPlaceholderText('Full folder path')
            row.addWidget(field)
            button = QPushButton(button_text); row.addWidget(button)
            button.clicked.connect(lambda checked=False, edit=field, caption=title: self.browse(edit, caption))
        note = QLabel('Creating a project imports your files. Use Run / Resume afterward to generate captions.')
        note.setWordWrap(True); layout.addWidget(note)
        self.error_label = QLabel(); self.error_label.setWordWrap(True); layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Create and import')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('Cancel')
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)

    def browse(self, field, title):
        path = QFileDialog.getExistingDirectory(self, title, field.text())
        if path: field.setText(path)

    def accept(self):
        try:
            if not self.source.text().strip() or not self.target.text().strip():
                raise ValueError('Specify both folders: Input for source images and Output for project results.')
            source = Path(self.source.text().strip()).expanduser().resolve()
            root = Path(self.target.text().strip()).expanduser().resolve()
            if not source.is_dir():
                raise ValueError('Input: the source image folder does not exist.')
            if root == source or root in source.parents or source in root.parents:
                raise ValueError('Input and Output must be separate folders; neither can be inside the other.')
            if root.exists() and (not root.is_dir() or any(root.iterdir())):
                raise ValueError('Output: choose an empty folder or enter a path for a new folder.')
            self.paths = source, root
        except (ValueError, OSError) as error:
            self.error_label.setText(str(error)); return
        super().accept()


class Window(QMainWindow):
    def __init__(self, root=None, previews=True):
        super().__init__()
        self.setWindowTitle('Dataset Tools — Captions')
        self.resize(1120, 780)
        self.root = None
        self.records = {}
        self.selected = None
        self.loaded_text = ''
        self.loaded_sha = None
        self.previews = previews
        self.busy = False
        self.stop_dir = None
        self.completed = 0
        self.settings = QSettings('DatasetTools', 'Captioning')
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.buffer = ''

        body = QWidget(); self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        bar = QHBoxLayout(); layout.addLayout(bar)
        self.new_button = self.button(bar, 'New project…', self.new_project)
        self.open_button = self.button(bar, 'Open project…', self.choose_project)
        self.recent_button = self.button(bar, 'Recent project', self.open_recent)
        self.refresh_button = self.button(bar, 'Refresh', self.refresh_clicked)
        self.path_label = QLabel('Open a project or import an image folder.')
        self.path_label.setWordWrap(True); layout.addWidget(self.path_label)
        controls = QHBoxLayout(); layout.addLayout(controls)
        self.run_button = self.button(controls, 'Run / Resume', lambda: self.run())
        self.regenerate_button = self.button(controls, 'Regenerate selected', self.regenerate)
        self.stop_button = self.button(controls, 'Stop', self.stop)
        self.export_button = self.button(controls, 'Export approved', self.export)
        self.folder_button = self.button(controls, 'Open export folder', self.open_export)
        self.progress = QProgressBar(); self.progress.setRange(0, 1); self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel('Ready'); self.status.setWordWrap(True); layout.addWidget(self.status)

        split = QSplitter(); layout.addWidget(split, 1)
        left = QWidget(); left_layout = QVBoxLayout(left); split.addWidget(left)
        self.filter = QComboBox(); self.filter.addItems(['All', 'Unreviewed', 'Approved', 'Errors'])
        self.filter.setAccessibleName('Filter samples')
        self.filter.currentIndexChanged.connect(self.filter_changed); left_layout.addWidget(self.filter)
        self.samples = QListWidget(); self.samples.setAccessibleName('Project images')
        self.samples.currentItemChanged.connect(self.selection_changed); left_layout.addWidget(self.samples)
        right = QWidget(); right_layout = QVBoxLayout(right); split.addWidget(right)
        self.preview = QLabel('Select a sample'); self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(300, 200); right_layout.addWidget(self.preview, 1)
        self.details = QLabel(); self.details.setWordWrap(True); right_layout.addWidget(self.details)
        right_layout.addWidget(QLabel('Caption — saving an edit resets approval'))
        self.editor = QPlainTextEdit(); self.editor.setAccessibleName('Caption text'); right_layout.addWidget(self.editor)
        actions = QHBoxLayout(); right_layout.addLayout(actions)
        self.save_button = self.button(actions, 'Save edit', lambda: self.save(False))
        self.approve_button = self.button(actions, 'Save and approve', lambda: self.save(True))
        self.unapprove_button = self.button(actions, 'Remove approval', lambda: self.save(False))
        self.open_button.setShortcut(QKeySequence.StandardKey.Open)
        self.save_button.setShortcut(QKeySequence.StandardKey.Save)
        split.setSizes([320, 760])
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(120)
        self.log.setMaximumBlockCount(500); self.log.setAccessibleName('Processing log'); layout.addWidget(self.log)
        self.set_busy(False)
        if root:
            self.open_project(root)

    def button(self, layout, title, handler):
        button = QPushButton(title); button.clicked.connect(handler); layout.addWidget(button)
        return button

    def error(self, error):
        self.status.setText(str(error))
        QMessageBox.warning(self, 'Dataset Tools', str(error))

    def dirty(self):
        return self.selected is not None and self.editor.toPlainText() != self.loaded_text

    def can_leave(self):
        if not self.dirty():
            return True
        choice = QMessageBox.question(self, 'Unsaved edit', 'Save changes to this caption?',
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if choice == QMessageBox.StandardButton.Save:
            return self.save(False)
        return choice == QMessageBox.StandardButton.Discard

    def set_busy(self, busy):
        self.busy = busy
        for button in (self.new_button, self.open_button, self.recent_button):
            button.setEnabled(not busy)
        for button in (self.refresh_button, self.run_button, self.export_button, self.folder_button):
            button.setEnabled(not busy and self.root is not None)
        for button in (self.save_button, self.approve_button, self.unapprove_button, self.regenerate_button):
            button.setEnabled(not busy and self.selected is not None)
        self.editor.setReadOnly(busy)
        self.samples.setEnabled(not busy); self.filter.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def choose_project(self):
        if not self.can_leave(): return
        path = QFileDialog.getExistingDirectory(self, 'Open the folder containing project.json')
        if path: self.open_project(Path(path))

    def open_recent(self):
        if not self.can_leave(): return
        path = self.settings.value('recent', '')
        if path: self.open_project(Path(path))

    def new_project(self):
        if not self.can_leave(): return
        dialog = NewProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted: return
        source, root = dialog.paths
        self.root = root; self.records = {}; self.populate()
        self.path_label.setText(str(root))
        self.start_worker(['--input', str(source), '--output', str(root), '--import-only',
            '--profile', 'compact', '--device', 'auto', '--stage', 'captions', '--no-training-config',
            '--prompt', str(HERE / 'prompts/neutral/general.txt')])

    def open_project(self, root):
        try:
            root = Path(root).resolve()
            project = projects.load_project(root)
            records = projects.load_records(root, project)
            projects.sync_edits(root, records, write=False)
        except Exception as error:
            self.error(error); return False
        self.root = root; self.records = records
        self.settings.setValue('recent', str(root))
        self.path_label.setText(str(root)); self.populate(); self.set_busy(False)
        return True

    def refresh_clicked(self):
        if self.can_leave() and self.root: self.open_project(self.root)

    def filter_changed(self):
        if self.can_leave(): self.populate()
        else:
            self.filter.blockSignals(True); self.filter.setCurrentIndex(self.previous_filter); self.filter.blockSignals(False)

    def populate(self):
        previous = self.selected
        self.previous_filter = self.filter.currentIndex()
        self.samples.blockSignals(True); self.samples.clear()
        chosen = None
        for record in self.records.values():
            approved = record.get('review_status') == 'approved'
            failed = any(s.get('status') == 'failed' for s in record['stages'].values())
            mode = self.filter.currentIndex()
            if (mode == 1 and approved) or (mode == 2 and not approved) or (mode == 3 and not failed): continue
            status = '✓' if approved else ('!' if failed else '○')
            item = QListWidgetItem(f"{status}  {record['source_file']}")
            item.setData(Qt.ItemDataRole.UserRole, record['sample_id']); self.samples.addItem(item)
            if record['sample_id'] == previous: chosen = item
        self.samples.setCurrentItem(chosen or self.samples.item(0)); self.samples.blockSignals(False)
        self.show_selected(self.samples.currentItem())

    def selection_changed(self, current, previous):
        target = current.data(Qt.ItemDataRole.UserRole) if current else None
        if not self.can_leave():
            self.samples.blockSignals(True); self.samples.setCurrentItem(previous); self.samples.blockSignals(False); return
        # Saving may rebuild the list; find the intended selection again by stable ID.
        self.samples.blockSignals(True)
        for index in range(self.samples.count()):
            item = self.samples.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == target:
                self.samples.setCurrentItem(item); break
        self.samples.blockSignals(False)
        self.show_selected(self.samples.currentItem())

    def show_selected(self, item):
        self.selected = item.data(Qt.ItemDataRole.UserRole) if item else None
        self.editor.clear(); self.loaded_text = ''; self.loaded_sha = None; self.preview.clear(); self.details.clear()
        if item:
            record = next(r for r in self.records.values() if r['sample_id'] == self.selected)
            path = projects.caption_path(self.root, record)
            try:
                raw = path.read_bytes() if path.exists() else b''
                text = raw.decode('utf-8-sig')
                self.loaded_sha = hashlib.sha256(raw).hexdigest() if path.exists() else None
                self.editor.setPlainText(text)
                # Qt normalizes newlines; retain that baseline, preserve disk bytes on unchanged save.
                self.loaded_text = self.editor.toPlainText()
                self.disk_text = text
            except Exception as error:
                self.error(error); self.selected = None; self.set_busy(self.busy); return
            self.details.setText(f"{record['source_file']} • {record.get('review_status', 'unreviewed')}\n"
                                 f"Caption: {record['stages'].get('caption', {}).get('status', 'missing')}")
            if self.previews:
                reader = QImageReader(str(self.root / 'images' / record['file']))
                reader.setAutoTransform(True)
                size = reader.size()
                if size.isValid(): reader.setScaledSize(size.scaled(QSize(720, 360), Qt.AspectRatioMode.KeepAspectRatio))
                picture = reader.read()
                if picture.isNull(): self.preview.setText('Unable to read this image')
                else: self.preview.setPixmap(QPixmap.fromImage(picture))
        else:
            self.preview.setText('No samples match this filter')
        self.set_busy(self.busy)

    def save(self, approved):
        if not self.selected or self.busy: return False
        try:
            text = self.editor.toPlainText() if self.dirty() else self.disk_text
            projects.edit_caption(self.root, self.selected, text, self.loaded_sha, approved)
            self.open_project(self.root)
            self.status.setText('Caption approved' if approved else 'Edit saved; approval required')
            return True
        except Exception as error:
            self.error(error); return False

    def run(self, extra=None):
        if self.root and self.can_leave():
            self.start_worker(['--resume', str(self.root)] + (extra or []))

    def regenerate(self):
        if self.selected: self.run(['--caption-policy', 'regenerate', '--select', self.selected, '--stage', 'captions'])

    def export(self):
        self.run(['--export-only', '--no-training-config'])

    def open_export(self):
        try:
            project = projects.load_project(self.root)
            relative = project.get('latest_export')
            if not relative:
                self.status.setText('Export approved captions first.'); return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(projects.local(self.root, relative))))
        except Exception as error: self.error(error)

    def start_worker(self, arguments):
        if self.busy: return
        self.stop_dir = tempfile.TemporaryDirectory(prefix='datasettools-worker-')
        self.stop_path = Path(self.stop_dir.name) / 'stop'
        self.buffer = ''; self.decoder.reset(); self.completed = 0
        self.last_summary = {}
        self.log.clear(); self.progress.setRange(0, 0); self.status.setText('Preparing…')
        self.set_busy(True)
        self.process.start(sys.executable, ['-u', str(HERE / 'auto_captioning_tool.py'),
            *arguments, '--events', '--stop-file', str(self.stop_path)])

    def stop(self):
        if self.busy:
            atomic_write(self.stop_path, b'stop')
            self.stop_button.setEnabled(False)
            self.status.setText('Stopping after the current operation. Loading a model may take some time.')

    def read_output(self):
        self.buffer += self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.consume_line(line)

    def consume_line(self, line):
        if line.startswith('DATASET_EVENT '):
            try:
                event = json.loads(line[len('DATASET_EVENT '):])
                if event['event'] == 'plan':
                    total = sum(event['pending'].values()); self.progress.setRange(0, max(1, total)); self.progress.setValue(0)
                    self.status.setText(f'Planned operations: {total}')
                elif event['event'] == 'stage':
                    if event['status'] in ('complete', 'failed'):
                        self.completed += 1; self.progress.setValue(self.completed)
                    self.status.setText(f"{event['stage']}: {event['status']} • completed {self.completed}")
                elif event['event'] == 'result':
                    self.last_summary = event['summary']
            except (ValueError, KeyError, TypeError): self.log.appendPlainText(line)
        else: self.log.appendPlainText(line)

    def finished(self, code, exit_status):
        self.read_output()
        self.buffer += self.decoder.decode(b'', final=True)
        if self.buffer: self.consume_line(self.buffer); self.buffer = ''
        if self.stop_dir: self.stop_dir.cleanup(); self.stop_dir = None
        self.set_busy(False)
        if self.root and (self.root / 'project.json').exists(): self.open_project(self.root)
        self.progress.setRange(0, 1); self.progress.setValue(1 if code == 0 else 0)
        if exit_status == QProcess.ExitStatus.CrashExit:
            message = 'The worker crashed. Saved checkpoints are available; you can resume. See the log.'
        elif code == 130: message = 'Stopped. Completed operations are saved; you can resume.'
        elif code: message = 'Processing finished with an error. See the log; you can resume.'
        else:
            count = self.last_summary.get('export', {}).get('accepted_pairs', 0)
            message = f'Done. Approved pairs in export: {count}.'
        self.status.setText(message)

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.finished(1, QProcess.ExitStatus.NormalExit)
            self.status.setText('Unable to start the Python worker: ' + self.process.errorString())

    def closeEvent(self, event):
        if self.busy:
            self.stop(); event.ignore()
        elif self.can_leave(): event.accept()
        else: event.ignore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path)
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = Window(args.project); window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
