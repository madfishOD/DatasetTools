"""Project training intent and explicit model choices, independent of host capacity."""
import json
from pathlib import Path
import sys
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
    QLineEdit, QPlainTextEdit, QTabWidget, QVBoxLayout, QWidget)
from model_catalog import CATALOG, selected
from training_advice import CONTEXT_DEFAULTS


class SettingsDialog(QDialog):
    def __init__(self, options, parent=None, probe=True):
        super().__init__(parent)
        self.setWindowTitle('Project settings — training intent and processing models')
        self.resize(760, 700)
        self.options = options
        self.fields = {}; self.model_boxes = {}; self.host = None
        layout = QVBoxLayout(self)
        tabs = QTabWidget(); layout.addWidget(tabs)
        training = QWidget(); form = QFormLayout(training); tabs.addTab(training, 'Training intent')
        self.combo(form, 'training_goal', 'Learning goal', [('Not specified','unspecified'),('Style','style'),('Character / identity','character'),('Concept / object','concept'),('Pose / interaction','pose'),('Custom','custom')])
        self.combo(form, 'training_family', 'Training model family', [(x.title(),x) for x in ('unknown','sdxl','flux','qwen','wan','other')])
        self.combo(form, 'training_method', 'Training method', [('Unknown','unknown'),('LoRA','lora'),('Full fine-tuning','full')])
        for key, label, placeholder in (
            ('base_model','Training base model (optional)','Exact repository ID or local model path'),
            ('goal_description','Goal and controllable attributes','What should stay fixed? What should remain variable?'),
            ('trigger_word','Trigger word (optional)','For example: my_character'),
            ('trainer','Trainer (optional)','OneTrainer, kohya, Diffusers, or another tool'),
            ('training_hardware','Training hardware (optional)','May be a different computer or cloud GPU'),
        ):
            edit = QLineEdit(str(options.get(key, CONTEXT_DEFAULTS[key])))
            edit.setPlaceholderText(placeholder); edit.setAccessibleName(label)
            form.addRow(label, edit); self.fields[key] = edit
        note = QLabel('These fields guide TRAINING_ADVICE.txt for approved exports. They do not rewrite captions or start training. Unknown context stays explicit; exact optimal parameters are not inferred.')
        note.setWordWrap(True); form.addRow(note)
        models = QWidget(); model_layout = QVBoxLayout(models); tabs.addTab(models, 'Processing models')
        self.hardware_label = QLabel('Detecting local hardware…' if probe else 'Hardware detection not requested.')
        self.hardware_label.setWordWrap(True); model_layout.addWidget(self.hardware_label)
        model_form = QFormLayout(); model_layout.addLayout(model_form)
        self.combo(model_form, 'device', 'Processing device', [(x,x) for x in ('auto','cuda','mps','cpu')])
        self.combo(model_form, 'dtype', 'Precision', [(x,x) for x in ('auto','float16','bfloat16','float32')])
        self.combo(model_form, 'stage', 'Stages to run', [('Captions','captions'),('Regions and masks','regions'),('All stages','all')])
        names = selected(options.get('profile','compact'), {s:options.get(s+'_model') for s in ('caption','grounding','sam')})
        for stage,label in (('caption','Caption model'),('grounding','Region model'),('sam','Mask model')):
            box = QComboBox(); box.setAccessibleName(label)
            for name,spec in CATALOG.items():
                if stage in spec['stages']: box.addItem(spec['label'], name)
            box.setCurrentIndex(box.findData(names[stage])); model_form.addRow(label,box)
            self.model_boxes[stage] = box; box.currentIndexChanged.connect(self.show_models)
        self.model_details = QPlainTextEdit(); self.model_details.setReadOnly(True)
        self.model_details.setAccessibleName('Selected model revisions and hardware estimates'); model_layout.addWidget(self.model_details)
        note = QLabel('All supported variants remain selectable, including models for larger GPUs. Budgets are rough working-memory estimates for unquantized inference, not minimum requirements or fit guarantees. Stages run sequentially. Float32 can require more memory. No automatic fallback or model substitution occurs.')
        note.setWordWrap(True); model_layout.addWidget(note)
        self.fields['device'].currentIndexChanged.connect(self.show_models)
        self.show_models()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.probe = QProcess(self)
        self.probe.finished.connect(self.hardware_ready)
        self.probe.errorOccurred.connect(lambda _: self.hardware_label.setText('Hardware detection failed. Models remain selectable; verify the target device before inference.'))
        if probe: self.probe.start(sys.executable, [str(Path(__file__).with_name('model_catalog.py'))])

    def combo(self, form, key, label, choices):
        box = QComboBox(); box.setAccessibleName(label)
        for text,value in choices: box.addItem(text,value)
        index = box.findData(self.options.get(key, CONTEXT_DEFAULTS.get(key, 'auto' if key in ('device','dtype') else 'captions')))
        box.setCurrentIndex(max(0,index)); form.addRow(label,box); self.fields[key] = box

    def hardware_ready(self, code, status):
        try:
            if code: raise ValueError('Hardware probe failed')
            self.host = json.loads(bytes(self.probe.readAllStandardOutput()).decode())
            h = self.host
            self.hardware_label.setText(f"Detected: {h['name']} • {h['available_gib']} GiB available / {h['total_gib']} GiB total. System RAM: {h['ram_total_gib']} GiB. Availability is a snapshot; MPS shares RAM with the system.")
            self.show_models()
        except (ValueError, KeyError): self.hardware_label.setText('Hardware information unavailable. Model choice is still available.')

    def show_models(self):
        rows = []
        device = self.fields['device'].currentData()
        for stage,box in self.model_boxes.items():
            spec = CATALOG[box.currentData()]
            rows += [f"{stage}: {spec['label']}", f"Repository: {spec['repo']}", f"Pinned revision: {spec['revision']}",
                     f"Estimated working budget: ~{spec['budget_gib']} GiB. {spec['note']}"]
            if self.host and device in ('auto', self.host['device']):
                if self.host['available_gib'] < spec['budget_gib']:
                    rows += ['Current available memory is below this estimate; this run may run out of memory.']
                else: rows += ['Available memory exceeds this estimate; device compatibility and peak usage still need validation.']
            elif self.host:
                rows += ['Selected device differs from the detected accelerator; its available memory was not measured.']
            rows.append('')
        self.model_details.setPlainText('\n'.join(rows))

    def values(self):
        result = {key:(field.currentData() if isinstance(field,QComboBox) else field.text().strip()) for key,field in self.fields.items()}
        result.update({stage+'_model':box.currentData() for stage,box in self.model_boxes.items()})
        return result

    def done(self, result):
        if self.probe.state() != QProcess.ProcessState.NotRunning:
            self.probe.kill(); self.probe.waitForFinished(1000)
        super().done(result)
