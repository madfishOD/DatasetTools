"""OneTrainer export. Never edits image or caption content."""
import hashlib, json, math, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROFILES = {
 'qwen': ('QWEN', 'Qwen/Qwen-Image', 768),
 'sdxl': ('STABLE_DIFFUSION_XL_10_BASE', 'stabilityai/stable-diffusion-xl-base-1.0', 1024),
 'flux': ('FLUX_DEV_1', 'black-forest-labs/FLUX.1-dev', 768),
 'wan': (None, '', 512),
}
def sha(path):
 with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def merge(base,patch):
 for key,value in patch.items():
  if isinstance(value,dict) and isinstance(base.get(key),dict): merge(base[key],value)
  else: base[key]=value
 return base

def training_plan(count,family,target_steps=None,epochs=None):
 if count<0: raise ValueError('Negative dataset size')
 target=target_steps if target_steps is not None else min(3000,max(500,count*20))
 if target<1 or (epochs is not None and epochs<1): raise ValueError('Steps/epochs must be positive')
 actual=(epochs or math.ceil(target/count)) if count else 0
 return dict(family=family,accepted_pairs=count,batch_size=1,gradient_accumulation_steps=1,
  epochs=actual,target_optimizer_steps=target,estimated_optimizer_steps=count*actual,
  estimated_exposures_per_image=actual,
  heuristic='Target = clamp(20*N,500,3000); epochs = ceil(target/N). N is accepted pairs. No repeats.',
  notes=['Starting point only; dataset size does not determine optimal LR, rank, resolution or VRAM.',
   'Step estimate assumes all exported images load, no validation split and one image/text variation.',
   'Masks require review and are not used for masked training.']+
   (['Very small dataset; high repetition can overfit. Review checkpoints and reduce epochs.'] if 0<count<20 else []))

def export_training(out,files,args):
 dataset_only=getattr(args,'no_training_config',False)
 root=out/('dataset' if dataset_only else 'onetrainer');root.mkdir(exist_ok=False);data=root/'data';data.mkdir()
 included=[];excluded=[]
 for image in files:
  caption=out/'captions'/(image.stem+'.txt')
  if not caption.is_file() or not caption.read_bytes().strip():
   excluded.append(dict(file=image.name,reason='No accepted caption'));continue
  shutil.copy2(image,data/image.name);shutil.copy2(caption,data/caption.name)
  ih,ch=sha(image),sha(caption)
  if sha(data/image.name)!=ih or sha(data/caption.name)!=ch: raise IOError('Export hash mismatch')
  included.append(dict(file=image.name,image_sha256=ih,caption_sha256=ch))
 plan=training_plan(len(included),args.family,args.target_steps,args.epochs)
 plan.update(included=included,excluded=excluded,ready=False)
 model,base,res=PROFILES[args.family]
 if not included: plan['reason']='No accepted pairs. No train.json generated.'
 elif dataset_only: plan['reason']='Dataset-only export requested. No trainer configuration generated.'
 elif model is None: plan['reason']='WAN is not supported by the checked OneTrainer version. Dataset only; no incompatible train.json generated.'
 else:
  defaults=json.loads((HERE/'templates/defaults.json').read_text(encoding='utf-8'))
  preset=json.loads((HERE/'templates'/(args.family+'.json')).read_text(encoding='utf-8'))
  config=merge(defaults,preset)
  config.update(base_model_name=args.base_model or base,training_method='LORA',model_type=model,
   concept_file_name=str(root/'concepts.json'),concepts=None,
   workspace_dir=str(root/'workspace'),cache_dir=str(root/'cache'),
   output_model_destination=str(root/'models/lora.safetensors'),
   sample_definition_file_name=str(root/'samples.json'),samples=None,
   sample_after_unit='NEVER',backup_after_unit='NEVER',backup_before_save=False,
   save_every=1,save_every_unit='EPOCH',save_filename_prefix='auto_captioning_',
   batch_size=1,gradient_accumulation_steps=1,epochs=plan['epochs'],
   learning_rate=args.learning_rate,learning_rate_scheduler='CONSTANT',
   learning_rate_warmup_steps=min(100,plan['estimated_optimizer_steps']//20),
   resolution=str(args.resolution or res),lora_rank=args.rank,lora_alpha=float(args.rank),
   masked_training=False,train_dtype='BFLOAT_16',output_dtype='BFLOAT_16')
  config['optimizer'].update(optimizer='ADAMW',weight_decay=0.01,stochastic_rounding=True)
  concept=json.loads((HERE/'templates/concept.json').read_text(encoding='utf-8'))
  concept.update(name='auto_captioning_tool',path=str(data),enabled=True,include_subdirectories=False,
   balancing=1.0,image_variations=1,text_variations=1)
  concept['text'].update(prompt_source='sample',enable_tag_shuffling=False)
  concept['image'].update(enable_random_flip=False,enable_fixed_flip=False)
  save(root/'concepts.json',[concept]);save(root/'samples.json',[]);save(root/'train.json',config)
  plan.update(ready=True,base_model=config['base_model_name'],resolution=config['resolution'],
   learning_rate=config['learning_rate'],lora_rank=config['lora_rank'])
 save(root/'training_plan.json',plan)
 (root/'README.txt').write_text((
  'Dataset-only export: image/TXT pairs are in data/. Review the generated captions before training.\n'
  'No training has been run or trainer configuration generated.\n' if dataset_only else
  'Load train.json in OneTrainer, review Concepts, model path and training_plan.json, then start manually.\n'
  'Repository IDs may need a download/login; local base models need all encoders and VAE.\n'
  'No training has been run. Presets are starting points, not a quality or VRAM guarantee.\n'
  'Paths are absolute. After moving this output, run --rebase-output NEW_FOLDER to update generated paths.\n'
  'Only accepted captions are exported, byte-for-byte. Source images are copied unchanged.\n'
  'Mask artifacts are for review, not training conditioning. Checkpoints save every epoch; allow disk space.\n'
  +plan.get('reason','')+'\n'),encoding='utf-8')
 return {k:plan[k] for k in ('ready','accepted_pairs','epochs','estimated_optimizer_steps')} | {'reason':plan.get('reason')}

def rebase(out):
 root=out.resolve()/'onetrainer';path=root/'train.json'
 if not path.is_file(): raise ValueError('No train.json in this output')
 config=json.loads(path.read_text(encoding='utf-8'))
 for key,rel in [('concept_file_name','concepts.json'),('workspace_dir','workspace'),('cache_dir','cache'),
  ('output_model_destination','models/lora.safetensors'),('sample_definition_file_name','samples.json')]: config[key]=str(root/rel)
 concepts=json.loads((root/'concepts.json').read_text(encoding='utf-8'))
 for concept in concepts: concept['path']=str(root/'data')
 save(root/'concepts.json',concepts);save(path,config)
 print('Rebased generated training paths. Review base_model_name separately.')
