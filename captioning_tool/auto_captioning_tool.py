"""Portable neutral captioning + regions + byte-preserving training dataset export."""
import argparse,datetime,hashlib,html,json,os,re,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from training_export import PROFILES,export_training,rebase,save,sha
from engine import ensure_models,run_models
from devices import resolve_device
EXTENSIONS={'.png','.jpg','.jpeg','.webp','.bmp'}

def discover(folder,recursive=True):
 if not folder.is_dir():raise ValueError('Input directory not found: '+str(folder))
 # Do not follow directory junctions/symlinks out of the selected dataset.
 paths=[]
 for current,dirs,files in os.walk(folder,followlinks=False):
  dirs[:]=sorted(d for d in dirs if not (Path(current)/d).is_symlink() and not (Path(current)/d).is_junction()) if recursive else []
  for name in sorted(files):
   f=Path(current)/name
   if f.suffix.lower() in EXTENSIONS and f.is_file():paths.append(f)
 return sorted(paths,key=lambda f:f.relative_to(folder).as_posix().casefold())

def identifiers(files,root):
 counts={}
 for f in files:counts[f.stem.casefold()]=counts.get(f.stem.casefold(),0)+1
 result={};used=set()
 for f in files:
  relative=f.relative_to(root).as_posix()
  stem=f.stem if counts[f.stem.casefold()]==1 and len(f.stem)<=100 else re.sub(r'[^\w.-]','_',f.stem)[:70]+'_'+hashlib.sha256(relative.encode()).hexdigest()[:16]
  if stem.casefold() in used:stem='image_'+hashlib.sha256(relative.encode()).hexdigest()
  if stem.casefold() in used:raise ValueError('Output identifier collision')
  used.add(stem.casefold());result[f]=stem+f.suffix.lower()
 return result

def positive(value):
 value=int(value)
 if value<1:raise argparse.ArgumentTypeError('Must be positive')
 return value

def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path);p.add_argument('--output',type=Path)
 p.add_argument('--prompt',type=Path,default=HERE/'prompts/neutral/instruction.txt')
 p.add_argument('--system-prompt',type=Path,default=HERE/'prompts/neutral/system.txt')
 p.add_argument('--region-prompt',type=Path,default=HERE/'prompts/neutral/regions.txt')
 p.add_argument('--models',type=Path,default=HERE/'models',help='Captioning model cache; downloads missing models automatically')
 p.add_argument('--packages',type=Path,help='Optional existing Python package directory')
 p.add_argument('--family',choices=PROFILES,default='qwen',help='Training architecture, independent of captioning model')
 p.add_argument('--base-model',default='',help='Training model repository ID or local path')
 p.add_argument('--stage',choices=['all','captions','regions'],default='all')
 p.add_argument('--profile',choices=['quality','compact'],default='quality')
 p.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='auto')
 p.add_argument('--dtype',choices=['auto','float16','bfloat16','float32'],default='auto')
 p.add_argument('--max-image-side',type=positive,default=None,help='Caption input longest side (Compact default: 768); original files are unchanged')
 p.add_argument('--no-training-config',action='store_true',help='Export image/TXT pairs without OneTrainer training config')
 p.add_argument('--no-recursive',action='store_true',help='Only images directly in the input folder')
 p.add_argument('--target-steps',type=positive);p.add_argument('--epochs',type=positive)
 p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--rank',type=positive,default=16)
 p.add_argument('--resolution',type=positive);p.add_argument('--caption-tokens',type=positive,default=320)
 p.add_argument('--interactive',action='store_true');p.add_argument('--check',action='store_true',help='Read-only preflight, no downloads or inference')
 p.add_argument('--download-models',action='store_true',help='Download captioning models only; no input required')
 p.add_argument('--rebase-output',type=Path,help='Update generated OneTrainer paths after moving an output folder')
 return p

def main():
 args=parser().parse_args()
 if args.rebase_output:rebase(args.rebase_output);return 0
 if args.interactive:
  for name in ('input','output','prompt','system_prompt','region_prompt'):
   current=getattr(args,name);answer=input(f'{name} [{current or "required"}] (Enter = default): ').strip().strip('"')
   if answer:setattr(args,name,Path(answer))
  answer=input(f'Training family: qwen / sdxl / flux / wan [{args.family}]: ').strip().lower()
  if answer:args.family=answer
  if args.family not in PROFILES:raise ValueError('Unknown model family')
  answer=input(f'Base model: local path or repository ID [{args.base_model or PROFILES[args.family][1]}]: ').strip().strip('"')
  if answer:args.base_model=answer
 if args.packages:sys.path.insert(0,str(args.packages.resolve()))
 args.models=args.models.resolve()
 if args.profile=='compact' and args.max_image_side is None:args.max_image_side=768
 if args.download_models:
  missing=ensure_models(args.models,args.stage,check=args.check,profile=args.profile);print(json.dumps({'missing_models':missing,'check_only':args.check}));return 0
 if args.input is None:raise ValueError('--input is required (or use --interactive)')
 if not 0<args.learning_rate<1:raise ValueError('Learning rate must be between 0 and 1')
 args.input=args.input.resolve()
 out=(args.output or HERE/'outputs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
 if out==args.input or args.input in out.parents:raise ValueError('Output must be outside input directory')
 if out.exists() and (not out.is_dir() or any(out.iterdir())):raise ValueError('Output must be new or empty; existing runs are never overwritten')
 paths={'instruction':args.prompt,'system':args.system_prompt,'regions':args.region_prompt}
 prompts={};prompt_hashes={}
 for key,path in paths.items():
  raw=path.read_bytes();text=raw.decode('utf-8-sig')
  if not text.strip():raise ValueError('Empty prompt: '+str(path))
  prompts[key]=text;prompt_hashes[str(path.resolve())]=sha(path)
 originals=discover(args.input,not args.no_recursive)
 if not originals:raise ValueError('No supported images in input directory')
 names=identifiers(originals,args.input)
 missing=ensure_models(args.models,args.stage,check=True,profile=args.profile)
 print(f'PREFLIGHT: {len(originals)} images; recursive={not args.no_recursive}; family={args.family}; stage={args.stage}',flush=True)
 print('OUTPUT: '+str(out),flush=True)
 if missing:print('Models to download: '+', '.join(missing),flush=True)
 if args.family=='wan':print('WAN: dataset export only; OneTrainer does not support this architecture.',flush=True)
 args.device,args.dtype=resolve_device(args.device,args.dtype)
 print(f'COMPUTE: {args.device}; dtype={args.dtype}; profile={args.profile}',flush=True)
 if args.check:return 0
 ensure_models(args.models,args.stage,profile=args.profile)
 out.mkdir(parents=True,exist_ok=True)
 for folder in ('images','captions','regions','errors','prompts','previews','grounding_raw'): (out/folder).mkdir()
 manifest=[];files=[];records={};run_error=None
 for key,path in paths.items():shutil.copy2(path,out/'prompts'/(key+'.txt'))
 try:
  for original in originals:
   digest=sha(original);dest=out/'images'/names[original];shutil.copy2(original,dest)
   if sha(dest)!=digest:raise IOError('Source changed during copy: '+str(original))
   row={'file':original.relative_to(args.input).as_posix(),'source':str(original),'output_file':dest.name,'sha256':digest}
   manifest.append(row);files.append(dest)
   records[dest.name]={'file':dest.name,'source_file':row['file'],'caption':'','characters':[],'qa_flags':[]}
  save(out/'source_manifest.json',manifest)
  save(out/'run_config.json',dict(source=str(args.input),output=str(out),prompt_files=prompt_hashes,
   options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
   caption_policy='Neutral only; accepted text saved exactly. Rejected responses are errors, not rewritten captions.'))
  run_models(out,files,records,args,prompts)
 except Exception as e:
  run_error=str(e);save(out/'errors/run.json',{'error':run_error});print('RUN ERROR: '+run_error,file=sys.stderr)
 finally:
  def changed(path,digest):
   try:return sha(Path(path))!=digest
   except OSError:return True
  changed_sources=[r['file'] for r in manifest if changed(r['source'],r['sha256'])]
  changed_prompts=[p for p,h in prompt_hashes.items() if changed(p,h)]
  summary={'images':len(originals),'copied_images':len(files),'captions':len(list((out/'captions').glob('*.txt'))),
   'mask_files':len(list((out/'masks').glob('**/*.png'))),'changed_sources':changed_sources,'changed_prompt_files':changed_prompts}
  if args.stage!='regions' and not changed_sources and not changed_prompts:
   try:summary['training_export']=export_training(out,files,args)
   except Exception as e:save(out/'errors/export.json',{'error':str(e)})
  cards=[]
  for f in files:
   d=records[f.name];save(out/'regions'/(f.stem+'.json'),d)
   preview=out/'previews'/(f.stem+'.jpg');src=preview.relative_to(out).as_posix() if preview.exists() else f.relative_to(out).as_posix()
   cards.append('<article><h2>'+html.escape(d['source_file'])+'</h2><img loading="lazy" src="'+html.escape(src,quote=True)+'"><p>'+html.escape(d['caption'])+'</p><pre>'+html.escape(json.dumps(d,ensure_ascii=False,indent=2))+'</pre></article>')
  (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>auto_captioning_tool</title><style>body{font:16px system-ui;background:#181b21;color:#eee;margin:24px}article{border:1px solid #555;padding:20px;margin:20px 0}img{max-width:100%;max-height:720px}pre{white-space:pre-wrap}</style><h1>auto_captioning_tool</h1><p>Automatic annotations. Review masks and captions. Accepted captions are unchanged model text.</p>'+''.join(cards),encoding='utf-8')
  summary['errors']=len(list((out/'errors').glob('*.json')));save(out/'validation.json',summary);save(out/'review_data.json',list(records.values()))
  print(json.dumps(summary),flush=True);print('Review: '+str(out/'review.html'),flush=True)
 return 1 if summary['errors'] or changed_sources or changed_prompts else 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:print('ERROR: '+str(e),file=sys.stderr);sys.exit(1)
