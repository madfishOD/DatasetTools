"""Portable neutral captioning + regions + byte-preserving training dataset export."""
import argparse,datetime,hashlib,html,json,os,re,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from training_export import PROFILES,rebase,save,sha,atomic_write
import project as projects
from engine import ensure_models,run_models
from devices import resolve_device
from model_catalog import CATALOG, selections
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
 p.add_argument('--resume',type=Path,help='Open a versioned project; input is optional after import')
 p.add_argument('--import-only',action='store_true',help='Import images and TXT without inference')
 p.add_argument('--export-only',action='store_true',help='Export approved current captions without inference')
 p.add_argument('--caption-policy',choices=['preserve','fill-missing','regenerate'],default='fill-missing')
 p.add_argument('--select',nargs='+',help='Explicit sample IDs or source filenames for selective work')
 p.add_argument('--approve-all',action='store_true',help='Approve all current complete captions for export')
 p.add_argument('--events',action='store_true',help=argparse.SUPPRESS)
 p.add_argument('--stop-file',type=Path,help=argparse.SUPPRESS)
 p.add_argument('--prompt',type=Path,default=HERE/'prompts/neutral/instruction.txt')
 p.add_argument('--system-prompt',type=Path,default=HERE/'prompts/neutral/system.txt')
 p.add_argument('--region-prompt',type=Path,default=HERE/'prompts/neutral/regions.txt')
 p.add_argument('--models',type=Path,default=HERE/'models',help='Captioning model cache; downloads missing models automatically')
 p.add_argument('--packages',type=Path,help='Optional existing Python package directory')
 p.add_argument('--family',choices=PROFILES,default='qwen',help='Training architecture, independent of captioning model')
 p.add_argument('--base-model',default='',help='Training model repository ID or local path')
 p.add_argument('--stage',choices=['all','captions','regions'],default='all')
 p.add_argument('--profile',choices=['quality','compact'],default='quality')
 for stage in ('caption', 'grounding', 'sam'):
  p.add_argument('--'+stage+'-model', choices=[key for key,spec in CATALOG.items() if stage in spec['stages']], help='Override this stage model from the profile')
 p.add_argument('--training-goal',choices=['unspecified','style','character','concept','pose','custom'],default='unspecified')
 p.add_argument('--goal-description',default='')
 p.add_argument('--training-family',choices=['unknown','sdxl','flux','qwen','wan','other'],default='unknown')
 p.add_argument('--training-method',choices=['unknown','lora','full'],default='unknown')
 p.add_argument('--trainer',default='')
 p.add_argument('--training-hardware',default='')
 p.add_argument('--trigger-word',default='')
 p.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='auto')
 p.add_argument('--dtype',choices=['auto','float16','bfloat16','float32'],default='auto')
 p.add_argument('--max-image-side',type=positive,default=None,help='Caption input longest side (Compact default: 768); original files are unchanged')
 p.add_argument('--no-training-config',action='store_true',help='Export image/TXT pairs without OneTrainer training config')
 p.add_argument('--no-recursive',action='store_true',help='Only images directly in the input folder')
 p.add_argument('--target-steps',type=positive);p.add_argument('--epochs',type=positive)
 p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--rank',type=positive,default=16)
 p.add_argument('--resolution',type=positive);p.add_argument('--caption-tokens',type=positive,default=320)
 p.add_argument('--region-tokens',type=positive,default=1000,help='Maximum generated tokens for region JSON; increase for crowded scenes')
 p.add_argument('--interactive',action='store_true');p.add_argument('--check',action='store_true',help='Read-only preflight, no downloads or inference')
 p.add_argument('--download-models',action='store_true',help='Download captioning models only; no input required')
 p.add_argument('--rebase-output',type=Path,help='Update generated OneTrainer paths after moving an output folder')
 return p

PERSISTED_OPTIONS = ('stage','profile','device','dtype','max_image_side','caption_tokens','region_tokens',
                     'family','base_model','resolution','rank','learning_rate','epochs','target_steps',
                     'no_training_config','models','no_recursive','caption_model','grounding_model','sam_model',
                     'training_goal','goal_description','training_family','training_method','trainer','training_hardware','trigger_word')
PROMPT_OPTIONS = {'instruction':'prompt','system':'system_prompt','regions':'region_prompt'}


def read_prompts(args, root=None, project=None, explicit=()):
    prompts, raw_prompts = {}, {}
    for key, option in PROMPT_OPTIONS.items():
        if project and option not in explicit and key in project['prompts']:
            info = project['prompts'][key]
            path = projects.local(root, info['path'])
            if sha(path) != info['sha256']:
                raise ValueError('Saved prompt modified; pass a new prompt file explicitly.')
        else:
            path = getattr(args, option)
        raw = path.read_bytes()
        text = raw.decode('utf-8-sig')
        if not text.strip():
            raise ValueError('Empty prompt: ' + str(path))
        prompts[key], raw_prompts[key] = text, raw
    return prompts, raw_prompts


def write_review(out, records, summary):
    cards = []
    for record in records.values():
        image = out / 'images' / record['file']
        preview = out / 'previews' / (image.stem + '.jpg')
        source = preview if preview.exists() else image
        cards.append('<article><h2>' + html.escape(record['source_file']) + '</h2><img loading="lazy" src="' +
                     html.escape(source.relative_to(out).as_posix(), quote=True) + '"><p>' +
                     html.escape(record.get('caption','')) + '</p><pre>' +
                     html.escape(json.dumps(record, ensure_ascii=False, indent=2)) + '</pre></article>')
    page = '<!doctype html><meta charset="utf-8"><title>Dataset project</title><style>body{font:16px system-ui;background:#181b21;color:#eee;margin:24px}article{border:1px solid #555;padding:20px;margin:20px 0}img{max-width:100%;max-height:720px}pre{white-space:pre-wrap}</style><h1>Dataset project</h1><p>Editable captions are in captions/. Only approved complete captions are exported.</p>' + ''.join(cards)
    atomic_write(out / 'review.html', page.encode('utf-8'))
    save(out / 'review_data.json', list(records.values()))
    save(out / 'validation.json', summary)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = parser()
    args = cli.parse_args(argv)
    def event(kind, **values):
        if args.events:
            print('DATASET_EVENT '+json.dumps({'event':kind, **values}), flush=True)
    def stop_requested():
        if args.stop_file and args.stop_file.exists():
            raise KeyboardInterrupt
    option_names = {option: action.dest for action in cli._actions for option in action.option_strings}
    explicit = {option_names[token.split('=')[0]] for token in argv if token.split('=')[0] in option_names}
    if args.rebase_output:
        rebase(args.rebase_output); return 0
    if args.import_only and args.export_only:
        raise ValueError('Choose either --import-only or --export-only')
    if args.export_only and not args.resume:
        raise ValueError('--export-only requires --resume')
    if args.interactive and not args.resume:
        for name in ('input','output','prompt','system_prompt','region_prompt'):
            current = getattr(args,name)
            answer = input(f'{name} [{current or "required"}] (Enter = default): ').strip().strip('"')
            if answer:
                setattr(args,name,Path(answer)); explicit.add(name)
    if args.resume:
        out = args.resume.resolve()
        if args.output and args.output.resolve() != out:
            raise ValueError('--output must match --resume, or be omitted')
        project = projects.load_project(out)
        for key,value in project['options'].items():
            if key in PERSISTED_OPTIONS and key not in explicit:
                setattr(args,key,Path(value) if key == 'models' else value)
    else:
        out = (args.output or HERE/'outputs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
        project = None
        if not args.download_models and out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise ValueError('Output must be new or empty. Use --resume for an existing project.')
    if args.packages:
        sys.path.insert(0,str(args.packages.resolve()))
    args.models = args.models.resolve()
    if args.profile == 'compact' and args.max_image_side is None:
        args.max_image_side = 768
    if args.download_models:
        missing = ensure_models(args.models,args.stage,check=args.check,profile=args.profile,overrides=selections(args))
        print(json.dumps({'missing_models':missing,'check_only':args.check})); return 0
    if not 0 < args.learning_rate < 1:
        raise ValueError('Learning rate must be between 0 and 1')
    if not project and args.input is None:
        raise ValueError('--input is required for a new project')
    originals = None
    if args.input:
        args.input = args.input.resolve()
        if out == args.input or args.input in out.parents or out in args.input.parents:
            raise ValueError('Input and project must be separate non-nested directories')
        originals = discover(args.input, not args.no_recursive)
        if not originals:
            raise ValueError('No supported images in input directory')
    prompts, raw_prompts = read_prompts(args,out,project,explicit)
    if args.check:
        if project:
            records = projects.load_records(out,project)
            projects.sync_edits(out,records,write=False)
            work = projects.plan_work(out,records,args,prompts)
            print(json.dumps({'samples':[{'sample_id':r['sample_id'],'source_file':r['source_file'],
                              'review_status':r['review_status'],'stages':r['stages']} for r in records.values()],
                              'pending':{key:len(value) for key,value in work.items()}},indent=2))
        else:
            print(json.dumps({'images':len(originals),'output':str(out),'check_only':True}))
        return 0
    with projects.project_lock(out):
        # Reload under lock in case another process committed after the read-only setup above.
        if project:
            project = projects.load_project(out)
        else:
            if (out/'project.json').exists():
                raise ValueError('Project was created by another process; use --resume.')
            project = projects.create_project(out)
        for key,raw in raw_prompts.items():
            checksum = hashlib.sha256(raw).hexdigest()
            relative = f'prompts/{key}-{checksum}.txt'
            atomic_write(out/relative,raw)
            project['prompts'][key] = {'path':relative,'sha256':checksum}
        project['options'] = {key:str(getattr(args,key)) if isinstance(getattr(args,key),Path) else getattr(args,key) for key in PERSISTED_OPTIONS}
        save(out/'project.json',project)
        if originals is not None:
            projects.import_sources(out,project,args.input,originals)
        else:
            source = Path(project['source_root']) if project.get('source_root') else None
            projects.finish_import(out,project,source if source and source.is_dir() else None)
        records = projects.load_records(out,project)
        projects.sync_edits(out,records)
        work = projects.plan_work(out,records,args,prompts) if not args.export_only else {s:[] for s in ('grounding','sam','caption')}
        for record in records.values():
            save(out/'regions'/(Path(record['file']).stem+'.json'),record)
        pending = {key:len(value) for key,value in work.items()}
        print('PENDING: '+json.dumps(pending),flush=True)
        event('plan', pending=pending)
        interrupted = False
        run_error = None
        try:
            stop_requested()
            if any(pending.values()) and not args.import_only and not args.export_only:
                device, precision = resolve_device(args.device,args.dtype)
                print(f'COMPUTE: {device}; dtype={precision}; profile={args.profile}',flush=True)
                ensure_models(args.models,args.stage,profile=args.profile,required_keys=[key for key in work if work[key]],overrides=selections(args))
                files = list(dict.fromkeys(f for values in work.values() for f in values))
                def checkpoint(image, stage, status, error=None):
                    if status == 'running':
                        stop_requested()
                    projects.checkpoint(out,records,args,prompts,image,stage,status,error)
                    event('stage', sample_id=records[image.name]['sample_id'], stage=stage, status=status)
                    if status in ('complete', 'failed'):
                        stop_requested()
                run_models(out,files,records,args,prompts,work=work,
                           checkpoint_fn=checkpoint)
        except KeyboardInterrupt:
            interrupted = True
            print('Interrupted. Completed checkpoints are saved; use --resume.',flush=True)
        except Exception as error:
            run_error = str(error)
            print('RUN ERROR: '+run_error,file=sys.stderr)
        # Verify private image copies again before any export; original source is no longer required.
        projects.load_records(out,project)
        if args.approve_all and not interrupted and not run_error:
            for record in records.values():
                if projects.artifact_valid(out,record,'caption'):
                    record['review_status'] = 'approved'
                    save(out/'regions'/(Path(record['file']).stem+'.json'),record)
        summary = {'images':len(records),'pending_at_start':pending,'interrupted':interrupted,'run_error':run_error,
                   'captions':sum(projects.artifact_valid(out,r,'caption') for r in records.values()),
                   'errors':sum(state.get('status')=='failed' for r in records.values() for state in r['stages'].values())}
        if not interrupted and not run_error:
            try:
                summary['export'] = projects.export_snapshot(out,project,records,args)
            except Exception as error:
                summary['export_error'] = str(error)
        write_review(out,records,summary)
        print(json.dumps(summary),flush=True)
        event('result', summary=summary)
        print('Review: '+str(out/'review.html'),flush=True)
        return 130 if interrupted else int(bool(run_error or summary['errors'] or summary.get('export_error')))


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:print('ERROR: '+str(error),file=sys.stderr);sys.exit(1)
