"""Sequential local captioning, box grounding and SAM 2 segmentation."""
import gc,json,re
from pathlib import Path
from training_export import save,sha
REPOS={'caption':'fancyfeast/llama-joycaption-beta-one-hf-llava',
 'grounding':'Qwen/Qwen3-VL-8B-Instruct','sam':'facebook/sam2.1-hiera-large'}
REVISIONS={'caption':'ebf414ea497a020da0f82df3913e5b6cb8e9663a','grounding':'0c351dd01ed87e9c1b53cbc748cba10e6187ff3b','sam':'665f8e2ad61cf5f53d65644ff27c8ee525124610'}
# Neutral-only workflow. Rejected responses are not rewritten into accepted captions.
def parse_regions(raw):
 text=raw.strip();fenced=re.search(r'```(?:json)?\s*([\s\S]*?)```',text,re.I)
 if fenced:text=fenced.group(1).strip()
 start=re.search(r'[\[{]',text)
 if not start:raise ValueError('No grounding JSON')
 obj,end=json.JSONDecoder().raw_decode(text[start.start():])
 if text[start.start()+end:].strip():raise ValueError('Unexpected trailing grounding text')
 if isinstance(obj,list):items=obj
 elif isinstance(obj,dict) and 'characters' in obj:items=obj['characters']
 elif isinstance(obj,dict) and any(k in obj for k in ('bbox','bbox_2d','box')):items=[obj]
 else:raise ValueError('Expected characters array or boxes')
 if not isinstance(items,list):raise ValueError('characters must be an array')
 result=[]
 for entry in items:
  if not isinstance(entry,dict):raise ValueError('Region must be an object')
  c=dict(entry);b=c.get('bbox',c.get('bbox_2d',c.get('box')))
  if not isinstance(b,list) or len(b)!=4 or not all(type(x) in (int,float) and 0<=x<=1000 for x in b) or not(b[0]<b[2] and b[1]<b[3]):raise ValueError('Invalid normalized box')
  c['bbox']=b;result.append(c)
 return result

def model_complete(path):
 if not (path/'config.json').is_file():return False
 indices=list(path.glob('*.safetensors.index.json'))
 if indices:
  try:return all((path/name).is_file() and (path/name).stat().st_size>0 for index in indices for name in set(json.loads(index.read_text())['weight_map'].values()))
  except (ValueError,KeyError):return False
 return (path/'model.safetensors').is_file() and (path/'model.safetensors').stat().st_size>0

def ensure_models(cache,stage,check=False):
 required=['caption'] if stage=='captions' else ['grounding','sam'] if stage=='regions' else list(REPOS)
 missing=[]
 for key in required:
  repo=REPOS[key];path=cache/repo.split('/')[-1]
  # Marker indicates tokenizer/processor assets were included, not just model weights.
  complete=model_complete(path) and any(path.glob('*processor*.json')) and (key=='sam' or (path/'tokenizer.json').is_file())
  if not complete:
   missing.append(repo)
   if not check:
    from huggingface_hub import snapshot_download
    print('DOWNLOAD '+repo+' -> '+str(path),flush=True)
    snapshot_download(repo_id=repo,revision=REVISIONS[key],local_dir=str(path),ignore_patterns=['*.bin','*.pth','*.pt','*.onnx','*.msgpack','*.h5'],max_workers=4)
    if not model_complete(path):raise RuntimeError('Incomplete download: '+repo)
 return missing

def run_models(out,files,records,args,prompts):
 import torch
 import numpy as np
 from PIL import Image,ImageOps,ImageDraw
 from transformers import AutoProcessor,LlavaForConditionalGeneration,Qwen3VLForConditionalGeneration,Sam2Processor,Sam2Model
 if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable. Install a compatible NVIDIA driver and CUDA PyTorch.')
 print('GPU: '+torch.cuda.get_device_name(0),flush=True)
 def load_image(f):
  with Image.open(f) as source:
   if getattr(source,'n_frames',1)>1:raise ValueError('Multi-frame image is unsupported; export still frames first')
   return ImageOps.exif_transpose(source).convert('RGB')
 valid=[]
 for f in files:
  try:
   im=load_image(f);records[f.name].update(width=im.width,height=im.height);valid.append(f)
  except Exception as e:
   records[f.name]['qa_flags'].append(str(e));save(out/'errors'/(f.stem+'_image.json'),{'error':str(e)})
 def failed(f,stage,e):
  records[f.name]['qa_flags'].append(stage+' failed: '+str(e));save(out/'errors'/(f.stem+'_'+stage+'.json'),{'error':str(e)})
 def store(f):save(out/'regions'/(f.stem+'.json'),records[f.name])
 if args.stage!='captions':
  path=args.models/REPOS['grounding'].split('/')[-1]
  processor=AutoProcessor.from_pretrained(path,local_files_only=True)
  model=Qwen3VLForConditionalGeneration.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
  inputs=ids=None
  for i,f in enumerate(valid,1):
   d=records[f.name]
   try:
    im=load_image(f);im.thumbnail((1024,1024))
    messages=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':prompts['regions']}]}]
    inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
    with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=1000,do_sample=False)
    raw=processor.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    (out/'grounding_raw'/(f.stem+'.txt')).write_bytes(raw.encode('utf-8'))
    chars=parse_regions(raw)
    for j,c in enumerate(chars,1):
     b=c['bbox'];c['id']=f'person_{j}';c['bbox_xyxy_pixels']=[round(b[k]*d['width' if k%2==0 else 'height']/1000) for k in range(4)]
    d.update(characters=chars,grounding_model=REPOS['grounding'])
    print(f'REGIONS {i}/{len(valid)} {f.name}: {len(chars)}',flush=True)
   except Exception as e:failed(f,'regions',e)
   store(f)
  del model,processor,inputs,ids;gc.collect();torch.cuda.empty_cache()
  path=args.models/REPOS['sam'].split('/')[-1]
  processor=Sam2Processor.from_pretrained(path,local_files_only=True)
  model=Sam2Model.from_pretrained(path,local_files_only=True).to('cuda').eval()
  inputs=pred=all_masks=scores=None
  for i,f in enumerate(valid,1):
   d=records[f.name]
   try:
    im=load_image(f);masks=[];boxes=[c['bbox_xyxy_pixels'] for c in d['characters']]
    if boxes:
     inputs=processor(images=im,input_boxes=[boxes],return_tensors='pt').to('cuda')
     with torch.inference_mode():pred=model(**inputs,multimask_output=True)
     all_masks=processor.post_process_masks(pred.pred_masks.cpu(),inputs['original_sizes'])[0];scores=pred.iou_scores[0].detach().cpu()
     for j,c in enumerate(d['characters']):
      best=int(scores[j].argmax());mask=all_masks[j,best].numpy().astype(bool);masks.append(mask)
      dest=out/'masks'/f.stem/(c['id']+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
      Image.fromarray(mask.astype('uint8')*255).save(dest)
      c.update(mask=dest.relative_to(out).as_posix(),sam_predicted_iou=float(scores[j,best]),mask_area_fraction=float(mask.mean()))
      if c['sam_predicted_iou']<.75:d['qa_flags'].append(c['id']+': low SAM score')
      if mask.mean()<.005:d['qa_flags'].append(c['id']+': very small mask')
     for j in range(len(masks)):
      for k in range(j):
       overlap=np.logical_and(masks[j],masks[k]).sum()/max(1,min(masks[j].sum(),masks[k].sum()))
       if overlap>.25:d['qa_flags'].append(f'person_{k+1}/person_{j+1}: mask overlap {overlap:.2f}')
    else:d['qa_flags'].append('No characters detected')
    colors=[(255,80,70),(50,200,255),(100,240,100),(245,190,40),(220,90,240)];overlay=im.convert('RGBA')
    for j,mask in enumerate(masks):
     layer=Image.new('RGBA',im.size,colors[j%len(colors)]+(0,));layer.putalpha(Image.fromarray(mask.astype('uint8')*65));overlay=Image.alpha_composite(overlay,layer)
    draw=ImageDraw.Draw(overlay)
    for j,c in enumerate(d['characters']):
     box=c['bbox_xyxy_pixels'];draw.rectangle(box,outline=colors[j%len(colors)],width=max(2,im.width//350));draw.text((box[0]+3,box[1]+3),c['id'],fill='white',stroke_width=2,stroke_fill='black')
    overlay.thumbnail((1200,1200));overlay.convert('RGB').save(out/'previews'/(f.stem+'.jpg'),quality=88)
    d.update(segmentation_model=REPOS['sam'],segmentation_complete=True)
    print(f'MASK {i}/{len(valid)} {f.name} flags={len(d["qa_flags"])}',flush=True)
   except Exception as e:failed(f,'sam',e)
   store(f)
  del model,processor,inputs,pred,all_masks,scores;gc.collect();torch.cuda.empty_cache()
 if args.stage!='regions':
  path=args.models/REPOS['caption'].split('/')[-1];processor=AutoProcessor.from_pretrained(path,local_files_only=True)
  model=LlavaForConditionalGeneration.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
  inputs=ids=None
  for i,f in enumerate(valid,1):
   try:
    im=load_image(f);conversation=[{'role':'system','content':prompts['system']},{'role':'user','content':prompts['instruction']}]
    text=processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=[im],return_tensors='pt').to('cuda');inputs['pixel_values']=inputs['pixel_values'].to(torch.bfloat16)
    with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=args.caption_tokens,do_sample=False)
    caption=processor.tokenizer.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    if not caption.strip():raise ValueError('Empty caption')
    dest=out/'captions'/(f.stem+'.txt');dest.write_bytes(caption.encode('utf-8'))
    records[f.name].update(caption=caption,caption_model=REPOS['caption'],caption_precision='BF16',caption_sha256=sha(dest))
    print(f'CAPTION {i}/{len(valid)} {f.name}',flush=True)
   except Exception as e:failed(f,'caption',e)
   store(f)
  del model,processor,inputs,ids;gc.collect();torch.cuda.empty_cache()
