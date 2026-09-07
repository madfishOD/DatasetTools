"""No model downloads or inference. Test filesystem/export invariants."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from auto_captioning_tool import discover,identifiers
from training_export import export_training,training_plan,rebase
from engine import parse_regions,model_complete
class Checks(unittest.TestCase):
 def test_discovery_and_collisions(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'nested').mkdir()
   for rel in ['same.png','same.jpg','nested/same.png','UPPER.PNG','ignore.txt']:(root/rel).write_bytes(b'test')
   files=discover(root);self.assertEqual(len(files),4);self.assertEqual(len(discover(root,False)),3)
   names=identifiers(files,root);self.assertEqual(len(set(n.casefold() for n in names.values())),4)
 def test_regions(self):
  self.assertEqual(len(parse_regions('[{"bbox_2d":[0,0,500,500]},{"box":[500,0,1000,900]}]')),2)
  with self.assertRaises(ValueError):parse_regions('{"characters":[{"bbox":[0,0,0,900]}]}')
 def test_size_schedule(self):
  for n in (0,1,6,117,10000):
   p=training_plan(n,'qwen');self.assertEqual(p['estimated_optimizer_steps'],n*p['epochs'])
   if n:self.assertGreaterEqual(p['estimated_optimizer_steps'],p['target_optimizer_steps'])
  self.assertEqual(training_plan(117,'qwen')['epochs'],20)
  self.assertEqual(training_plan(6,'qwen',epochs=3)['estimated_optimizer_steps'],18)
 def test_export_bytes_and_architectures(self):
  with tempfile.TemporaryDirectory() as temp:
   parent=Path(temp)
   for family in ('qwen','sdxl','flux','wan'):
    out=parent/family;out.mkdir();(out/'captions').mkdir()
    a=out/'a.png';b=out/'b.png';a.write_bytes(b'image-bytes');b.write_bytes(b'missing-caption')
    caption=b'  Original caption.\r\n';(out/'captions/a.txt').write_bytes(caption)
    args=SimpleNamespace(family=family,target_steps=100,epochs=None,base_model='custom/model',resolution=None,rank=16,learning_rate=.0001)
    result=export_training(out,[a,b],args);self.assertEqual(result['accepted_pairs'],1)
    self.assertEqual((out/'onetrainer/data/a.txt').read_bytes(),caption)
    self.assertEqual((out/'onetrainer/data/a.png').read_bytes(),a.read_bytes())
    self.assertFalse((out/'onetrainer/data/b.png').exists())
    self.assertEqual((out/'onetrainer/train.json').exists(),family!='wan')
    if family!='wan':
     config=json.loads((out/'onetrainer/train.json').read_text())
     self.assertFalse(config['masked_training']);self.assertEqual(config['base_model_name'],'custom/model')
     rebase(out)
 def test_empty_export(self):
  with tempfile.TemporaryDirectory() as temp:
   args=SimpleNamespace(family='qwen',target_steps=None,epochs=None)
   result=export_training(Path(temp),[],args);self.assertFalse(result['ready'])
   self.assertFalse((Path(temp)/'onetrainer/train.json').exists())
 def test_partial_model(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);(p/'config.json').write_text('{}')
   (p/'model.safetensors.index.json').write_text('{"weight_map":{"a":"part1.safetensors","b":"part2.safetensors"}}')
   (p/'part1.safetensors').write_bytes(b'a');self.assertFalse(model_complete(p))
   (p/'part2.safetensors').write_bytes(b'b');self.assertTrue(model_complete(p))
if __name__=='__main__':unittest.main()
