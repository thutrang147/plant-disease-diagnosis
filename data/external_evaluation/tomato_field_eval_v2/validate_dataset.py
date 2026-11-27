"""Validate a portable v2 package. Requires Python 3 and Pillow; no model libraries."""
import argparse,csv,hashlib,json,zipfile,xml.etree.ElementTree as ET
from pathlib import Path,PurePosixPath
from collections import Counter
from PIL import Image,ImageOps
CLASSES={'Healthy':0,'Early Blight':1,'Late Blight':2,'Bacterial Spot':3,'Septoria Leaf Spot':4}
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_csv(path):
 with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read_workbook(path):
 """Read literal cell values, resolving worksheet relationships; ignore summary caches."""
 out={}
 with zipfile.ZipFile(path) as z:
  strings=[''.join(t.text or '' for t in e.findall('.//s:t',NS)) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS)] if 'xl/sharedStrings.xml' in z.namelist() else []
  rels={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
  for sh in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',NS):
   target=rels[sh.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')];target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
   rows=[];headers=None
   for row in ET.fromstring(z.read(target)).findall('s:sheetData/s:row',NS):
    cells={};formulas=[]
    for c in row.findall('s:c',NS):
     letter=''.join(filter(str.isalpha,c.get('r')));v=c.find('s:v',NS);v=v.text if v is not None else ''
     if c.get('t')=='s':v=strings[int(v)]
     elif c.get('t')=='inlineStr':v=''.join(t.text or '' for t in c.findall('.//s:t',NS))
     cells[letter]=v
     if c.find('s:f',NS) is not None:formulas.append(letter)
    if headers is None:headers={k:v.strip() for k,v in cells.items()};continue
    d={headers[k]:v for k,v in cells.items() if headers.get(k)}
    if not any(str(v).strip() for v in d.values()):continue
    d['_excel_row']=int(row.get('r'));d['_formula_headers']=[headers[k] for k in formulas if headers.get(k)];rows.append(d)
   out[sh.get('name')]=rows
 return out

def eligible(a):return a.get('source','').strip().lower()=='self_collected' and a.get('class_name','').strip().lower()=='healthy'
def active_holds(decision,crop):
 reasons=[]
 if decision.get('Preparation status')!='ready for human decision':reasons.append('unresolved technical discussion hold')
 if crop['technical_status']!='technically_usable':reasons.append('unresolved technical status')
 if crop['label_review_status']!='no_documented_label_conflict':reasons.append('unresolved annotation conflict')
 if crop['duplicate_group_id']:reasons.append('unresolved duplicate group')
 if crop['overlaps_v1']=='True' or crop['previous_crop_reference_ids'] or crop['similarity_review_reference_ids']:reasons.append('unresolved historical overlap/similarity')
 return reasons

def validate(root):
 root=Path(root).resolve();m=read_csv(root/'manifest.csv');config=json.loads((root/'provenance/build_config.json').read_text());checks=[]
 def check(ok,name):
  if not ok:raise AssertionError(name)
  checks.append(name)
 def local(rel):
  p=PurePosixPath(rel);check(bool(rel) and not p.is_absolute() and '..' not in p.parts and '\\' not in rel,'portable path: '+rel)
  path=root/rel;check(path.is_file(),'file exists: '+rel);return path
 decisions=read_workbook(root/'provenance/inputs/final_review_workbook.xlsx')['New candidates'];review={r['Crop ID']:r for r in decisions}
 annotation=read_workbook(root/'provenance/inputs/Tomato Field Photo Review.xlsx')['review'];ann={r['filename']:r for r in annotation if r.get('filename')}
 cropmap={r['crop_id']:r for r in read_csv(root/'provenance/inputs/crop_mapping.csv')};v1={r['image_id']:r for r in read_csv(root/'provenance/inputs/v1_manifest.csv')}
 for rel,h in config['input_sha256'].items():check(sha(local(rel))==h,'input hash: '+rel)
 check(len({r['image_id'] for r in m})==len(m),'unique image IDs');check(len({r['image_path'] for r in m})==len(m),'unique image paths');check(len({r['sha256'] for r in m})==len(m),'no duplicate evaluation file bytes')
 check({r['image_id'] for r in m if r['source']=='online_sources'}=={k for k,v in v1.items() if v['source']=='online_sources'},'all 46 original online records retained')
 check(sum(r['source']=='online_sources' for r in m)==46,'46 online disease images')
 check(all(r.get('Review decision') in {'accept','reject','pending','fix_crop'} for r in decisions),'known literal human decisions')
 newexpected={r['Crop ID'] for r in decisions if r['Review decision']=='accept' and not active_holds(r,cropmap[r['Crop ID']])}
 check({r['image_id'] for r in m if r['approval_basis']=='explicit_human_accept'}==newexpected,'all and only eligible new accepts included')
 originals={k for k,v in v1.items() if v['source']=='self_collected' and v['image_type']!='cropped'}
 check({r['image_id'] for r in m if r['evaluation_subset']=='healthy_field_originals'}==originals,'complete original Healthy subset')
 historical=set()
 for k,v in v1.items():
  if v['source']!='self_collected' or v['image_type']!='cropped':continue
  exact=[c for c in cropmap.values() if c['crop_sha256']==v['sha256']]
  held=any(c['technical_status']!='technically_usable' or c['label_review_status']!='no_documented_label_conflict' or c['duplicate_group_id'] for c in exact)
  if config['historical_policy']=='retain-keep' and not held:historical.add(k)
 check({r['image_id'] for r in m if r['evaluation_subset']=='healthy_leaf_crops' and r['approval_basis']=='historical_v1_keep'}==historical,'complete hold-free historical crop subset')
 leafgroups=[];cropchecks=0;v1checks=0
 for r in m:
  path=local(r['image_path']);check(sha(path)==r['sha256'],'image SHA-256: '+r['image_id'])
  with Image.open(path) as im:im.load();check(ImageOps.exif_transpose(im).size==(int(r['width']),int(r['height'])),'image dimensions: '+r['image_id'])
  check(CLASSES[r['class_name']]==int(r['label_id']),'canonical class: '+r['image_id'])
  if r['source']=='self_collected':
   check(r['class_name']=='Healthy' and eligible(ann[r['original_filename']]),'human Healthy source: '+r['image_id']);check(r['parent_image_id']==Path(r['original_filename']).stem and r['scene_group_id']=='self::'+r['plant_id'],'parent/scene link: '+r['image_id']);check(r['plant_id']==ann[r['original_filename']]['plant_id'],'plant provenance: '+r['image_id'])
   check(r['evaluation_subset']==('healthy_leaf_crops' if r['is_crop']=='True' else 'healthy_field_originals'),'Healthy subset: '+r['image_id'])
   check(r['label_source']=='human_annotated_healthy','label authority: '+r['image_id'])
   if r['is_crop']=='True':
    leafgroups.append(r['leaf_group_id']);check(r['technical_hold']=='False' and r['duplicate_hold']=='False' and r['annotation_hold']=='False','no unresolved crop hold: '+r['image_id'])
    if r['approval_basis']=='explicit_human_accept':
     d=review[r['image_id']];c=cropmap[r['image_id']];check(d['Review decision']=='accept' and not active_holds(d,c),'explicit human crop accept: '+r['image_id']);check(r['sha256']==c['crop_sha256'] and r['crop_coordinates']==c['crop_coordinates'] and r['leaf_group_id']==c['leaf_group_id'],'accepted crop mapping: '+r['image_id'])
    else:check(config['historical_policy']=='retain-keep' and r['approval_basis']=='historical_v1_keep' and v1[r['image_id']]['decision']=='keep','authorized historical approval: '+r['image_id'])
  else:check(r['source']=='online_sources' and r['evaluation_subset']=='online_disease_images' and r['class_name']!='Healthy','online subset: '+r['image_id'])
  if r['image_id'] in v1:
   old=v1[r['image_id']];check(r['sha256']==old['sha256'],'v1 bytes preserved: '+r['image_id']);v1checks+=1
   for key in ['source_url','original_image_url','source_group_id','verification_status','review_reason','leakage_check_status']:check(r[key]==old[key],'v1 attribution '+key+': '+r['image_id'])
  if r['is_crop']=='True':
   parent=local(r['parent_image_path']);check(sha(parent)==r['parent_sha256'],'parent hash: '+r['image_id']);p=ImageOps.exif_transpose(Image.open(parent)).convert('RGB');box=json.loads(r['crop_coordinates']);check(len(box)==4 and 0<=box[0]<box[2]<=p.width and 0<=box[1]<box[3]<=p.height,'valid source coordinates: '+r['image_id']);actual=Image.open(path).convert('RGB');expected=p.crop(box);check(actual.size==expected.size and actual.tobytes()==expected.tobytes(),'native crop pixels: '+r['image_id']);check(r['crop_sha256']==r['sha256'],'crop hash mapping: '+r['image_id']);cropchecks+=1
 check(len(set(leafgroups))==len(leafgroups),'unique recorded Healthy leaf groups')
 check(sum(r['evaluation_subset']=='healthy_field_originals' for r in m)==8,'eight historical Healthy originals retained')
 check({str(f.relative_to(root)) for f in (root/'images').rglob('*') if f.is_file()}=={r['image_path'] for r in m},'no orphan or missing evaluation images')
 # Explicit-accept rule applies to NEW Healthy crops. Historical keep approval is a separately authorized exception when selected.
 report={'status':'passed','images':len(m),'subsets':dict(Counter(r['evaluation_subset'] for r in m)),'class_counts':dict(Counter(r['class_name'] for r in m)),'new_human_decisions':dict(Counter(r['Review decision'] for r in decisions)),'new_crops_included':len(newexpected),'historical_policy':config['historical_policy'],'crop_coordinate_pixel_checks':cropchecks,'v1_retained_byte_checks':v1checks,'source_photographs':len({(r['source'],r['original_filename']) for r in m}),'self_collected_source_photographs':len({r['original_filename'] for r in m if r['source']=='self_collected'}),'self_collected_plant_ids':len({r['plant_id'] for r in m if r['source']=='self_collected' and r['plant_id']}),'self_collected_scene_groups':len({r['scene_group_id'] for r in m if r['source']=='self_collected' and r['scene_group_id']}),'validation_assertions':len(checks)}
 return report
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('dataset',nargs='?',default=str(Path(__file__).resolve().parent));args=ap.parse_args();print(json.dumps(validate(args.dataset),indent=2))
