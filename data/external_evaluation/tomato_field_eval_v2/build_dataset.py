"""Rebuild v2 from immutable reviewed inputs; refuses to overwrite any dataset."""
import argparse,csv,hashlib,json,shutil,sys,subprocess
from pathlib import Path
from collections import Counter
from PIL import Image,ImageOps
from validate_dataset import sha,read_csv,read_workbook,eligible,active_holds,validate,CLASSES

def write_csv(path,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with open(path,'w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def build(raw,v1,mapping_path,dest,historical_policy,repo):
 raw=Path(raw).resolve();v1=Path(v1).resolve();dest=Path(dest).resolve();repo=Path(repo).resolve();reviewroot=raw/'healthy_leaf_crops_review_v3'
 assert not dest.exists(),'Refusing to overwrite existing dataset: '+str(dest)
 workbook=reviewroot/'final_review_workbook.xlsx';annotations=raw/'Tomato Field Photo Review.xlsx'
 protected={p:sha(p) for p in v1.rglob('*') if p.is_file() and p.name!='.DS_Store'}
 protected.update({p:sha(p) for p in reviewroot.rglob('*') if p.is_file() and p.name!='.DS_Store'});protected[annotations]=sha(annotations)
 untouched={p:(p.stat().st_size,p.stat().st_mtime_ns) for d in [repo/'notebooks',repo/'models'] if d.exists() for p in d.rglob('*') if p.is_file()}
 sheets=read_workbook(workbook);decisions=sheets['New candidates'];annotation=read_workbook(annotations)['review'];ann={r['filename']:r for r in annotation if r.get('filename')};crops=read_csv(reviewroot/'crop_mapping.csv');cropmap={r['crop_id']:r for r in crops}
 assert len(decisions)==65 and len({r['Crop ID'] for r in decisions})==len(decisions)
 for d in decisions:
  assert d['Review decision'] in {'accept','reject','pending','fix_crop'},'Unexpected literal decision: '+repr(d['Review decision'])
  assert 'Review decision' not in d['_formula_headers'],'A review decision must be a saved literal, not a cached formula'
  c=cropmap[d['Crop ID']];a=ann[c['original_filename']]
  assert d['Original filename']==c['original_filename'] and d['Original Excel label']==a['class_name'] and d['Original Excel decision']==a['decision']
  assert eligible(a) and c['class_name']==a['class_name'] and c['source']==a['source'] and c['plant_id']==a['plant_id']
  assert c['original_note']==a['note'] and c['original_decision']==a['decision'] and c['workbook_sha256']==sha(annotations)
 # Inspect approval cells directly; cached summary formulas may be stale after manual edits.
 current_counts=Counter(d['Review decision'] for d in decisions)
 v1records=read_csv(v1/'manifest.csv');vm={r['image_id']:r for r in v1records};vmap={r['image_id']:r for r in read_csv(mapping_path)}
 historical=[r for r in v1records if r['source']=='self_collected' and r['image_type']=='cropped']
 originals=[r for r in v1records if r['source']=='self_collected' and r['image_type']!='cropped'];online=[r for r in v1records if r['source']=='online_sources']
 assert len(historical)==8 and len(originals)==8 and len(online)==46
 dest.mkdir(parents=True);(dest/'provenance/inputs').mkdir(parents=True);(dest/'provenance/parents').mkdir()
 inputs={workbook:'final_review_workbook.xlsx',annotations:'Tomato Field Photo Review.xlsx',v1/'manifest.csv':'v1_manifest.csv',Path(mapping_path):'v1_image_mapping.csv',reviewroot/'crop_mapping.csv':'crop_mapping.csv',reviewroot/'duplicate_groups.csv':'duplicate_groups.csv',reviewroot/'manual_review_queue.csv':'manual_review_queue.csv'}
 input_hashes={}
 for source,name in inputs.items():
  target=dest/'provenance/inputs'/name;shutil.copy2(source,target);input_hashes[str(target.relative_to(dest))]=sha(target)
 records=[];dispositions=[];held_legacy=[]
 # Eight existing original photographs remain their original subset; crop children never become independent plants.
 original_paths={r['original_filename']:'images/healthy_field_originals/'+r['filename'] for r in originals}
 def parent_path(source,filename):
  src=raw/('self_collected' if source=='self_collected' else 'online_sources')/filename
  assert src.is_file(),'Missing raw parent: '+str(src)
  if source=='self_collected' and filename in original_paths:
   rel=original_paths[filename]
   assert sha(src)==vm[next(r['image_id'] for r in originals if r['original_filename']==filename)]['sha256']
  else:
   rel='provenance/parents/'+source+'/'+filename;target=dest/rel
   if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
  return rel,sha(src)
 def copy_v1(r):
  m=vmap[r['image_id']];is_crop=r['image_type']=='cropped';is_self=r['source']=='self_collected';subset='healthy_leaf_crops' if is_crop and is_self else ('healthy_field_originals' if is_self else 'online_disease_images')
  rel='images/'+subset+'/'+(('' if is_self else r['class_name'].replace(' ','_')+'/'))+r['filename'];target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(v1/r['image_path'],target)
  assert sha(target)==r['sha256'];a=ann.get(r['original_filename'],{});row=dict(r)
  row.update(image_path=rel,evaluation_subset=subset,original_filename=r['original_filename'],parent_image_id=Path(r['original_filename']).stem,scene_group_id=('self::'+r['plant_id']) if is_self else '',leaf_group_id=('self::'+r['plant_id']+'::v1_'+r['image_id']) if is_self and is_crop else '',is_crop=str(is_crop),crop_coordinates='',crop_sha256=r['sha256'] if is_crop else '',parent_sha256=m['original_sha256'],parent_image_path=rel if is_self and not is_crop else '',review_status='historical_v1_kept',annotation_source='provenance/inputs/'+('Tomato Field Photo Review.xlsx#review!row='+str(a['_excel_row']) if is_self else 'v1_manifest.csv#image_id='+r['image_id']),label_source='human_annotated_healthy' if is_self else 'historical_online_source_label',approval_basis='historical_v1_keep' if is_self else 'historical_v1_online_selection',human_review_decision=r['decision'],human_reviewer='',human_review_date='',human_review_notes='',original_annotation_decision=a.get('decision',''),original_annotation_reviewer=a.get('reviewer',''),original_annotation_note=a.get('note',''),annotation_source_ref=a.get('source_ref',''),technical_hold='False',duplicate_hold='False',annotation_hold='False',coordinate_frame='EXIF-oriented RGB; zero-based [left,top,right,bottom)' if is_crop else '',copyright_license='',redistribution_rights='not_established' if not is_self else '',image_origin_version='v1')
  if is_self:assert eligible(a)
  if is_crop:
   row['crop_coordinates']=json.dumps([int(m['crop_'+k]) for k in ['x0','y0','x1','y1']]);row['parent_image_path'],row['parent_sha256']=parent_path(r['source'],r['original_filename']);assert row['parent_sha256']==m['parent_sha256']
  records.append(row)
 for r in originals:copy_v1(r)
 for r in historical:
  exact=[c for c in crops if c['crop_sha256']==r['sha256']];holds=[]
  if historical_policy!='retain-keep':holds.append('no explicit accept; historical keep retention not authorized')
  for c in exact:
   if c['technical_status']!='technically_usable':holds.append('current exact-match v3 target requires technical review: '+c['crop_id'])
   if c['label_review_status']!='no_documented_label_conflict':holds.append('current annotation conflict: '+c['crop_id'])
   if c['duplicate_group_id']:holds.append('current unresolved duplicate group: '+c['duplicate_group_id'])
  if holds:held_legacy.append(dict(image_id=r['image_id'],original_filename=r['original_filename'],historical_decision=r['decision'],disposition='excluded_hold',reason=' | '.join(holds)))
  else:copy_v1(r)
 for d in decisions:
  c=cropmap[d['Crop ID']];a=ann[c['original_filename']];raw_decision=d['Review decision'];holds=active_holds(d,c)
  included=raw_decision=='accept' and not holds
  disposition='included' if included else ('accepted_but_held' if raw_decision=='accept' else {'reject':'rejected','pending':'pending','fix_crop':'needs_crop_correction'}[raw_decision])
  dispositions.append(dict(crop_id=d['Crop ID'],original_filename=c['original_filename'],human_decision=raw_decision,preparation_status=d['Preparation status'],disposition=disposition,hold_reasons=' | '.join(holds),human_review_notes=d.get('Review notes',''),human_reviewer=d.get('Human reviewer',''),human_review_date=d.get('Review date',''),review_worksheet='New candidates',review_excel_row=d['_excel_row']))
  if not included:continue
  src=reviewroot/c['crop_filename'];assert sha(src)==c['crop_sha256'];rel='images/healthy_leaf_crops/'+c['crop_id']+'.png';target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);parentrel,parenthash=parent_path('self_collected',c['original_filename']);assert parenthash==c['parent_sha256']
  row=dict.fromkeys(list(v1records[0]),'')
  row.update(image_id=c['crop_id'],image_path=rel,filename=Path(rel).name,class_name='Healthy',label_id='0',source='self_collected',evaluation_subset='healthy_leaf_crops',original_filename=c['original_filename'],parent_image_id=c['parent_image_id'],parent_filename=c['original_filename'],plant_id=c['plant_id'],scene_group_id=c['scene_group_id'],group_id=c['scene_group_id'],leaf_group_id=c['leaf_group_id'],is_crop='True',crop_coordinates=c['crop_coordinates'],crop_sha256=c['crop_sha256'],sha256=c['crop_sha256'],parent_sha256=parenthash,parent_image_path=parentrel,review_status='human_accepted',annotation_source='provenance/inputs/Tomato Field Photo Review.xlsx#review!row='+str(a['_excel_row']),label_source='human_annotated_healthy',source_url='',source_stratum='self_collected_leaf_level',image_type='cropped',visible_leaf_context='individual_leaflet_target_with_context',width=c['crop_width'],height=c['crop_height'],split='external_evaluation',original_excel_filename=c['original_filename'],excel_sheet='review',excel_row=a['_excel_row'],decision=raw_decision,verification_status='human_crop_review_accept',review_reason=d.get('Review notes',''),approval_basis='explicit_human_accept',human_review_decision=raw_decision,human_reviewer=d.get('Human reviewer',''),human_review_date=d.get('Review date',''),human_review_notes=d.get('Review notes',''),human_review_workbook='provenance/inputs/final_review_workbook.xlsx',human_review_sheet='New candidates',human_review_excel_row=d['_excel_row'],original_annotation_decision=a['decision'],original_annotation_reviewer=a['reviewer'],original_annotation_note=a['note'],annotation_source_ref=a.get('source_ref',''),technical_hold='False',duplicate_hold='False',annotation_hold='False',coordinate_frame=c['coordinate_frame'],copyright_license='',redistribution_rights='',image_origin_version='v3_human_review')
  records.append(row)
 for r in online:copy_v1(r)
 write_csv(dest/'manifest.csv',records);write_csv(dest/'provenance/review_dispositions.csv',dispositions);write_csv(dest/'provenance/historical_crop_dispositions.csv',held_legacy,fields=['image_id','original_filename','historical_decision','disposition','reason'])
 # Preserve full per-row v1 attribution and copyright context in an unmodified manifest snapshot.
 (dest/'class_mapping.json').write_text(json.dumps(CLASSES,indent=2)+'\n')
 (dest/'provenance/v1_integrity.json').write_text(json.dumps({str(p.relative_to(v1)):h for p,h in protected.items() if p.is_relative_to(v1)},indent=2)+'\n')
 config=dict(historical_policy=historical_policy,historical_policy_authorization='User explicitly selected Retain historical v1 keep approvals in this task.' if historical_policy=='retain-keep' else 'Only explicit accept decisions admitted.',human_accepted_new=current_counts['accept'],human_rejected_new=current_counts['reject'],human_pending_new=current_counts['pending'],human_fix_crop_new=current_counts['fix_crop'],new_accepted_but_held=sum(r['disposition']=='accepted_but_held' for r in dispositions),historical_crops_available=len(historical),historical_crops_excluded=len(held_legacy),input_sha256=input_hashes)
 (dest/'provenance/build_config.json').write_text(json.dumps(config,indent=2)+'\n')
 shutil.copy2(Path(__file__).with_name('validate_dataset.py'),dest/'validate_dataset.py');shutil.copy2(__file__,dest/'build_dataset.py')
 report=validate(dest)
 assert all(sha(p)==h for p,h in protected.items()),'An input or v1 file changed during the build'
 assert all((p.stat().st_size,p.stat().st_mtime_ns)==sig for p,sig in untouched.items()),'Notebook or checkpoint state changed'
 report.update(v1_original_files_unchanged=len([p for p in protected if p.is_relative_to(v1)]),review_package_files_unchanged=len([p for p in protected if p.is_relative_to(reviewroot)]),notebook_checkpoint_stat_checks=len(untouched),human_accepted_new=current_counts['accept'],accepted_but_held=config['new_accepted_but_held'],historical_crops_available=8,historical_crops_retained=8-len(held_legacy),historical_crops_held=len(held_legacy))
 if dest.is_relative_to(repo):
  imagepaths=[str(p.relative_to(repo)) for directory in [dest/'images',dest/'provenance/parents'] for p in directory.rglob('*') if p.is_file()]
  ignored=subprocess.run(['git','check-ignore','--no-index','--stdin'],cwd=repo,input='\n'.join(imagepaths)+'\n',text=True,capture_output=True)
  assert set(ignored.stdout.splitlines())==set(imagepaths),'Some image files are not ignored by Git'
  tracked=subprocess.run(['git','ls-files','--',str(dest/'images'),str(dest/'provenance/parents')],cwd=repo,text=True,capture_output=True,check=True).stdout
  assert not tracked,'Evaluation image files must not be tracked'
  report.update(git_ignored_image_files=len(imagepaths),git_tracked_image_files=0)
 (dest/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
 held=[r for r in dispositions if r['disposition']=='accepted_but_held']
 text=f'''# Tomato Field Evaluation v2

Finalized from the saved human crop decisions on 26 November 2026. No models were run. Version 1 and its historical results are unchanged.

## Composition

| Evaluation subset | Images | Source |
|---|---:|---|
| `healthy_field_originals` | 8 | Previously selected v1 self-collected original photographs |
| `healthy_leaf_crops` — historical | {8-len(held_legacy)} | v1 crops retained under the user's explicit authorization to preserve historical `keep` approvals |
| `healthy_leaf_crops` — newly accepted | {report['new_crops_included']} | Explicit `accept` decisions with no unresolved recorded holds |
| `online_disease_images` | 46 | Existing v1 online disease images, copied without modification |
| **Total** | **{len(records)}** | Manifest-listed evaluation samples only |

The final review workbook contains **46 accept, 19 reject, 0 pending and 0 fix_crop** decisions. Read from the saved literal cells of `New candidates`, not formula caches. Four accepts remain excluded because the preparation status still says `technical discussion hold` and the human review notes do not resolve it. Their accept decisions are preserved verbatim in `provenance/review_dispositions.csv`:

'''
 for r in held:text+=f"- {r['crop_id']} ({r['original_filename']}): {r['hold_reasons']}.\n"
 text+='''
The review workbook does not provide a human reviewer name or review date on the decision rows; those fields remain blank. Original source annotators are retained separately and are not presented as crop reviewers. No human decision was rewritten or automatically promoted.

Historical v1 `keep` is recorded verbatim as `human_review_decision=keep`, `approval_basis=historical_v1_keep`. It is an explicitly authorized historical approval basis, not a fabricated new `accept` entry. The explicit literal `accept` test applies to all newly included Healthy crops. Online historical crops are disease images and are outside that Healthy approval test.

'''
 for r in held_legacy:text+=f"- Historical exclusion: {r['image_id']} ({r['original_filename']}), originally `{r['historical_decision']}`: {r['reason']}. The file remains untouched in v1.\n"
 text+=f'''
IMG_3798/P031 versus IMG_3799/P032 is still an unresolved scene relationship. No related leaf crop is included. IMG_3782/IMG_3783 (P020) annotation uncertainty remains held and absent. Healthy labels derive exclusively from `class_name=Healthy` and `source=self_collected` in the original annotation workbook; no visual disease suspicion was used to relabel or exclude a source.

## Independence and intended evaluation

There are {report['source_photographs']} distinct source photograph identifiers overall, including {report['self_collected_source_photographs']} self-collected source photographs. Self-collected samples span {report['self_collected_plant_ids']} distinct recorded plant IDs and {report['self_collected_scene_groups']} recorded scene groups. These identifiers are not a verified count of independent physical plants. Different leaf crops and original/crop pairs share source and scene context. Use `scene_group_id` and `parent_image_id` for clustered analysis; never treat each crop as an independent plant.

The Healthy field-original and Healthy leaf-crop subsets describe different image conditions. They support a field-context Healthy check and a human-annotated Healthy leaf-level check. Online disease images provide a separate source stratum; they are not self-collected field photographs. A combined five-class table would mix sources and image conditions and must not be described as an independent five-class field test or laboratory-confirmed health.

The newly accepted crops are visually distinct targets according to the review records. Exact file hashes and recorded leaf-group IDs are unique. Neighboring target rectangles may share background or leaf context; geometric intersection alone is not a repeated-leaf verdict. The source eligibility, historical overlap flags, possible duplicate groups and current preparation holds are applied before selection. Biological independence beyond the recorded groups has not been established.

## Metadata and portability

`manifest.csv` uses canonical class IDs: Healthy 0, Early Blight 1, Late Blight 2, Bacterial Spot 3, Septoria Leaf Spot 4. All file paths are relative to this directory. V1 image IDs and all existing attribution/verification fields are retained for reused records. New image IDs preserve the HLV3 crop IDs.

`is_crop` describes actual image preparation for both self-collected and online images. The 10 previously cropped online images remain in `online_disease_images`; they do not become Healthy leaf samples. Every crop has an original parent, zero-based `[left,top,right,bottom)` coordinates in EXIF-oriented RGB pixels, image/parent SHA-256 hashes and a portable parent reference. Saved evaluation images retain their exact bytes; no resizing or enhancement was performed.

`provenance/parents/` contains raw parent photographs required to reproduce crop pixels when a parent is not already in `healthy_field_originals`. They are audit inputs, **not additional evaluation samples**, and are not counted in the manifest total. The two original workbooks and relevant CSV snapshots are copied unchanged under `provenance/inputs/` so provenance and decisions can be checked after relocation. See their SHA-256 values in `provenance/build_config.json`.

## Validation and reproduction

Run with the project environment (Python 3 and Pillow only):

```sh
python validate_dataset.py .
```

The validator reads the actual decision cells and original annotations, recomputes the accepted/held selection, verifies class/subset membership and parent/group links, loads every image, checks every SHA-256 hash and crop's native source pixels, tests unique evaluation IDs/files/leaf groups, verifies the complete online image set and checks v1 attribution against its manifest snapshot. `validation.json` additionally records the build-time checks that original v1, review-package files, notebooks and checkpoints were not changed.

To reproduce into a **new, nonexistent** directory, using the archived v1 mapping CSV:

```sh
python build_dataset.py --raw-root /path/to/tomato_field_raw \\
  --v1-root /path/to/tomato_field_eval_v1 \\
  --mapping provenance/inputs/v1_image_mapping.csv \\
  --historical-policy retain-keep --repo-root /path/to/plant-disease-diagnosis \\
  --output /path/to/a/new/tomato_field_eval_v2
```

The builder refuses to overwrite an existing dataset. Inputs must match the completed review and original annotations; both workbook snapshots and review dispositions make the selection inspectable. No training or inference dependencies are imported.

## Attribution, copyright and Git safety

All 46 online evaluation files are byte-identical to v1. `source`, `source_url`, `original_image_url`, `source_group_id`, verification status and historical review metadata are retained. The unmodified v1 manifest and original annotation workbook preserve all supplied attribution metadata. There was no explicit redistribution license in the v1 manifest; `copyright_license` remains blank and `redistribution_rights=not_established` for online images. Source attribution does not grant redistribution rights, and no new license is asserted.

The repository ignores both `tomato_field_eval_v2/images/` and `tomato_field_eval_v2/provenance/parents/`, including all third-party image bytes. Do not force-add or publicly redistribute third-party photos. Local validation and build files may be reviewed independently. No commit or push was performed.
'''
 (dest/'README.md').write_text(text)
 # Integrity inventory is separate from sample hashes; includes parent images and all reproducibility inputs.
 inventory={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
 (dest/'provenance/package_sha256.json').write_text(json.dumps(inventory,indent=2)+'\n')
 print(json.dumps(report,indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--raw-root',required=True);p.add_argument('--v1-root',required=True);p.add_argument('--mapping',required=True);p.add_argument('--historical-policy',choices=['retain-keep','require-accept'],required=True);p.add_argument('--repo-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.raw_root,a.v1_root,a.mapping,a.output,a.historical_policy,a.repo_root)
