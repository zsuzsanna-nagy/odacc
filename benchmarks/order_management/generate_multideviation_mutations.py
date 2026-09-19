#!/usr/bin/env python3
"""Generate deterministic multi-deviation JSON-OCEL benchmarks from fitting components.

The injected mutation count is metadata, NOT a claimed optimal alignment cost.
"""
from __future__ import annotations
import argparse, csv, json, shutil, re
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

MIX2 = [
 ('missing_pick','package_weight'), ('extra_pick','extra_relation'),
 ('order_price','missing_relation'), ('missing_pick','extra_relation'),
 ('extra_pick','package_weight'), ('order_price','extra_pick'),
]
MIX3 = [
 ('missing_pick','package_weight','extra_relation'),
 ('extra_pick','order_price','missing_relation'),
 ('missing_pick','order_price','missing_relation'),
 ('extra_pick','package_weight','extra_relation'),
]
SAME2 = [('missing_pick','missing_pick'),('extra_pick','extra_pick'),
         ('package_weight','order_price'),('missing_relation','missing_relation'),
         ('extra_relation','extra_relation')]
STRESS = [
 ('missing_pick','extra_pick','package_weight','missing_relation'),
 ('missing_pick','order_price','extra_relation','extra_pick'),
 ('missing_pick','missing_pick','package_weight','missing_relation','extra_relation'),
 ('extra_pick','extra_pick','order_price','missing_relation','extra_relation'),
]

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def dump(p,d): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def typ(d,oid): return d['ocel:objects'].get(oid,{}).get('ocel:type')
def by_act(d,a): return [(eid,e) for eid,e in d['ocel:events'].items() if e.get('ocel:activity')==a]
def evcount(p):
 m=re.search(r'events_(\d+)',p.name); return int(m.group(1)) if m else len(load(p)['ocel:events'])

def apply(d, kind, occurrence, serial):
    """Mutate d in place. Return detail dict, or None if unavailable."""
    if kind in ('missing_pick','extra_pick'):
        picks=by_act(d,'pick item')
        # Ignore already-created controlled duplicates when selecting originals.
        picks=[x for x in picks if '__multi_extra_' not in x[0]]
        if not picks: return None
        # Spread repeated mutations through the lifecycle.
        idx=min(len(picks)-1, round((occurrence+1)*len(picks)/(occurrence+2))-1)
        idx=max(0,idx); eid,e=picks[idx]
        if kind=='missing_pick':
            del d['ocel:events'][eid]
            return {'kind':kind,'dimension':'control_flow','target_event':eid,'target_object':'','details':'removed pick item event'}
        ne=deepcopy(e); nid=f'{eid}__multi_extra_{serial}'
        ts=ne.get('ocel:timestamp')
        if ts:
            try: ne['ocel:timestamp']=(datetime.fromisoformat(ts.replace('Z','+00:00'))+timedelta(microseconds=serial)).isoformat()
            except ValueError: pass
        d['ocel:events'][nid]=ne
        return {'kind':kind,'dimension':'control_flow','target_event':nid,'target_object':'','details':f'duplicate of {eid}'}

    if kind in ('package_weight','missing_relation','extra_relation'):
        cps=by_act(d,'create package')
        if not cps: return None
        # Cycle through package events when repeated.
        eid,e=cps[occurrence % len(cps)]
        if kind=='package_weight':
            pkgs=[o for o in e.get('ocel:omap',[]) if typ(d,o)=='PACKAGE']
            if not pkgs: return None
            oid=pkgs[0]; ov=d['ocel:objects'][oid].setdefault('ocel:ovmap',{}); old=ov.get('weight')
            if not isinstance(old,(int,float)): return None
            delta=0.137*(occurrence+1); ov['weight']=old+delta
            return {'kind':kind,'dimension':'data','target_event':eid,'target_object':oid,'details':f'package weight {old!r} -> {ov["weight"]!r}'}
        if kind=='missing_relation':
            items=[o for o in e.get('ocel:omap',[]) if typ(d,o)=='ITEM']
            if not items: return None
            oid=items[occurrence % len(items)]; e['ocel:omap'].remove(oid)
            return {'kind':kind,'dimension':'object','target_event':eid,'target_object':oid,'details':'removed ITEM relation from create package'}
        linked=set(e.get('ocel:omap',[])); candidates=[oid for oid,o in d['ocel:objects'].items() if o.get('ocel:type')=='ITEM' and oid not in linked]
        if not candidates: return None
        oid=candidates[occurrence % len(candidates)]; e.setdefault('ocel:omap',[]).append(oid)
        return {'kind':kind,'dimension':'object','target_event':eid,'target_object':oid,'details':'added unrelated existing ITEM relation to create package'}

    if kind=='order_price':
        pays=by_act(d,'pay order')
        if not pays: return None
        eid,e=pays[occurrence % len(pays)]; orders=[o for o in e.get('ocel:omap',[]) if typ(d,o)=='ORDER']
        if not orders: return None
        oid=orders[0]; ov=d['ocel:objects'][oid].setdefault('ocel:ovmap',{}); old=ov.get('price')
        if not isinstance(old,(int,float)): return None
        delta=7.31*(occurrence+1); ov['price']=old+delta
        return {'kind':kind,'dimension':'data','target_event':eid,'target_object':oid,'details':f'order price {old!r} -> {ov["price"]!r}'}
    return None

def make_case(src,out,combo,case_id,family):
    d=load(src); seen={}; details=[]
    for serial,kind in enumerate(combo,1):
        occ=seen.get(kind,0); info=apply(d,kind,occ,serial); seen[kind]=occ+1
        if info is None: return None
        info['ordinal']=serial; details.append(info)
    dump(out,d)
    dims={x['dimension'] for x in details}
    return {'case_id':case_id,'family':family,'source_file':src.name,'mutated_file':str(out),
            'injected_deviation_count':len(details),'mutation_kinds':'+'.join(combo),
            'dimensions':'+'.join(sorted(dims)),'expected_optimum':'TO_VALIDATE_WITH_JODAP_AND_SMT',
            'mutations':details}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('input_dir',nargs='?',default='fit'); ap.add_argument('output_dir',nargs='?',default='multi_deviation_benchmark'); args=ap.parse_args()
 inp=Path(args.input_dir); out=Path(args.output_dir)
 files=sorted(inp.glob('*.jsonocel'), key=lambda p:(evcount(p),p.name))
 if not files: raise SystemExit(f'No .jsonocel files in {inp}')
 if out.exists(): shutil.rmtree(out)
 (out/'fit').mkdir(parents=True); [shutil.copy2(p,out/'fit'/p.name) for p in files]
 rows=[]; case_no=0
 def add(src,combo,family):
  nonlocal case_no
  case_no+=1; cid=f'{family}_{case_no:03d}'; dest=out/family/f'{src.stem}__{cid}.jsonocel'
  r=make_case(src,dest,combo,cid,family)
  if r: rows.append(r); return True
  case_no-=1
  return False
 def add_first(src, combos, family, offset=0):
  for j in range(len(combos)):
   combo=combos[(offset+j)%len(combos)]
   if add(src,combo,family): return True
  return False
 # 34 mixed pairs: one per component.
 for i,src in enumerate(files): add_first(src,MIX2,'two_mixed',i%len(MIX2))
 # 12 repeated/same-dimension pairs, stratified over size.
 sel=[files[round(i*(len(files)-1)/11)] for i in range(12)]
 for i,src in enumerate(sel): add_first(src,SAME2,'two_same_dimension',i%len(SAME2))
 # 20 mixed triples, stratified.
 sel=[files[round(i*(len(files)-1)/19)] for i in range(20)]
 for i,src in enumerate(sel): add_first(src,MIX3,'three_mixed',i%len(MIX3))
 # 10 stress cases, biased toward medium/large components.
 start=len(files)//3; pool=files[start:]
 sel=[pool[round(i*(len(pool)-1)/9)] for i in range(10)]
 for i,src in enumerate(sel): add_first(src,STRESS,'stress_4_5',i%len(STRESS))
 # Flatten manifests.
 flat=[]
 for r in rows:
  rr={k:v for k,v in r.items() if k!='mutations'}; rr['mutations_json']=json.dumps(r['mutations'],ensure_ascii=False); flat.append(rr)
 fields=['case_id','family','source_file','mutated_file','injected_deviation_count','mutation_kinds','dimensions','expected_optimum','mutations_json']
 with (out/'mutation_manifest.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(flat)
 dump(out/'mutation_manifest.json',rows)
 summary={}
 for r in rows: summary[r['family']]=summary.get(r['family'],0)+1
 dump(out/'README.json',{'source_components':len(files),'generated_cases':len(rows),'by_family':summary,
  'correctness_rule':'Injected deviation count is NOT ground truth. A result is correct if it is feasible and has globally minimal total cost. Co-optimal alignments with different CF/data/object decompositions are accepted.',
  'recommended_order':['two_mixed','two_same_dimension','three_mixed','stress_4_5'],
  'smt_note':'Use SMT on a representative size-stratified subset first; expand while runtime remains practical.'})
 print(json.dumps({'cases':len(rows),'by_family':summary},indent=2))
if __name__=='__main__': main()
