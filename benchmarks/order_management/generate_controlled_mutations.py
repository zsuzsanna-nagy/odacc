#!/usr/bin/env python3
"""Generate reproducible controlled deviations from fitting JSON-OCEL 1.0 components."""
from __future__ import annotations
import csv, json, shutil
from pathlib import Path
from datetime import datetime, timedelta

INPUT_DIR = Path('components')
OUTPUT_DIR = Path('controlled_mutations')
# Set to None for every component, or a positive integer for a size-stratified subset.
MAX_COMPONENTS = None


def load(p):
    with p.open(encoding='utf-8') as f: return json.load(f)
def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f: json.dump(d,f,indent=2,ensure_ascii=False)
def activity(e): return e.get('ocel:activity')
def omap(e): return e.setdefault('ocel:omap',[])
def otype(d,oid): return d['ocel:objects'].get(oid,{}).get('ocel:type')
def events_by_activity(d,a): return [(eid,e) for eid,e in d['ocel:events'].items() if activity(e)==a]

def record(rows, src, out, cls, mutation, target_event='', target_object='', details='', nominal=1):
    rows.append(dict(source_file=src.name, mutated_file=str(out.relative_to(OUTPUT_DIR)),
        mutation_class=cls, mutation=mutation, target_event=target_event,
        target_object=target_object, intended_dimension=cls,
        nominal_single_deviation_cost=nominal,
        expected_optimum='TO_VALIDATE_WITH_BOTH_BACKENDS', details=details))

def select_files(files):
    if MAX_COMPONENTS is None or len(files)<=MAX_COMPONENTS: return files
    # evenly spaced by encoded event count in filename/file contents
    sized=[]
    for p in files:
        d=load(p); sized.append((len(d['ocel:events']),p))
    sized.sort()
    idx=sorted(set(round(i*(len(sized)-1)/(MAX_COMPONENTS-1)) for i in range(MAX_COMPONENTS)))
    return [sized[i][1] for i in idx]

def main():
    files=select_files(sorted(INPUT_DIR.glob('*.jsonocel')))
    if not files: raise SystemExit(f'No .jsonocel files under {INPUT_DIR.resolve()}')
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    (OUTPUT_DIR/'fit').mkdir(parents=True)
    rows=[]
    for src in files:
        base=load(src); shutil.copy2(src,OUTPUT_DIR/'fit'/src.name)

        # DATA 1: package weight violates create-package aggregate while structure is unchanged.
        cps=events_by_activity(base,'create package')
        if cps:
            eid,e=cps[0]; pkgs=[o for o in omap(e) if otype(base,o)=='PACKAGE']
            if pkgs:
                oid=pkgs[0]; d=load(src); old=d['ocel:objects'][oid]['ocel:ovmap'].get('weight')
                if isinstance(old,(int,float)):
                    new=old+0.137
                    d['ocel:objects'][oid]['ocel:ovmap']['weight']=new
                    out=OUTPUT_DIR/'data_package_weight'/src.name; dump(out,d)
                    record(rows,src,out,'data','wrong_package_weight',eid,oid,f'weight {old!r} -> {new!r}')

        # DATA 2: order price violates pay-order aggregate while event/object graph is unchanged.
        pays=events_by_activity(base,'pay order')
        if pays:
            eid,e=pays[0]; orders=[o for o in omap(e) if otype(base,o)=='ORDER']
            if orders:
                oid=orders[0]; d=load(src); old=d['ocel:objects'][oid]['ocel:ovmap'].get('price')
                if isinstance(old,(int,float)):
                    new=old+7.31
                    d['ocel:objects'][oid]['ocel:ovmap']['price']=new
                    out=OUTPUT_DIR/'data_order_price'/src.name; dump(out,d)
                    record(rows,src,out,'data','wrong_order_price',eid,oid,f'price {old!r} -> {new!r}')

        # CONTROL FLOW: remove exactly one observed pick-item event.
        picks=events_by_activity(base,'pick item')
        if picks:
            eid,_=picks[len(picks)//2]; d=load(src); del d['ocel:events'][eid]
            out=OUTPUT_DIR/'cf_missing_pick'/src.name; dump(out,d)
            record(rows,src,out,'control_flow','remove_pick_event',eid,'','Removed one observed pick item event')

        # CONTROL FLOW: duplicate a pick event with a unique ID and timestamp 1 microsecond later.
        if picks:
            eid,e=picks[len(picks)//2]; d=load(src); ne=json.loads(json.dumps(d['ocel:events'][eid]))
            nid=eid+'__controlled_extra'; ts=ne.get('ocel:timestamp')
            if ts:
                try:
                    dt=datetime.fromisoformat(ts.replace('Z','+00:00'))+timedelta(microseconds=1)
                    ne['ocel:timestamp']=dt.isoformat()
                except ValueError: pass
            d['ocel:events'][nid]=ne
            out=OUTPUT_DIR/'cf_extra_pick'/src.name; dump(out,d)
            record(rows,src,out,'control_flow','duplicate_pick_event',nid,'',f'Duplicate of {eid}')

        # OBJECT: remove one ITEM relation from create package; all objects/attributes remain.
        if cps:
            eid,e=cps[0]; items=[o for o in omap(e) if otype(base,o)=='ITEM']
            if items:
                oid=items[0]; d=load(src); d['ocel:events'][eid]['ocel:omap'].remove(oid)
                out=OUTPUT_DIR/'object_missing_item_relation'/src.name; dump(out,d)
                record(rows,src,out,'object','remove_item_relation_from_create_package',eid,oid,'Removed exactly one ITEM from create package ocel:omap')

        # OBJECT: add an ITEM not originally related to the chosen create-package event, when available.
        if cps:
            eid,e=cps[0]; linked=set(omap(e)); candidates=[oid for oid,o in base['ocel:objects'].items() if o.get('ocel:type')=='ITEM' and oid not in linked]
            if candidates:
                oid=candidates[0]; d=load(src); d['ocel:events'][eid]['ocel:omap'].append(oid)
                out=OUTPUT_DIR/'object_extra_item_relation'/src.name; dump(out,d)
                record(rows,src,out,'object','add_unrelated_item_relation_to_create_package',eid,oid,'Added one existing ITEM not originally related to this create package event')

    fields=['source_file','mutated_file','mutation_class','mutation','target_event','target_object','intended_dimension','nominal_single_deviation_cost','expected_optimum','details']
    with (OUTPUT_DIR/'mutation_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    dump(OUTPUT_DIR/'mutation_manifest.json',rows)
    summary={}
    for r in rows: summary[r['mutation']]=summary.get(r['mutation'],0)+1
    dump(OUTPUT_DIR/'README.json',{'source_components':len(files),'mutations':len(rows),'by_type':summary,
        'important_note':'nominal_single_deviation_cost is the intended injected deviation count, not a proven alignment optimum. Validate optimum/cost decomposition with both JODAP and SMT before using it as ground truth.'})
    print(f'Generated {len(rows)} mutations from {len(files)} components in {OUTPUT_DIR}')
    print(summary)
if __name__=='__main__': main()
