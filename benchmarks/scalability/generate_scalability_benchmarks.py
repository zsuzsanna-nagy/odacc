from __future__ import annotations
import argparse, copy, json, os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


def iso(base, minutes):
    return (base + timedelta(minutes=minutes)).isoformat()


def make_log(num_products=2, num_orders=1, interleave=False, payment='bank', violation=None, joint_merge=False):
    base = datetime(2025,1,1,9,0,0,tzinfo=timezone.utc)
    objects = {}
    per_order_events = []
    eid = 0
    for oi in range(1, num_orders+1):
        oid=f'o{oi}'
        budget=max(1000, num_products*120)
        objects[oid]={'ocel:type':'ORDER','ocel:ovmap':{'vip':1,'budget':budget,'priority':2}}
        pids=[]
        for pi in range(1,num_products+1):
            pid=f'o{oi}_p{pi}' if num_orders>1 else f'p{pi}'
            pids.append(pid)
            objects[pid]={'ocel:type':'PRODUCT','ocel:ovmap':{'cost':100}}
        events=[]
        d=3
        if violation=='place_data' and oi==1:
            d=1
        events.append(('place order',[oid]+pids,{'d':d}))
        for pid in pids:
            events.append(('pick item',[oid,pid],{}))
        if payment=='bank':
            events.append(('pay bank transfer',[oid]+pids,{}))
        else:
            events.append(('pay credit card',[oid],{}))
        ship_d, ship_m = 3,0
        if violation=='ship_data' and oi==1:
            ship_m=1
        events.append(('ship',[oid]+pids,{'d':ship_d,'m':ship_m}))
        per_order_events.append(events)
    if violation=='vip': objects['o1']['ocel:ovmap']['vip']=0
    if violation=='priority': objects['o1']['ocel:ovmap']['priority']=1
    if violation=='budget': objects['o1']['ocel:ovmap']['budget']=max(0,num_products*100-1)
    if violation=='cost':
        objects['p1' if num_orders==1 else 'o1_p1']['ocel:ovmap']['cost']=10000

    flat=[]
    if interleave and num_orders>1:
        maxlen=max(map(len,per_order_events))
        for k in range(maxlen):
            for oi,evs in enumerate(per_order_events):
                if k<len(evs): flat.append((oi,evs[k]))
    else:
        for oi,evs in enumerate(per_order_events):
            for e in evs: flat.append((oi,e))

    events={}
    for pos,(oi,(act,omap,vmap)) in enumerate(flat):
        events[str(pos)]={'ocel:activity':act,'ocel:timestamp':iso(base,pos),'ocel:omap':omap,'ocel:vmap':vmap}
    if joint_merge and num_orders>1:
        # Add a deliberately cross-component event. It is useful for merge-cost/scalability tests,
        # not intended to be fitting with the supplied model.
        omap=[]
        for oi in range(1,num_orders+1):
            oid=f'o{oi}'; pid=f'o{oi}_p1'
            omap += [oid,pid]
        pos=len(events)
        events[str(pos)]={'ocel:activity':'ship','ocel:timestamp':iso(base,pos),'ocel:omap':omap,'ocel:vmap':{'d':3,'m':0}}

    attrs=['vip','budget','priority','cost','d','m']
    return {
        'ocel:global-log':{'ocel:attribute-names':attrs,'ocel:object-types':['ORDER','PRODUCT'],'ocel:version':['1.0'],'ocel:ordering':['timestamp']},
        'ocel:events':events,'ocel:objects':objects
    }


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def model_variant(src:Path,dst:Path,level:str):
    text=src.read_text(encoding='utf-8')
    if level=='simple':
        # No object-attribute guard except aggregate bank transfer.
        text=text.replace('guard="((d &gt; 2) &amp;&amp; (vip(o) == 1))"','guard="(d &gt; 2)"')
        text=text.replace('guard="(((((d &lt;= 5) &amp;&amp; (m == 0)) || ((d &gt; 5) &amp;&amp; (m == 1)))) &amp;&amp; (priority(o) &gt;= 2))"',
                          'guard="(((d &lt;= 5) &amp;&amp; (m == 0)) || ((d &gt; 5) &amp;&amp; (m == 1)))"')
    elif level=='complex':
        text=text.replace('guard="((d &gt; 2) &amp;&amp; (vip(o) == 1))"',
                          'guard="((((d &gt; 2) &amp;&amp; (vip(o) == 1)) &amp;&amp; (priority(o) &gt;= 2)) &amp;&amp; (budget(o) &gt;= sum(cost(P))))"')
        text=text.replace('guard="(sum(cost(P)) &lt;= budget(o))"',
                          'guard="((((sum(cost(P)) &lt;= budget(o)) &amp;&amp; (budget(o) &gt;= 100)) &amp;&amp; (vip(o) == 1)) &amp;&amp; (priority(o) &gt;= 2))"')
        text=text.replace('guard="(((((d &lt;= 5) &amp;&amp; (m == 0)) || ((d &gt; 5) &amp;&amp; (m == 1)))) &amp;&amp; (priority(o) &gt;= 2))"',
                          'guard="((((((d &lt;= 5) &amp;&amp; (m == 0)) || ((d &gt; 5) &amp;&amp; (m == 1))) &amp;&amp; (priority(o) &gt;= 2)) &amp;&amp; (vip(o) == 1)) &amp;&amp; (budget(o) &gt;= sum(cost(P))))"')
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(text,encoding='utf-8')
    ET.parse(dst)


def generate(root: Path, base_model: Path):
    if not base_model.is_file():
        raise FileNotFoundError(f'Base model not found: {base_model}')

    models = root / 'models'
    model_variant(base_model, models / 'net_guard_simple.pnml', 'simple')
    model_variant(base_model, models / 'net_guard_medium.pnml', 'medium')
    model_variant(base_model, models / 'net_guard_complex.pnml', 'complex')

    # Length / object-cardinality scaling in ONE component: n products -> n+3 observed events.
    for n in [1,2,5,10,20,50,100]:
        write_json(root/'length_scale'/f'products_{n:03d}_events_{n+3:03d}.jsonocel',make_log(num_products=n))

    # Independent component scaling. Each component contains one order and two products.
    for n in [1,2,5,10,20]:
        write_json(root/'component_scale'/f'orders_{n:02d}_interleaved.jsonocel',make_log(num_products=2,num_orders=n,interleave=True))

    # Merge scaling. Keep the default set intentionally small enough for routine
    # batch runs; the extended set contains the known worst-case combinations.
    quick_merge = [(2, 1), (2, 2), (3, 1)]
    extended_merge = [
        (2, 5), (3, 2), (3, 5),
        (5, 1), (5, 2), (5, 5),
    ]
    for folder_name, cases in [
        ('merge_scale', quick_merge),
        ('merge_scale_extended', extended_merge),
    ]:
        for components, products_per_component in cases:
            write_json(
                root/folder_name/
                f'merge_c{components:02d}_p{products_per_component:02d}.jsonocel',
                make_log(
                    num_products=products_per_component,
                    num_orders=components,
                    interleave=True,
                    joint_merge=True,
                ),
            )

    # Deviation scaling. Routine runs stop at 10 products; the 20-product set is
    # separated because a deviation can deliberately trigger expensive repair.
    deviation_cases = [
        ('fit', None),
        ('early_place_data', 'place_data'),
        ('late_ship_data', 'ship_data'),
        ('vip', 'vip'),
        ('priority', 'priority'),
        ('budget', 'budget'),
        ('cost', 'cost'),
    ]
    for folder_name, sizes in [
        ('deviation_scale', [1, 2, 5, 10]),
        ('deviation_scale_extended', [20]),
    ]:
        for n in sizes:
            for name, violation in deviation_cases:
                write_json(
                    root/folder_name/f'products_{n:03d}_{name}.jsonocel',
                    make_log(num_products=n, violation=violation),
                )

    # Same fitting workload, intended to be run against simple/medium/complex model variants.
    write_json(root/'guard_complexity'/'fit_20_products.jsonocel',make_log(num_products=20))
    write_json(root/'guard_complexity'/'violate_budget_20_products.jsonocel',make_log(num_products=20,violation='budget'))

    readme=f'''# ODACC synthetic scalability benchmark

This package complements the small correctness examples and varies one main feature at a time.

## Models
- `models/net_guard_simple.pnml`: same control/object flow, simple guards.
- `models/net_guard_medium.pnml`: current object-attribute model.
- `models/net_guard_complex.pnml`: additional conjunctions and aggregate object-attribute conditions.

## Workloads
- `length_scale`: one ORDER with 1, 2, 5, 10, 20, 50, or 100 PRODUCT objects.
- `component_scale`: increasing numbers of independent interleaved components.
- `merge_scale`: routine merge tests: (2 components, 1 product each), (2,2), and (3,1).
- `merge_scale_extended`: harder merge cases up to 5 components and 5 products/component. Run these only after the routine merge cases complete.
- `deviation_scale`: every deviation type at 1, 2, 5, and 10 products.
- `deviation_scale_extended`: the same deviation types at 20 products; these may deliberately trigger expensive repair/fallback behavior.
- `guard_complexity`: identical workloads for simple, medium, and complex guard variants.

## Suggested experiments
1. Run `length_scale` with `--mode online` for both SMT and symbolic backends. Plot every row in `prefix_timings.csv`.
2. Run `deviation_scale` before `deviation_scale_extended`. Compare fitting, early/late event-data, and object-attribute deviations at the same product count.
3. Run `merge_scale` with `--progress-updates`. The line marked `[MERGE]` is the important latency. Only then try `merge_scale_extended`.
4. For merge analysis, compare both the number of components and products per component instead of reporting only total log size.
5. Run `guard_complexity` against all three PNML variants to isolate formula complexity.

Example routine merge run:
```powershell
python batch_folder.py --mode online --backend symbolic --cocomot-root ../cocomot-main --folder examples/scalability/merge_scale --model examples/scalability/models/net_guard_simple.pnml --out results/merge_symbolic --progress-updates
```

Example deviation run:
```powershell
python batch_folder.py --mode online --backend symbolic --cocomot-root ../cocomot-main --folder examples/scalability/deviation_scale --model examples/scalability/models/net_guard_simple.pnml --out results/deviation_symbolic --progress-updates
```

`prefix_timings.csv` contains one row per online update, including `merged`, activity, event-object count, component-object count, cost, and wall time.

## Public logs
Public OCELs remain useful for structural scalability. Synthetic logs are preferable for controlled data-aware guard experiments. The current OCEL 1.0 importer does not ingest explicit O2O updates, so O2O-density experiments require an OCEL 2.0/stream-native importer.
'''
    (root/'README.md').write_text(readme,encoding='utf-8')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Generate the ODACC synthetic scalability benchmark suites.'
    )
    ap.add_argument(
        '--model',
        required=True,
        type=Path,
        help='Base PNML model from which the simple/medium/complex guard variants are derived.',
    )
    ap.add_argument(
        '--out',
        default='scalability',
        type=Path,
        help='Output directory (default: scalability).',
    )
    args = ap.parse_args()
    generate(args.out, args.model)
    print(args.out.resolve())
