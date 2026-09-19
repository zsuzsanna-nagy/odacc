#!/usr/bin/env python3
"""Validate the data assumptions used by the curated order-management DOPID."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from ocel2_to_odacc import parse_ocel2


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ocel',required=True,type=Path)
    ap.add_argument('--tolerance',type=float,default=1e-6)
    args=ap.parse_args()
    objects, events, _=parse_ocel2(args.ocel)
    rels={oid:obj.rels for oid,obj in objects.items()}
    def latest(oid,name):
        vals=sorted((tm,v) for n,tm,v in objects[oid].attrs if n==name)
        return vals[-1][1] if vals else None
    order_bad=[]; order_n=0
    for oid,obj in objects.items():
        if obj.typ!='orders': continue
        items=[x for x,q in obj.rels if q=='comprises' and x in objects]
        op=latest(oid,'price')
        vals=[latest(i,'price') for i in items]
        if op is None or any(v is None for v in vals): continue
        order_n+=1; expected=sum(vals)+5.0
        if abs(op-expected)>args.tolerance: order_bad.append((oid,op,expected))
    pkg_bad=[]; pkg_n=0
    for oid,obj in objects.items():
        if obj.typ!='packages': continue
        items=[x for x,q in obj.rels if q=='contains' and x in objects]
        pw=latest(oid,'weight'); vals=[latest(i,'weight') for i in items]
        if pw is None or any(v is None for v in vals): continue
        pkg_n+=1; expected=sum(vals)
        if abs(pw-expected)>args.tolerance: pkg_bad.append((oid,pw,expected))
    out={
      'orders_checked':order_n,'order_price_guard_violations':len(order_bad),
      'packages_checked':pkg_n,'package_weight_guard_violations':len(pkg_bad),
      'sample_order_violations':order_bad[:10],'sample_package_violations':pkg_bad[:10],
    }
    print(json.dumps(out,indent=2))
    return 1 if order_bad or pkg_bad else 0
if __name__=='__main__': raise SystemExit(main())
