#!/usr/bin/env python3
"""Independent, conservative witness verifier for ODACC's PNML/OCEL fixtures.

This verifier never invokes ODACC's feasibility/search code. 'valid' only means
that all IMPLEMENTED obligations passed; unsupported PNML features produce
'unsupported', not 'valid'. No global optimality claim is made.
"""
import argparse
import ast
import collections
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Issue:
    code: str
    step: int
    detail: str

class Unsupported(Exception): pass

class Model:
    def __init__(self, path):
        root=ET.parse(path).getroot()
        self.places={};self.transitions={};self.ins=collections.defaultdict(list);self.out=collections.defaultdict(list)
        for p in root.iter('place'):
            self.places[p.get('id')]=dict(color=p.get('color',''), initial=p.findtext('./initialMarking/text'))
        for t in root.iter('transition'):
            self.transitions[t.get('id')]=dict(name=t.findtext('./name/text') or t.get('id'),guard=t.get('guard'),silent=t.get('invisible')=='true')
        self.by_name={v['name']:k for k,v in self.transitions.items()}
        for a in root.iter('arc'):
            src=a.get('source');dst=a.get('target'); ins=a.get('inscription') or ''
            vars=[]
            for part in ins.split(','):
                if ':' not in part: raise Unsupported('Unparsed inscription: '+ins)
                name,typ=part.split(':',1); vars.append((name.strip(),typ.strip()))
            if src in self.places and dst in self.transitions: self.ins[dst].append((src,vars))
            elif src in self.transitions and dst in self.places: self.out[src].append((dst,vars))
            else: raise Unsupported('Unexpected arc '+src+' -> '+dst)

class Replay:
    def __init__(self,model,ocel,strict=True):
        self.model=model;self.ocel=ocel
        self.objects={k:v for k,v in ocel.get('ocel:objects',{}).items()}
        self.events=ocel.get('ocel:events',{})
        self.marking=collections.defaultdict(collections.Counter)
        self.created=set(); self.issues=[]; self.unsupported=[];self.strict=strict; self.marking_reliable=True
        # Explicit initial markings not supported until independently decoded.
        if any(x['initial'] and x['initial'].strip() not in ('','0') for x in model.places.values()):
            self.unsupported.append('PNML nonempty initial marking requires explicit token decoding')

    def issue(self,code,step,detail):self.issues.append(Issue(code,step,detail))
    def unk(self,step,detail):self.unsupported.append(f'step {step}: {detail}')
    def object_type(self,oid):
        obj=self.objects.get(oid)
        if obj:return obj.get('ocel:type')
        # Invented object not in observed OCEL: type may be inferred from firing binding.
        return None
    @staticmethod
    def inscription_tokens(vars,bind):
        singles=[bind[n] for n,t in vars if not t.endswith(' LIST')]
        lists=[[(o,) for o in bind[n]] for n,t in vars if t.endswith(' LIST')]
        if len(lists)>1:raise Unsupported('Multiple LIST vars in one arc')
        if lists:
            return [tuple(singles)+elem for elem in lists[0]]
        return [tuple(singles)]
    def get_binding(self,tr,object_ids,step):
        declarations={}
        for _,arcs in self.model.ins[tr]+self.model.out[tr]:
            for name,t in arcs:
                if name in declarations and declarations[name]!=t:
                    self.unk(step,f'Variable {name} has inconsistent type')
                declarations[name]=t
        # Determine object IDs by type. Invented objects with unknown type are
        # deliberately not guessed except unique fresh-object output variables.
        ids=list(dict.fromkeys(object_ids));typed=collections.defaultdict(list);unknown=[]
        for oid in ids:
            typ=self.object_type(oid)
            if typ:typed[typ].append(oid)
            else:unknown.append(oid)
        if unknown:
            # Support fresh object with unambiguous type and no other unbound unknowns.
            fresh={t for _,a in self.model.out[tr] for n,t in a if n=='nu' and not t.endswith(' LIST')}
            if len(fresh)==1:
                typed[next(iter(fresh))].extend(unknown)
            else:
                self.unk(step,'Unknown object types: '+repr(unknown))
                return None
        bind={}
        for n,t in declarations.items():
            basetype=t.replace(' LIST','')
            if basetype in ('Integer','Real','String','Boolean'):
                self.unk(step,'Data token variable '+n+' cannot be reconstructed solely from object bindings')
                return None
            options=typed[basetype]
            if t.endswith(' LIST'):bind[n]=tuple(options)
            elif len(options)==1:bind[n]=options[0]
            elif len(options)>1:
                self.unk(step,f'Ambiguous scalar variable {n}: {options}');return None
            else:
                self.issue('missing_binding',step,f'No object for {n}:{t}');return None
        consumed=set(o for v in bind.values() for o in (v if isinstance(v,tuple) else [v]))
        if consumed!=set(ids):
            self.issue('unused_object',step,f'Witness objects {ids}; bound objects {sorted(consumed)}')
        return bind
    def check_guard(self,tr,bind,step):
        guard=self.model.transitions[tr]['guard']
        if not guard:return
        # Restricted AST evaluator, never eval() untrusted PNML.
        exp=guard.replace('&&',' and ').replace('||',' or ')
        try:tree=ast.parse(exp,mode='eval')
        except SyntaxError:self.unk(step,'Cannot parse guard: '+guard);return
        def val(n):
            if isinstance(n,ast.Constant):return n.value
            if isinstance(n,ast.Name):
                if n.id in bind:return bind[n.id]
                raise Unsupported('Unbound guard variable '+n.id)
            if isinstance(n,ast.BinOp):
                a,b=val(n.left),val(n.right)
                if isinstance(n.op,ast.Add):return a+b
                if isinstance(n.op,ast.Sub):return a-b
                if isinstance(n.op,ast.Mult):return a*b
                raise Unsupported('Unsupported arithmetic operator')
            if isinstance(n,ast.BoolOp):
                xs=[bool(val(x)) for x in n.values]
                if isinstance(n.op,ast.And):return all(xs)
                if isinstance(n.op,ast.Or):return any(xs)
            if isinstance(n,ast.Compare):
                a=val(n.left)
                for op,right in zip(n.ops,n.comparators):
                    b=val(right)
                    match= (a==b if isinstance(op,ast.Eq) else a<=b if isinstance(op,ast.LtE) else a>=b if isinstance(op,ast.GtE) else a<b if isinstance(op,ast.Lt) else a>b if isinstance(op,ast.Gt) else a!=b if isinstance(op,ast.NotEq) else None)
                    if match is None:raise Unsupported('Unsupported comparator')
                    if not match:return False
                    a=b
                return True
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):
                fn=n.func.id
                if fn=='sum' and len(n.args)==1:return sum(val(n.args[0]))
                if len(n.args)==1:
                    target=val(n.args[0]); ids=target if isinstance(target,(list,tuple)) else [target]
                    out=[]
                    for oid in ids:
                        obj=self.objects.get(oid)
                        if obj is None or fn not in (obj.get('ocel:ovmap') or {}):
                            raise Unsupported('Missing attribute '+fn+' on '+str(oid))
                        out.append(obj['ocel:ovmap'][fn])
                    return out if isinstance(target,(list,tuple)) else out[0]
            raise Unsupported('Unsupported guard construct '+type(n).__name__)
        try:
            if not bool(val(tree.body)):self.unk(step,f'Guard false with OCEL static attributes (model-side repaired values not independently reconstructed): {self.model.transitions[tr]["name"]}: {guard} with {bind}')
        except (Unsupported,TypeError,KeyError,ValueError) as e:self.unk(step,'Guard unresolved: '+str(e))
    def run(self,alignment):
        moves=alignment.get('moves',[])
        consumed_events=[];sum_move_cost=0
        for step,m in enumerate(moves):
            kind=m.get('kind');eid=m.get('event_id');obs=set(m.get('observed_objects') or []);modelobs=set(m.get('model_objects') or m.get('objects') or [])
            sum_move_cost+=m.get('cost') or 0
            if kind not in ('sync','log','model'):
                self.unk(step,'Unsupported move kind '+repr(kind));continue
            if kind in ('sync','log'):
                if not eid or eid not in self.events:
                    self.issue('missing_event',step,f'Unknown event {eid}');continue
                consumed_events.append(eid)
                event=self.events[eid];actual=set(event.get('ocel:omap') or [])
                if obs and obs!=actual:self.unk(step,f'Reported observed object subset does not match OCEL: {eid} event objects {sorted(actual)} reported {sorted(obs)}')
                if kind=='sync':
                    if self.strict and actual!=modelobs:self.issue('hard_sync_mismatch',step,f'{eid}: observed {sorted(actual)} model {sorted(modelobs)}')
                    if event.get('ocel:activity')!=m.get('activity'):self.issue('activity_mismatch',step,f'{eid}: {event.get("ocel:activity")} vs {m.get("activity")}')
            elif eid:self.issue('model_move_has_event',step,str(eid))
            if kind=='log':continue
            label=m.get('transition') or m.get('activity')
            tr=self.model.by_name.get(label) or (label if label in self.model.transitions else None)
            if not tr:
                self.issue('transition_missing',step,str(label));continue
            if kind=='sync' and self.model.transitions[tr]['name']!=m.get('activity'):
                self.issue('transition_activity_mismatch',step,str(label))
            bind=self.get_binding(tr,list(m.get('model_objects') or m.get('objects') or []),step)
            if bind is None:
                self.marking_reliable=False
                continue
            self.check_guard(tr,bind,step)
            inputs=[];outputs=[]
            try:
                for place,arc in self.model.ins[tr]:
                    inputs.extend((place,tuple(tok)) for tok in self.inscription_tokens(arc,bind))
                for place,arc in self.model.out[tr]:
                    outputs.extend((place,tuple(tok)) for tok in self.inscription_tokens(arc,bind))
            except (Unsupported,KeyError) as e:self.unk(step,'Arc unresolved: '+str(e));self.marking_reliable=False;continue
            # DOPID benchmark policy: a transition with input arcs must consume
            # at least one real token.  Empty LIST instantiations do not make
            # an otherwise token-less transition enabled.
            if self.model.ins[tr] and not inputs:
                self.issue('vacuous_input_firing',step,
                           f'{label}: no input token consumed (empty LIST binding)')
                break
            demand=collections.Counter(inputs)
            if self.marking_reliable and any(self.marking[p][token]<num for (p,token),num in demand.items()):
                self.issue('not_enabled',step,f'{label}: missing '+repr([(p,t,n,self.marking[p][t]) for (p,t),n in demand.items() if self.marking[p][t]<n]))
                # Do not proceed with guessed marking after an invalid move.
                break
            for (p,t),n in demand.items():self.marking[p][t]-=n
            for p,t in outputs:self.marking[p][t]+=1
        dups=[i for i,n in collections.Counter(consumed_events).items() if n!=1]
        for eid in dups:self.issue('duplicate_event',-1,eid)
        expected={m.get('event_id') for m in moves if m.get('event_id')} # component-specific expected event list not embedded
        # A complete OCEL may contain unrelated components: do not assume all OCEL events belong to this witness.
        if alignment.get('cost') is not None and alignment.get('cost')<sum_move_cost:
            self.issue('cost_below_move_sum',-1,f'{alignment["cost"]} < {sum_move_cost}')
        status='invalid' if self.issues else ('unresolved' if self.unsupported else 'passed_checked_rules')
        return dict(status=status,reported_cost=alignment.get('cost'),move_cost_sum=sum_move_cost,
                    checked_moves=len(moves),consumed_events=len(consumed_events),issues=[vars(x) for x in self.issues],
                    unsupported=self.unsupported)

def main():
    ap=argparse.ArgumentParser(description='Conservative independent ODACC witness verifier')
    ap.add_argument('--model',required=True);ap.add_argument('--ocel',required=True);ap.add_argument('--result',required=True)
    ap.add_argument('--prefix',type=int,default=-1,help='0-based prefix index; -1 final saved prefix')
    ap.add_argument('--sync',choices=['hard','soft'],default='hard')
    ap.add_argument('--output')
    args=ap.parse_args();p=json.loads(Path(args.result).read_text(encoding='utf8'))
    al=(p['prefix'][args.prefix] if 'prefix' in p else p)
    result=Replay(Model(args.model),json.loads(Path(args.ocel).read_text(encoding='utf8')),args.sync=='hard').run(al)
    result.update(result_file=str(args.result),prefix_position=al.get('prefix_position'),sync=args.sync)
    output=json.dumps(result,indent=2)
    if args.output:Path(args.output).write_text(output,encoding='utf8')
    print(output)
    return 1 if result['status']=='invalid' else 0
if __name__=='__main__':sys.exit(main())
