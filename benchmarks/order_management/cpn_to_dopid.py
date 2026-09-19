#!/usr/bin/env python3
"""CPN -> DOPID conversion helpers for the ODACC experiments.

This is deliberately a *semantic* converter, not a general CPN-ML compiler.
CPN Tools models often contain simulator bookkeeping, stochastic routing and
logging transitions that should not become conformance constraints.

For ``--profile order-management`` the script:
  1. parses the supplied CPN and writes a transition/guard report;
  2. validates that the expected business transitions occur in the CPN;
  3. emits the curated DOPID PNML used by ODACC.

The emitted model keeps the CPN's business lifecycle but removes stochastic
``fbool`` routing and simulation infrastructure.  Two meaningful guards are
added from the model/log semantics:

    pay order:       (price(o) == (sum(price(I)) + 5.0))
    create package:  (weight(p) == sum(weight(I)))

The model intentionally uses only ORDER, ITEM and PACKAGE.  This corresponds
to the benchmark perspective configured in order_management_profile.json.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


def norm(text: Optional[str]) -> str:
    return " ".join((text or "").replace("\n", " ").split())


def parse_cpn(path: Path) -> Dict[str, object]:
    root = ET.parse(path).getroot()
    transitions = []
    places = []
    for t in root.iter("trans"):
        name = norm(t.findtext("text"))
        guard = ""
        cond = t.find("cond")
        if cond is not None:
            guard = norm("".join(cond.itertext()))
        transitions.append({"id": t.get("id"), "name": name, "guard": guard})
    for p in root.iter("place"):
        name = norm(p.findtext("text"))
        colour = ""
        typ = p.find("type")
        if typ is not None:
            colour = norm("".join(typ.itertext()))
        places.append({"id": p.get("id"), "name": name, "colour": colour})
    return {"transitions": transitions, "places": places}


def load_profile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _x(s: str) -> str:
    return html.escape(s, quote=True)


def order_management_pnml(profile: dict) -> str:
    guards = profile.get("guard_design", {})
    pay_guard = guards.get("pay order", "(price(o) == (sum(price(I)) + 5.0))")
    package_guard = guards.get("create package", "(weight(p) == sum(weight(I)))")

    # q_order_items and q_package_items are relation places.  The list-valued
    # arc inscriptions let one transition bind exactly the objects appearing in
    # the corresponding OCEL event without flattening the object-centric log.
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<pnml>
  <net id="order_management_dopid" type="http://www.pnml.org/version-2009/grammar/pnmlcoremodel">
    <page id="page1">
      <name><text>Order management DOPID (curated from CPN)</text></name>

      <!-- Fresh object pools.  Creation is silent; the first visible OCEL event
           binds the created identity. -->
      <place id="q_order_new" color="ORDER"><name><text>order new</text></name></place>
      <place id="q_item_available" color="ITEM"><name><text>item available</text></name></place>
      <place id="q_package_new" color="PACKAGE"><name><text>package new</text></name></place>

      <!-- Order lifecycle. -->
      <place id="q_order_after_place" color="ORDER"><name><text>order placed</text></name></place>
      <place id="q_order_items" color="ORDER,ITEM"><name><text>order items</text></name></place>
      <place id="q_order_confirmed" color="ORDER"><name><text>order confirmed</text></name></place>
      <place id="q_order_paid" color="ORDER"><name><text>order paid</text></name><finalMarking><text>1</text></finalMarking></place>

      <!-- Item lifecycle. -->
      <place id="q_item_oos" color="ITEM"><name><text>item out of stock</text></name></place>
      <place id="q_item_picked" color="ITEM"><name><text>item picked</text></name></place>

      <!-- Package/item relation lifecycle. -->
      <place id="q_package_ready" color="PACKAGE,ITEM"><name><text>package ready</text></name></place>
      <place id="q_package_sent" color="PACKAGE,ITEM"><name><text>package sent</text></name></place>
      <place id="q_package_delivered" color="PACKAGE,ITEM"><name><text>package delivered</text></name><finalMarking><text>1</text></finalMarking></place>

      <transition id="create_order" invisible="true"><name><text>create order object</text></name></transition>
      <transition id="create_item" invisible="true"><name><text>create item object</text></name></transition>
      <transition id="create_package_object" invisible="true"><name><text>create package object</text></name></transition>

      <transition id="place_order"><name><text>place order</text></name></transition>
      <transition id="confirm_order"><name><text>confirm order</text></name></transition>
      <transition id="pick_item"><name><text>pick item</text></name></transition>
      <transition id="item_oos"><name><text>item out of stock</text></name></transition>
      <transition id="reorder_item"><name><text>reorder item</text></name></transition>
      <transition id="pay_order" guard="{_x(pay_guard)}"><name><text>pay order</text></name></transition>
      <transition id="payment_reminder"><name><text>payment reminder</text></name></transition>
      <transition id="create_package" guard="{_x(package_guard)}"><name><text>create package</text></name></transition>
      <transition id="send_package"><name><text>send package</text></name></transition>
      <transition id="package_delivered"><name><text>package delivered</text></name></transition>
      <transition id="failed_delivery"><name><text>failed delivery</text></name></transition>

      <arc source="create_order" target="q_order_new" inscription="nu:ORDER"/>
      <arc source="create_item" target="q_item_available" inscription="nu:ITEM"/>
      <arc source="create_package_object" target="q_package_new" inscription="nu:PACKAGE"/>

      <!-- place order binds the order and exactly the event's item list, while
           keeping the items independently available for stock/picking events. -->
      <arc source="q_order_new" target="place_order" inscription="o:ORDER"/>
      <arc source="q_item_available" target="place_order" inscription="I:ITEM LIST"/>
      <arc source="place_order" target="q_order_after_place" inscription="o:ORDER"/>
      <arc source="place_order" target="q_order_items" inscription="o:ORDER,I:ITEM LIST"/>
      <arc source="place_order" target="q_item_available" inscription="I:ITEM LIST"/>

      <!-- Confirmation uses the persistent order-item relation because the
           OCEL confirmation event contains the order and all its items. -->
      <arc source="q_order_after_place" target="confirm_order" inscription="o:ORDER"/>
      <arc source="q_order_items" target="confirm_order" inscription="o:ORDER,I:ITEM LIST" synchronization="exact"/>
      <arc source="confirm_order" target="q_order_confirmed" inscription="o:ORDER"/>
      <arc source="confirm_order" target="q_order_items" inscription="o:ORDER,I:ITEM LIST"/>

      <!-- Payment is independent from the item/package branch, as in the CPN.
           The guard deliberately occurs late: it checks historical/static
           object data after the order has already progressed. -->
      <arc source="q_order_confirmed" target="pay_order" inscription="o:ORDER"/>
      <arc source="q_order_items" target="pay_order" inscription="o:ORDER,I:ITEM LIST" synchronization="exact"/>
      <arc source="pay_order" target="q_order_paid" inscription="o:ORDER"/>
      <arc source="pay_order" target="q_order_items" inscription="o:ORDER,I:ITEM LIST"/>

      <arc source="q_order_confirmed" target="payment_reminder" inscription="o:ORDER"/>
      <arc source="q_order_items" target="payment_reminder" inscription="o:ORDER,I:ITEM LIST" synchronization="exact"/>
      <arc source="payment_reminder" target="q_order_confirmed" inscription="o:ORDER"/>
      <arc source="payment_reminder" target="q_order_items" inscription="o:ORDER,I:ITEM LIST"/>

      <!-- Item stock loop. -->
      <arc source="q_item_available" target="pick_item" inscription="i:ITEM"/>
      <arc source="pick_item" target="q_item_picked" inscription="i:ITEM"/>
      <arc source="q_item_available" target="item_oos" inscription="i:ITEM"/>
      <arc source="item_oos" target="q_item_oos" inscription="i:ITEM"/>
      <arc source="q_item_oos" target="reorder_item" inscription="i:ITEM"/>
      <arc source="reorder_item" target="q_item_available" inscription="i:ITEM"/>

      <!-- A package may combine picked items from different orders.  This is
           the important object-centric component-merging behavior of the log. -->
      <arc source="q_package_new" target="create_package" inscription="p:PACKAGE"/>
      <arc source="q_item_picked" target="create_package" inscription="I:ITEM LIST"/>
      <arc source="create_package" target="q_package_ready" inscription="p:PACKAGE,I:ITEM LIST"/>

      <arc source="q_package_ready" target="send_package" inscription="p:PACKAGE,I:ITEM LIST" synchronization="exact"/>
      <arc source="send_package" target="q_package_sent" inscription="p:PACKAGE,I:ITEM LIST"/>
      <arc source="q_package_sent" target="package_delivered" inscription="p:PACKAGE,I:ITEM LIST" synchronization="exact"/>
      <arc source="package_delivered" target="q_package_delivered" inscription="p:PACKAGE,I:ITEM LIST"/>
      <arc source="q_package_sent" target="failed_delivery" inscription="p:PACKAGE,I:ITEM LIST" synchronization="exact"/>
      <arc source="failed_delivery" target="q_package_sent" inscription="p:PACKAGE,I:ITEM LIST"/>
    </page>
    <variables/>
  </net>
</pnml>
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpn", required=True, type=Path)
    ap.add_argument("--profile", choices=["order-management"], default="order-management")
    ap.add_argument("--profile-json", type=Path, default=Path(__file__).with_name("order_management_profile.json"))
    ap.add_argument("--out-model", required=True, type=Path)
    ap.add_argument("--out-report", type=Path)
    args = ap.parse_args()

    profile = load_profile(args.profile_json)
    parsed = parse_cpn(args.cpn)
    present = {norm(t["name"]).lower() for t in parsed["transitions"]}
    expected = set(profile.get("cpn_transition_map", {}).keys())
    missing = sorted(x for x in expected if x.lower() not in present)

    report = {
        "source": str(args.cpn),
        "profile": profile["name"],
        "transition_count": len(parsed["transitions"]),
        "place_count": len(parsed["places"]),
        "missing_expected_business_transitions": missing,
        "cpn_transitions": parsed["transitions"],
        "semantic_choices": profile.get("notes", []),
        "guards_emitted": profile.get("guard_design", {}),
    }
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    args.out_model.write_text(order_management_pnml(profile), encoding="utf-8")
    out_report = args.out_report or args.out_model.with_suffix(".mapping.json")
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"DOPID:  {args.out_model}")
    print(f"Report: {out_report}")
    if missing:
        print("WARNING: expected CPN transitions not found:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
