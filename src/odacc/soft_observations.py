from __future__ import annotations

import base64
from typing import Any, Iterable, List, Optional, Set, Tuple


def object_attr_var_name(obj: str, attr: str) -> str:
    enc = lambda x: base64.urlsafe_b64encode(str(x).encode("utf-8")).decode("ascii").rstrip("=")
    return f"__odacc_oa__{enc(obj)}__{enc(attr)}"


def install_soft_object_property_semantics() -> bool:
    """Patch CoCoMoT so observed object properties remain model-side variables.

    CoCoMoT normally substitutes an OCEL object property such as ``vip(o)`` by
    the recorded constant.  ODACC instead keeps the model-side value symbolic;
    the recorded value is compared with it separately in the conformance
    objective.  The patch is process-local and idempotent.
    """
    try:
        import objectcentric.encoding as oc_encoding
        from dpn.expr_utils import ObjectPropertyReplacer as BaseReplacer
        from dpn.expr import Var
    except Exception:
        return False

    current = getattr(oc_encoding, "ObjectPropertyReplacer", None)
    if getattr(current, "_odacc_soft_observations", False):
        return True

    class SoftObjectPropertyReplacer(BaseReplacer):
        _odacc_soft_observations = True

        def replace_arg(self, arg):
            unary_fun = lambda t: hasattr(t, "_args") and hasattr(t, "_name") and len(t._args) == 1
            if unary_fun(arg) and isinstance(arg._args[0], Var):
                obj = arg._args[0].name
                if isinstance(obj, str) and obj in self._objects:
                    ovmap = self._objects[obj].get("ovmap", {})
                    if arg._name in ovmap:
                        return Var(object_attr_var_name(obj, arg._name), None)
            return arg

    oc_encoding.ObjectPropertyReplacer = SoftObjectPropertyReplacer
    return True


def observed_object_attribute_terms(component, solver, *, relevant_attributes: Optional[Set[str]] = None):
    """Build soft model-vs-observation terms for current object attributes.

    The observation itself remains immutable.  A zero-cost explanation uses the
    same model-side value.  A different model-side value is allowed and costs 1.
    ``relevant_attributes`` can restrict the terms to attributes referenced by
    the selected candidate path (used by JODAP); the monolithic SMT backend can
    safely use all observed attributes because unselected transition guards are
    already conditionally activated by CoCoMoT's transition variables.
    """
    try:
        from dpn.expr import Expr
    except Exception:
        Expr = None

    terms: List[Any] = []
    metadata: List[Tuple[str, str, Any, Any, Any]] = []
    attrs = component.observation_formula.current_object_attributes()
    for obj in sorted(attrs):
        for attr, observed in sorted(attrs[obj].items()):
            if relevant_attributes is not None and attr not in relevant_attributes:
                continue
            var = solver.realvar(object_attr_var_name(obj, attr))
            if isinstance(observed, bool):
                encoded = int(observed)
            elif isinstance(observed, (int, float)):
                encoded = observed
            elif isinstance(observed, str) and Expr is not None:
                encoded = Expr.numval(observed)
            else:
                continue
            valexpr = solver.real(encoded)
            terms.append(solver.ite(solver.eq(var, valexpr), solver.num(0), solver.num(1)))
            metadata.append((obj, attr, observed, encoded, var))
    return terms, metadata
