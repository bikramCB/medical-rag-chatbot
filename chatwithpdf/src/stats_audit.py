"""Decide whether a printed estimate is significant from its interval and p-value.

Ratio estimates (HR, OR, RR) use a null of 1. Difference estimates use a null of 0.
A result is not called significant unless the interval stays entirely on one side of that null.
"""
import re

_ALPHA = 0.05
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‐": "-"})
_MEASURE = re.compile(r"\b(HR|OR|RR|IRR|MD|RD)\b", re.I)
_DIFFERENCE_WORDS = re.compile(r"\b(mean difference|risk difference|absolute difference|difference)\b", re.I)
_RATIO_WORDS = re.compile(r"\b(hazard ratio|odds ratio|risk ratio|rate ratio)\b", re.I)
_INTERVAL = re.compile(
    r"(?P<est>-?\d+(?:\.\d+)?)\s*"
    r"\(\s*(?:[^)]{0,40}?CI[^)]{0,20}?)?"
    r"(?P<lo>-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(?P<hi>-?\d+(?:\.\d+)?)\s*\)",
    re.I,
)
_LEADING_ESTIMATE = re.compile(r"\b(?:HR|OR|RR|IRR|MD|RD)\b\s*(?P<est>-?\d+(?:\.\d+)?)", re.I)
_ANY_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_BARE_RANGE = re.compile(r"(?P<lo>-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(?P<hi>-?\d+(?:\.\d+)?)")
_P_VALUE = re.compile(
    r"\bp\s*(?P<op><=|>=|<|>|=)?\s*(?P<value>\d*\.\d+|\d+(?:\.\d+)?)",
    re.I,
)
_NOT_SIGNIFICANT = re.compile(r"\b(?:not|non|no)[\s-]*significant\b", re.I)
_SIGNIFICANT = re.compile(r"\b(?:significant|significantly|superior)\b", re.I)
_PER_PROTOCOL = re.compile(r"\bper[- ]protocol\b", re.I)
_ITT = re.compile(r"\bintention[- ]to[- ]treat\b", re.I)


def _clean(text: str) -> str:
    return (text or "").translate(_DASHES)


def _float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def measure_kind(*parts: str):
    """Return 'ratio', 'difference', or None when the text does not name the estimate."""
    text = " ".join(part for part in parts if part)
    match = _MEASURE.search(text)
    if match:
        token = match.group(1).upper()
        if token in {"MD", "RD"}:
            return "difference"
        return "ratio"
    if _DIFFERENCE_WORDS.search(text):
        return "difference"
    if _RATIO_WORDS.search(text):
        return "ratio"
    return None


def null_value(kind):
    if kind == "ratio":
        return 1.0
    if kind == "difference":
        return 0.0
    return None


def parse_p_value(text: str):
    match = _P_VALUE.search(_clean(text))
    if not match:
        return None
    value = _float(match.group("value"))
    if value is None:
        return None
    op = match.group("op") or "="
    if op in {"<", "<="}:
        significant = value <= _ALPHA
    elif op in {">", ">="}:
        significant = False
    else:
        significant = value < _ALPHA
    return {"operator": op, "value": value, "significant": significant, "alpha": _ALPHA}


def claims_significance(text: str) -> bool:
    if not text or _NOT_SIGNIFICANT.search(text):
        return False
    return _SIGNIFICANT.search(text) is not None


def population_label(*parts: str):
    text = " ".join(part for part in parts if part)
    per_protocol = _PER_PROTOCOL.search(text) is not None
    intention = _ITT.search(text) is not None
    if per_protocol and not intention:
        return "per-protocol"
    if intention and not per_protocol:
        return "intention-to-treat"
    if per_protocol and intention:
        return "both"
    return ""


def _judgment(kind, estimate, low, high, p_info, wording: str) -> dict:
    null = null_value(kind)
    crosses = None
    significant_by_ci = None
    direction = ""
    if kind and estimate is not None and low is not None and high is not None and null is not None:
        left, right = min(low, high), max(low, high)
        crosses = left <= null <= right
        significant_by_ci = not crosses
        if estimate < null:
            direction = "below_null"
        elif estimate > null:
            direction = "above_null"
        else:
            direction = "on_null"
    significant_by_p = p_info["significant"] if p_info else None
    disagree = (
        significant_by_ci is not None
        and significant_by_p is not None
        and significant_by_ci != significant_by_p
    )
    spin = bool(crosses and claims_significance(wording))
    return {
        "measure": kind or "",
        "null": null,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "crosses_null": crosses,
        "significant_by_ci": significant_by_ci,
        "p_value": p_info,
        "significant_by_p": significant_by_p,
        "ci_p_disagree": disagree,
        "direction": direction,
        "spin": spin,
        "population": population_label(wording),
        "alpha": _ALPHA,
    }


def audit_estimate(effect: str = "", ci: str = "", p_value: str = "", quote: str = "") -> dict:
    """Audit one scorecard outcome. Missing pieces stay null instead of being guessed."""
    text = _clean(" ".join([effect or "", ci or "", p_value or "", quote or ""]))
    kind = measure_kind(effect, ci, quote)
    cleaned_effect = _clean(f"{effect} {ci}")
    found = _INTERVAL.search(cleaned_effect)
    if found is None:
        found = _INTERVAL.search(text)
    estimate = low = high = None
    if found:
        estimate = _float(found.group("est"))
        low = _float(found.group("lo"))
        high = _float(found.group("hi"))
    else:
        range_match = _BARE_RANGE.search(_clean(ci or ""))
        estimate_match = _LEADING_ESTIMATE.search(_clean(effect or ""))
        if range_match:
            low = _float(range_match.group("lo"))
            high = _float(range_match.group("hi"))
            if estimate_match:
                estimate = _float(estimate_match.group("est"))
            else:
                number = _ANY_NUMBER.search(_clean(effect or ""))
                estimate = _float(number.group(0)) if number else None
    return _judgment(kind, estimate, low, high, parse_p_value(p_value or text), f"{effect} {quote}")


def audit_claim(text: str) -> list:
    """Find every interval in a chat sentence and audit it."""
    cleaned = _clean(text or "")
    findings = []
    for match in _INTERVAL.finditer(cleaned):
        window = cleaned[max(0, match.start() - 40):match.end() + 40]
        kind = measure_kind(window)
        findings.append(_judgment(
            kind,
            _float(match.group("est")),
            _float(match.group("lo")),
            _float(match.group("hi")),
            parse_p_value(window),
            cleaned,
        ))
    if not findings and parse_p_value(cleaned):
        findings.append(_judgment(None, None, None, None, parse_p_value(cleaned), cleaned))
    return findings
