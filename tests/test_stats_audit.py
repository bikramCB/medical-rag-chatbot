import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chatwithpdf" / "src"))

from stats_audit import audit_estimate


def test_ratio_interval_below_one_is_significant():
    result = audit_estimate("HR 0.70", "0.50 to 0.90", "p=0.01", "")
    assert result["measure"] == "ratio"
    assert result["crosses_null"] is False
    assert result["significant_by_ci"] is True
    assert result["direction"] == "below_null"
    assert result["significant_by_p"] is True
    assert result["ci_p_disagree"] is False


def test_interval_that_includes_the_null_is_not_significant():
    result = audit_estimate("HR 0.90 (0.70 to 1.20)", "", "", "The result was significant")
    assert result["crosses_null"] is True
    assert result["significant_by_ci"] is False
    assert result["spin"] is True


def test_missing_numbers_stay_empty():
    result = audit_estimate("the drug helped", "", "", "")
    assert result["estimate"] is None
    assert result["ci_low"] is None
    assert result["significant_by_ci"] is None
    assert result["spin"] is False
