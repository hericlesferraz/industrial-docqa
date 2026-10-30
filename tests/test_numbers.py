"""Number readings by locale, shared by the verifier and the eval (DECISIONS #7)."""

import pytest

from docqa.text.numbers import numbers, pool, readings, ungrounded


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("9500", {9500.0}),
        ("9,500", {9500.0, 9.5}),
        ("13.000", {13000.0, 13.0}),
        ("1.800", {1800.0, 1.8}),
        ("1,800.5", {1800.5}),
        ("1.234,5", {1234.5}),
        ("0.30", {0.3}),
        ("22 000", {22000.0}),
        ("12.5.3", {12.0, 5.0, 3.0}),
    ],
)
def test_readings(token: str, expected: set[float]) -> None:
    """Each token gets every reading its separators allow, and only those.

    Args:
        token: Numeric token.
        expected: Its readings.
    """
    assert readings(token) == expected


def test_locale_pairs_ground_each_other() -> None:
    """`9,500` matches `9500Hrs.` and `13,000` matches `13.000 horas`, in both directions."""
    assert ungrounded("9,500 hours", ["|Over 210 to 280 incl.(180)|3600Hrs.|9500Hrs.|"]) == []
    assert ungrounded("13,000 hours", ["| CLP PG | 13.000 horas |"]) == []
    assert ungrounded("13.000 horas", ["13,000 hours"]) == []


def test_converted_value_is_rejected() -> None:
    """A unit conversion the source doesn't print fails: `48.8 m` against `160 ft (49m)`."""
    source = "Maximum submergence of motor is not to exceed 160 ft (49m) in depth"
    assert ungrounded("about 48.8 m", [source]) == ["48.8"]
    assert ungrounded("160 ft (49 m)", [source]) == []


def test_request_labels_are_not_numbers() -> None:
    """`S12` in an answer is a label, not the number 12; digits inside codes still count."""
    assert numbers("See S12 and S3.") == []
    assert [tok for tok, _ in numbers("IP55 and 6205-2RS")] == ["55", "6205", "2"]
    assert ungrounded("as S4 says, 24 months", ["12 months"]) == ["24"]


def test_groups_without_a_joined_reading_are_separate_numbers() -> None:
    """Numbers next to each other each need a source: the grounded one never vouches for its neighbour."""
    assert numbers("Frame 210 to 280 12000 h") == [("210", {210.0}), ("280", {280.0}), ("12000", {12000.0})]
    assert ungrounded("Frame 210 to 280 12000 h", ["Over 210 to 280 incl.(180)"]) == ["12000"]
    assert numbers("22 000 h") == [("22 000", {22000.0})]


def test_question_numbers_and_full_width_digits() -> None:
    """Numbers come from any source text, and NFKC folds full-width digits."""
    assert ungrounded("At 1800 RPM: 9500 hours", ["9500Hrs.", "frame 256 at 1800 RPM?"]) == []
    assert pool(["１８００"]) == {1800.0}
