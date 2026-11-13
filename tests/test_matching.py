"""Tests for two-round Hungarian peak matching in makeshift.spectra.matching.

Default tolerances are (0.03 1H, 0.3 15N) ppm; all synthetic offsets below
are along 1H only, so a 1H offset of k * 0.03 is k tolerance units.
"""

import numpy as np
import pandas as pd
import pytest

from makeshift.spectra import map_peaklists


def _left(h, n):
    return pd.DataFrame({"H_ppm": h, "N_ppm": n, "assn_label": [None] * len(h)})


def _right(h, n, labels):
    return pd.DataFrame({"H_ppm": h, "N_ppm": n, "assn_label": labels})


def _labels(out):
    return dict(zip(out["H_ppm"], out["assn_label"]))


def _rounds(out):
    return dict(zip(out["H_ppm"], out["match_round"]))


@pytest.fixture
def mixed_case():
    left = _left([8.00, 8.50, 9.00, 9.30, 7.50, 7.52],
                 [120.0, 115.0, 110.0, 125.0, 118.0, 118.0])
    right = _right(
        [8.000,   # A: exact -> round 1
         8.545,   # B: 1.5 tol off, isolated -> round 2
         9.045,   # C: 1.5 tol off, isolated -> round 2
         9.390,   # D: 3 tol off -> unmatched even in round 2
         7.551],  # E: within 2 tol of both 7.50 and 7.52 -> ambiguous, unmatched
        [120.0, 115.0, 110.0, 125.0, 118.0],
        ["A", "B", "C", "D", "E"],
    )
    return left, right


def test_two_rounds(mixed_case):
    left, right = mixed_case
    out, _ = map_peaklists(left, right, offset=(0, 0), how="left")

    labels, rounds = _labels(out), _rounds(out)
    assert labels[8.00] == "A" and rounds[8.00] == 1
    assert labels[8.50] == "B" and rounds[8.50] == 2
    assert labels[9.00] == "C" and rounds[9.00] == 2
    for h in (9.30, 7.50, 7.52):          # D too far; E ambiguous
        assert pd.isna(labels[h]) and np.isnan(rounds[h])
    assert not out["conflict"].any()


def test_tol2_none_disables_round2_and_enforces_hard_cutoff(mixed_case):
    left, right = mixed_case
    out, _ = map_peaklists(left, right, offset=(0, 0), how="left", tol2=None)

    # Only the exact match survives: B/C at 1.5 tol must NOT match in round 1
    # (the old cost matrix accepted pairs up to ~2 tol units).
    assert out["assn_label"].notna().sum() == 1
    assert _labels(out)[8.00] == "A"
    assert set(out["match_round"].dropna()) == {1.0}


def test_round2_blocked_by_round1_matched_neighbor():
    # Y is 1.17 tol from the free peak 8.08, but also within 2 tol of 8.00,
    # which was matched to X in round 1 -> Y stays unassigned.
    left = _left([8.00, 8.08], [120.0, 120.0])
    right = _right([8.000, 8.045], [120.0, 120.0], ["X", "Y"])
    out, _ = map_peaklists(left, right, offset=(0, 0), how="left")

    assert _labels(out)[8.00] == "X"
    assert pd.isna(_labels(out)[8.08])


def test_round2_needs_unique_left_candidate():
    # A single free left peak with two free right peaks nearby -> no match.
    left = _left([8.00], [120.0])
    right = _right([8.045, 7.955], [120.0, 120.0], ["P", "Q"])
    out, _ = map_peaklists(left, right, offset=(0, 0), how="left")
    assert pd.isna(out.loc[0, "assn_label"])


def test_custom_tol2_widens_round2():
    # 3 tol off: unmatched with default tol2 (2x), matched with tol2=4.
    left = _left([8.00], [120.0])
    right = _right([8.09], [120.0], ["Z"])
    out_default, _ = map_peaklists(left, right, offset=(0, 0), how="left")
    out_wide, _ = map_peaklists(left, right, offset=(0, 0), how="left", tol2=4)
    assert pd.isna(out_default.loc[0, "assn_label"])
    assert out_wide.loc[0, "assn_label"] == "Z"
    assert out_wide.loc[0, "match_round"] == 2


def test_outer_includes_match_round(mixed_case):
    left, right = mixed_case
    out, _ = map_peaklists(left, right, offset=(0, 0), how="outer")
    extra = out[out["assn_label"].isin(["D", "E"])]
    assert len(extra) == 2
    assert extra["match_round"].isna().all()
