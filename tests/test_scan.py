import numpy as np
import pytest

from agents.common import TrainData, write_private
from agents.scan import scan_on_train
from harness.build import LearnerError, build_learner, build_template, expand_ranges, substitute
from tests.conftest import synthetic_images

TEMPLATE = "/* rule: band at row P0, height P1 */\ndouble weak_score(const vj_image *im) { return vj_haar2_tb(im, 2, P0, 15, P1); }"


def test_expand_ranges_and_substitute():
    combos = expand_ranges({"P0": [0, 4, 2], "P2": [7, 8]})
    assert combos == [(0, 0, 7, 0), (0, 0, 8, 0), (2, 0, 7, 0), (2, 0, 8, 0), (4, 0, 7, 0), (4, 0, 8, 0)]
    for bad in ({}, {"P0": [3, 1]}, {"P0": [0, 1.5]}, {"P9": [0, 1]}, {"P0": [0, 30], "P1": [0, 30], "P2": [0, 2]}):
        with pytest.raises(LearnerError):
            expand_ranges(bad)
    assert substitute("f(P0, P1, xP2, P3)", (4, 5, 6, 7)) == "f(4, 5, xP2, 7)"


def test_templates_and_submissions_stay_separate(tmp_path):
    with pytest.raises(LearnerError, match="must use at least one"):
        build_template("double weak_score(const vj_image *im) { return 0; }", tmp_path / "a", tmp_path / "c")
    with pytest.raises(LearnerError, match="only in scan templates"):
        build_learner(TEMPLATE, tmp_path / "b", tmp_path / "c")
    with pytest.raises(LearnerError, match="vj_p"):     # a hand-written extern is still caught by nm
        build_learner("extern int vj_p[4];\ndouble weak_score(const vj_image *im) { return vj_p[0]; }",
                      tmp_path / "d", tmp_path / "c")


def test_scan_finds_the_planted_band(tmp_path):
    images, y = synthetic_images(150, 250, seed=3)   # faces: dark rows 5-7 over bright rows 8-10
    write_private(tmp_path / "private", images, y, np.ones(len(y)))
    data = TrainData.load(tmp_path / "private")
    report, best = scan_on_train(TEMPLATE, {"P0": [0, 12], "P1": [1, 5]}, data, tmp_path / "scan", tmp_path / "cache")
    assert "Scanned 65 placements" in report
    assert best is not None and "vj_haar2_tb(im, 2, 5, 15, 3)" in best
    build_learner(best, tmp_path / "submit", tmp_path / "cache")   # the concrete rule can be submitted
