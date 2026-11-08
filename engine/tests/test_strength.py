"""The body's strength, checked against what healthy men of its size can do (anatomy.STRENGTH_TESTS)."""
import numpy as np
import pytest

from wuji.anatomy import STRENGTH, STRENGTH_TESTS, Anatomy, load_model, roots_of
from wuji.body import Body
from wuji.senses import region_effort


@pytest.fixture(scope="module")
def anatomy():
    return Anatomy(load_model())


@pytest.mark.parametrize("test", list(STRENGTH_TESTS))
def test_maximal_torques_sit_within_the_norms(anatomy, test):
    lo, hi = STRENGTH_TESTS[test][4]
    torque, _ = anatomy.max_torque(test)
    assert lo <= torque <= hi, f"{test}: {torque:.0f} N m, norm {lo:.0f}-{hi:.0f}"


def test_calibration_scales_active_and_passive_force_together(anatomy):
    published = load_model(calibrated=False)
    for group, (roots, factor, _why) in STRENGTH.items():
        idx = roots_of(roots, anatomy.muscles)
        assert np.allclose(anatomy.m.actuator_gainprm[idx, 2], factor * published.actuator_gainprm[idx, 2])
        assert np.allclose(anatomy.m.actuator_biasprm[idx, 2], factor * published.actuator_biasprm[idx, 2])
        assert np.allclose(anatomy.f0[idx], anatomy.m.actuator_gainprm[idx, 2])  # strength weighting sees the new forces
    # lengths are untouched: the passive onsets calibrate_rest moves, and the curves' shapes
    assert np.array_equal(anatomy.m.actuator_gainprm[:, [0, 1, 3, 4, 5, 6, 7, 8]],
                          published.actuator_gainprm[:, [0, 1, 3, 4, 5, 6, 7, 8]])
    assert np.array_equal(anatomy.m.actuator_lengthrange, published.actuator_lengthrange)


def test_rising_onto_the_toes_reads_like_emg():
    """Both heels up: about 1000 N through each Achilles tendon, which surface EMG puts at roughly a quarter to two fifths of
    a maximal contraction in the calves. With the published strengths the calves read about 10%."""
    body = Body(seed=7)
    body.intend(stance="wuji", sink=0.05, arms="hang", rise=1.0, over=2)
    for i in range(75):
        body.step(0.08)
        if i % 3 == 0:
            body.solve_muscles(0.24)
    sol = body.solve_muscles(0.25)
    calves = region_effort(sol.act, body.a.f0, body.a.muscle_region == "calves")
    assert 0.15 < calves < 0.45
