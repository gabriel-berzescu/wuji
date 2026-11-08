"""The body: standing, shifting, stepping and holding, checked against what a real body does."""
import numpy as np
import pytest

from wuji.anatomy import REGION_ORDER
from wuji.body import Body
from wuji.senses import Senses, region_effort


@pytest.fixture(scope="module")
def body():
    return Body(seed=1)


def settle(b: Body, seconds: float, dt: float = 0.04):
    for i in range(int(seconds / dt)):
        b.step(dt)
        if i % 6 == 0:
            b.solve_muscles(dt * 6)
    return b.solve_muscles(dt)


def share_left(sol) -> float:
    l, r = sol.grf.get("left", {}).get("force", [0, 0, 0])[2], sol.grf.get("right", {}).get("force", [0, 0, 0])[2]
    return l / (l + r)


def test_every_muscle_has_a_region(body):
    assert body.m.nu == 416
    assert set(body.a.muscle_region) == set(REGION_ORDER)
    assert (body.a.muscle_side == "left").sum() == (body.a.muscle_side == "right").sum() == 208


def test_quiet_standing_is_cheap_and_carried_by_the_calves(body):
    body.reset()
    sol = settle(body, 4.0)
    weight = body.m.body_mass.sum() * 9.81
    vertical = sum(g["force"][2] for g in sol.grf.values())
    assert vertical == pytest.approx(weight, rel=0.03)  # the ground carries the body
    assert share_left(sol) == pytest.approx(0.5, abs=0.06)
    assert sol.root_residual < 1.0  # nothing else holds it up
    f0, reg = body.a.f0, body.a.muscle_region
    calves = region_effort(sol.act, f0, reg == "calves")
    back = region_effort(sol.act, f0, reg == "lower_back")
    assert 0.01 < calves < 0.15  # the soleus works gently in quiet standing: ~15 N m per ankle, a tenth of its strength
    assert back < 0.08  # the back is nearly quiet


def test_shifting_the_weight_loads_the_left_leg(body):
    body.reset()
    settle(body, 3.0)
    body.intend(weight=0.8, over=2.0)
    sol = settle(body, 3.0)
    assert share_left(sol) > 0.65


def test_stepping_into_horse_lifts_each_foot_and_lands_wider(body):
    body.reset()
    settle(body, 3.5)
    body.intend(stance="horse", sink=0.5, over=6.0)
    lifted = set()
    unloaded = set()
    dt = 0.04
    for i in range(int(7.0 / dt)):
        body.step(dt)
        for side, on in body.loaded.items():
            if not on:
                lifted.add(side)
        if i % 5 == 0:
            sol = body.solve_muscles(dt * 5)
            for side in ("left", "right"):
                if not body.loaded[side]:
                    assert side not in sol.grf  # a lifted foot carries nothing
                    unloaded.add(side)
    assert lifted == {"left", "right"} and unloaded == {"left", "right"}
    feet = body.current["feet"]
    assert feet["left"] - feet["right"] == pytest.approx(0.40, abs=0.01)
    sol = settle(body, 2.0)
    assert share_left(sol) == pytest.approx(0.5, abs=0.12)
    knee = np.degrees(body.q[body.a.qadr["knee_angle_r"]])
    assert knee > 30  # a horse stance sits low


def test_held_tension_costs_more_than_the_posture_needs(body):
    body.reset()
    settle(body, 3.0)
    body.set_tension("shoulders", 0.8)
    sol = settle(body, 1.5)
    f0, reg = body.a.f0, body.a.muscle_region
    held = region_effort(sol.act, f0, reg == "shoulders")
    needed = region_effort(body.minimum.act, f0, reg == "shoulders")
    assert held > needed + 0.05
    body.set_tension("all", 0.0)


def test_the_felt_sense_speaks(body):
    body.reset()
    settle(body, 3.0)
    sense = Senses(body)
    for _ in range(10):
        body.step(0.04)
        sense.note_sway()
    out = sense.read()
    assert out["text"].startswith("Wuji stance")
    assert "Weight" in out["text"]
    assert set(out["body"]["effort"]["regions"]) == set(REGION_ORDER)


def test_bad_intentions_are_refused(body):
    with pytest.raises(ValueError):
        body.intend(stance="crane")
    with pytest.raises(ValueError):
        body.intend(weight=1.5)
    with pytest.raises(ValueError):
        body.set_tension("ears", 0.5)


def test_the_ambient_sense_habituates(body):
    """Injected before every thought, the felt sense says only what changed, as a body habituates: the whole picture the
    first time, one quiet line while nothing changes, then only the change, and a word when something lets go."""
    body.reset()
    settle(body, 4.0)  # arrived: no longer moving
    sense = Senses(body)
    first = sense.read(detail="ambient")["text"].splitlines()
    assert first[0].startswith("Wuji stance") and len(first) >= 3  # nothing noticed yet: the whole picture
    for _ in range(5):
        body.step(0.04)
        sense.note_sway()
    quiet = sense.read(detail="ambient")
    assert quiet["text"] == "Wuji stance as before; nothing new." and quiet["changed"] == []
    body.intend(weight=0.8, over=1.0)
    settle(body, 2.0)
    shifted = sense.read(detail="ambient")
    assert "weight" in shifted["changed"] and "Weight" in shifted["text"]
    assert "stance" not in shifted["changed"]  # the stance did not change, so it is not repeated
    body.set_tension("shoulders", 0.8)
    settle(body, 1.5)
    assert "Holding the shoulders" in sense.read(detail="ambient")["text"]
    body.set_tension("shoulders", 0.0)
    settle(body, 1.5)
    assert "The shoulders let go." in sense.read(detail="ambient")["text"]
    # an explicit sense counts as noticing everything
    sense.read(detail="brief")
    assert sense.read(detail="ambient")["text"] == "Wuji stance as before; nothing new."


def test_the_bellys_pressure_eases_the_lower_back(body):
    """Abdominal pressure gives the lumbar spine an extension moment: the back's extensors need to do a little less."""
    body.reset()
    body.intend(sink=0.6, arms="embrace", tailbone=1.0, over=2.0)
    settle(body, 4.0)
    reg = body.a.muscle_region == "lower_back"
    body.abdominal_pressure = 8.0
    rest = region_effort(settle(body, 1.5).act, body.a.f0, reg)
    body.abdominal_pressure = 30.0
    pressed = region_effort(settle(body, 1.5).act, body.a.f0, reg)
    body.abdominal_pressure = 8.0
    assert pressed < rest - 0.005


def test_the_diaphragm_works_with_the_rest(body):
    """The breath's own muscle among the working regions, at each breath's peak (the cycle itself is not change)."""
    body.reset()
    settle(body, 3.0)
    sense = Senses(body)
    for a in (0.05, 0.45, 0.1, 0.0, 0.05):  # one deep breath: up, then down to nothing
        sense.note_breath(a, 0.5)
    out = sense.read(breath_state={"diaphragm": 0.0, "pattern": "abdominal"}, breath_felt=["Exhale."])
    assert out["body"]["effort"]["regions"]["diaphragm"] > 0.35
    assert "diaphragm" in out["text"]
