"""The Eight Brocades: the new abilities they need (one-sided arms, the trunk, tiptoe) and the forms themselves."""
import mujoco
import numpy as np
import pytest

from wuji.anatomy import FORWARD, LEFTWARD
from wuji.body import Body
from wuji.forms import FORMS, _length


@pytest.fixture(scope="module")
def body():
    return Body(seed=3)


def settle(b: Body, seconds: float, dt: float = 0.04):
    for i in range(int(seconds / dt)):
        b.step(dt)
        if i % 6 == 0:
            b.solve_muscles(dt * 6)
    return b.solve_muscles(dt)


def world(b: Body, name: str) -> np.ndarray:
    d = mujoco.MjData(b.m)
    d.qpos[:] = b.q
    mujoco.mj_kinematics(b.m, d)
    return d.xpos[b.a.body[name]].copy()


def test_every_form_is_made_of_valid_intentions(body):
    for name, steps in FORMS.items():
        assert steps and _length(steps) > 0, name
        for _, op, args in steps:
            if op == "intend":
                body.intend(**args)
            elif op == "tension":
                body.set_tension(args["region"], args["level"])
            else:
                assert op == "breathe" and args["pattern"] in ("natural", "abdominal", "reverse", "hold")
    assert _length(FORMS["ba_duan_jin"]) > 240  # all eight, about five minutes
    body.reset()


def test_drawing_the_bow_one_arm_out_one_at_the_chest(body):
    body.reset()
    settle(body, 3.5)
    body.intend(arm_left="bow_extend", arm_right="bow_pull", twist=0.45, over=2)
    sol = settle(body, 3.0)
    mid = world(body, "pelvis") @ LEFTWARD
    left_out = world(body, "thirdmc_l") @ LEFTWARD - mid
    right_out = mid - world(body, "thirdmc_r") @ LEFTWARD
    assert left_out > 0.55  # the bow arm reaches out to the side
    assert right_out < 0.30  # the string hand is drawn in toward the chest
    assert sol.root_residual < 5
    body.intend(arms="hang", twist=0.0, over=1)  # both arms again: the one-arm shapes go
    assert body.posture.arm_left is None and body.posture.arm_right is None


def test_on_tiptoe_only_the_forefoot_carries_the_weight(body):
    body.reset()
    settle(body, 3.5)
    heel0 = world(body, "calcn_r")[2]
    body.intend(rise=1.0, over=1.5)
    sol = settle(body, 2.5)
    assert world(body, "calcn_r")[2] - heel0 > 0.04  # the heels are up
    assert all(body.loaded[s] == "forefoot" for s in ("left", "right"))
    lo, hi = body.posture_solver.heel_to_toe["right"]
    along = (sol.grf["right"]["cop"] @ FORWARD - lo) / (hi - lo)
    assert along > 0.55  # the pressure is over the ball of the foot
    assert sol.root_residual < 5  # muscles and floor alone hold it
    body.intend(rise=0.0, over=1)
    settle(body, 1.5)


def test_folding_forward_lowers_the_head_and_keeps_the_weight_over_the_feet(body):
    body.reset()
    settle(body, 3.5)
    head0 = world(body, "head")[2]
    body.intend(bend=0.9, arms="hang", over=3)
    sol = settle(body, 4.0)
    assert head0 - world(body, "head")[2] > 0.3
    assert sol.root_residual < 5
    assert set(sol.grf) == {"left", "right"}
    body.intend(bend=0.0, over=2)
    settle(body, 2.5)
