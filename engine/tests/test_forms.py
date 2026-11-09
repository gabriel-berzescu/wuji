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
                cue = args.get("breath")
                assert cue in (None, "inhale", "exhale"), (name, cue)
                if cue is not None:  # the breath lasts the movement's slot, never less than the movement
                    assert args["breath_over"] >= args["over"], (name, args)
                body.intend(**{k: v for k, v in args.items() if k not in ("breath", "breath_over")})
            elif op == "tension":
                body.set_tension(args["region"], args["level"])
            elif op in ("announce", "say"):
                assert args["text"], name
            else:
                assert op == "breathe" and args["pattern"] in ("natural", "abdominal", "reverse", "hold")
        assert [t for t, _, _ in steps] == sorted(t for t, _, _ in steps), name  # played in order
    # each brocade is announced, and its movements speak with their breath, once each, while under way
    assert sum(op == "announce" for _, op, _ in FORMS["ba_duan_jin"]) == 8
    spoken = [(t, a) for t, op, a in FORMS["ba_duan_jin"] if op == "say"]
    assert len(spoken) > 60 and all(a["breath"] in ("inhale", "exhale") for _, a in spoken)
    starts = {t for t, op, a in FORMS["ba_duan_jin"] if op == "intend" and a.get("breath")}
    assert not starts & {t for t, _ in spoken}  # never at the very start of the movement
    assert _length(FORMS["ba_duan_jin"]) > 270  # all eight, five or six minutes
    # the bounce goes at the breath's pace: seven rises in about a minute, not fifteen seconds
    rises = [t for t, op, a in FORMS["bounce"] if op == "intend" and a.get("rise") == 1.0]
    assert len(rises) == 7 and 7.0 <= np.diff(rises).min()
    # every movement of a brocade (past its setting up) is led by its breath
    from wuji.forms import BROCADES
    for form in BROCADES:
        moves = [args for _, op, args in form() if op == "intend"]
        assert all("breath" in m for m in moves[1:-1] if "stance" not in m), form.__name__
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


def test_in_the_full_fold_the_back_lets_go(body):
    """Flexion-relaxation: near the end of the lumbar range its ligaments and fascia carry the trunk, the back muscles
    go nearly quiet; halfway down they still work."""
    from wuji.senses import region_effort
    back = lambda s: region_effort(s.act, body.a.f0, body.a.muscle_region == "lower_back")
    body.reset()
    settle(body, 3.5)
    body.intend(bend=0.5, arms="hang", over=2)
    half = settle(body, 3.0)
    body.intend(bend=1.0, over=2)
    full = settle(body, 3.0)
    assert body.lumbar_flexion() > 40 and body.lumbar_passive() > 40  # degrees; N m
    assert back(full) < 0.5 * back(half) and back(full) < 0.2
    body.intend(bend=0.0, over=2)
    settle(body, 2.5)
    assert body.lumbar_passive() < 1  # upright, the back of the spine is slack


def test_swaying_the_tail_the_hands_rest_on_the_thighs_from_the_start(body):
    """The home posture of Sway the head and shake the tail: a deep horse stance, the trunk inclined a little, both palms
    resting on the thighs and bearing some of the trunk, before any lean."""
    from wuji.forms import sway_tail
    first = next(a for _, op, a in sway_tail() if op == "intend")
    body.reset()
    settle(body, 3.5)
    body.intend(**{k: v for k, v in first.items() if k not in ("breath", "breath_over")})
    sol = settle(body, first["over"] + 1.5)
    assert set(sol.hands) == {"left", "right"}
    assert all(h["force"][2] > 5 for h in sol.hands.values())  # each palm bears something, upward on the hand
    body.intend(stance="wuji", sink=0.25, arms="hang", bend=0.0, over=3)
    settle(body, 4)


def test_the_passive_tissues_carry_what_people_hang_on(body):
    """Ligaments, capsules and stretched connective tissue (passive.py) carry what real bodies hang on: in quiet standing
    the soleus works gently (as EMG has it, 5-15%), in a horse stance the shins stay quiet and the work is in the thighs,
    and in the full fold the stretched hamstrings hold the hips, felt as stretch at the back of the thighs."""
    from wuji.senses import region_effort
    a = body.a
    act = lambda s, name: s.act[a.muscles.index(name)]
    body.reset()
    sol = settle(body, 4.0)
    assert 0.04 < act(sol, "soleus_r") < 0.2
    body.intend(stance="horse", sink=0.6, arms="embrace", over=4)
    sol = settle(body, 6.0)
    assert act(sol, "tibant_r") < 0.2 and region_effort(sol.act, a.f0, a.muscle_region == "thighs") > 0.1
    body.intend(stance="wuji", sink=0.1, arms="hang", over=4)
    settle(body, 5.0)
    body.intend(bend=1.0, over=4)
    sol = settle(body, 5.0)
    assert act(sol, "semimem_r") < 0.5
    hamstrings = [f for f in body.passive_felt if f.name.endswith("hamstrings")]
    assert all(f.moment > 10 and f.share > 0.15 for f in hamstrings)  # N m, holding each hip (folded ~60 degrees)
    body.intend(bend=0.0, over=3)
    settle(body, 3.5)
