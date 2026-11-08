"""The forms: named sequences of intentions, with their breath.

Each step is (seconds after the form begins, operation, arguments), the operations those of the engine protocol (intend,
breathe, tension). The breath leads the movement: each movement carries its breath phase ("breath": "inhale" | "exhale"),
which begins with it and lasts as long: in while rising or opening, out while sinking, closing, or putting out force (the
bow drawn, the fist sent).

Ba Duan Jin, the Eight Brocades (八段錦), in the common modern sequence:
    1. Two hands hold up the heavens              lift_sky
    2. Drawing the bow to shoot the eagle         draw_bow
    3. Separating heaven and earth                separate
    4. The wise owl gazes backward                owl
    5. Sway the head and shake the tail           sway_tail
    6. Two hands hold the feet                    hold_feet
    7. Clench the fists and glare                 fists
    8. Bouncing on the toes                       bounce
Each movement is done a few times each side here (traditionally more). Limits of this body: the head has no joints of
its own, so the owl's gaze turns with the chest; the forward fold reaches about the knees to the shins, not the feet.
"""
from __future__ import annotations

Step = tuple[float, str, dict]

PACE = 4.0  # seconds per half-breath in the brocades: in for 4, out for 4


def _seq(*moves: tuple[float, dict]) -> list[Step]:
    """Intentions one after another, each lasting its duration."""
    out, t = [], 0.0
    for dur, args in moves:
        out.append((t, "intend", {**args, "over": dur}))
        t += dur
    return out


def _paced(steps: list[Step]) -> list[Step]:
    """Belly breathing, led by the movements' own breath cues."""
    return [(0.0, "breathe", {"pattern": "abdominal"})] + steps


IN, OUT = {"breath": "inhale"}, {"breath": "exhale"}


def _length(steps: list[Step]) -> float:
    return max(t + args.get("over", 0.0) for t, _, args in steps)


CENTRE = {"twist": 0.0, "side_bend": 0.0, "bend": 0.0, "rise": 0.0}


def lift_sky() -> list[Step]:
    up = {"arms": "lift_sky", "sink": 0.05, **IN}
    down = {"arms": "press_down", "sink": 0.3, **OUT}
    return _paced(_seq((3, {"stance": "wuji", "sink": 0.3, "arms": "hang", **CENTRE}), (PACE, up), (PACE, down), (PACE, up),
                       (PACE, down), (PACE, up), (PACE, {"arms": "hang", "sink": 0.25, **OUT})))


def draw_bow() -> list[Step]:
    left = {"arm_left": "bow_extend", "arm_right": "bow_pull", "twist": 0.45, "sink": 0.6, **OUT}
    right = {"arm_right": "bow_extend", "arm_left": "bow_pull", "twist": -0.45, "sink": 0.6, **OUT}
    home = {"arms": "embrace", "twist": 0.0, "sink": 0.45, **IN}
    return _paced(_seq((6, {"stance": "horse", "sink": 0.45, "arms": "embrace", **CENTRE}), (PACE, left), (PACE, home),
                       (PACE, right), (PACE, home), (PACE, left), (PACE, home), (PACE, right), (PACE, home),
                       (6, {"stance": "wuji", "sink": 0.25, "arms": "hang"})))


def separate() -> list[Step]:
    left = {"arm_left": "lift_sky", "arm_right": "earth_press", "sink": 0.1, **IN}
    right = {"arm_right": "lift_sky", "arm_left": "earth_press", "sink": 0.1, **IN}
    home = {"arms": "embrace", "sink": 0.3, **OUT}
    return _paced(_seq((3, {"stance": "wuji", "sink": 0.3, "arms": "embrace", **CENTRE}), (PACE, left), (PACE, home),
                       (PACE, right), (PACE, home), (PACE, left), (PACE, home), (PACE, right),
                       (PACE, {"arms": "hang", "sink": 0.25, **OUT})))


def owl() -> list[Step]:
    left = {"twist": 0.8, "arms": "owl", **IN}
    right = {"twist": -0.8, "arms": "owl", **IN}
    home = {"twist": 0.0, "arms": "hang", **OUT}
    return _paced(_seq((3, {"stance": "wuji", "sink": 0.25, "arms": "hang", **CENTRE}), (PACE, left), (PACE, home),
                       (PACE, right), (PACE, home), (PACE, left), (PACE, home), (PACE, right), (PACE, home)))


def sway_tail() -> list[Step]:
    left = {"side_bend": 0.8, "bend": 0.35, "twist": 0.3, "weight": 0.7, **OUT}
    right = {"side_bend": -0.8, "bend": 0.35, "twist": -0.3, "weight": 0.3, **OUT}
    home = {"side_bend": 0.0, "bend": 0.0, "twist": 0.0, "weight": 0.5, **IN}
    return _paced(_seq((6, {"stance": "horse", "sink": 0.6, "arms": "hands_thighs", **CENTRE}), (PACE, left), (PACE, home),
                       (PACE, right), (PACE, home), (PACE, left), (PACE, home), (PACE, right), (PACE, home),
                       (6, {"stance": "wuji", "sink": 0.25, "arms": "hang"})))


def hold_feet() -> list[Step]:
    up = {"arms": "lift_sky", "bend": 0.0, "sink": 0.05, **IN}
    fold = {"arms": "hang", "bend": 0.9, "sink": 0.1, **OUT}
    return _paced(_seq((3, {"stance": "wuji", "sink": 0.1, "arms": "hang", **CENTRE}), (PACE, up), (PACE + 1, fold),
                       (PACE + 1, up), (PACE + 1, fold), (PACE + 1, up), (PACE, {"arms": "hang", "sink": 0.25, **OUT})))


def fists() -> list[Step]:
    left = {"arm_left": "punch", "arm_right": "fist_waist", **OUT}
    right = {"arm_right": "punch", "arm_left": "fist_waist", **OUT}
    home = {"arms": "fist_waist", **IN}
    return _paced(_seq((6, {"stance": "horse", "sink": 0.55, "arms": "fist_waist", **CENTRE}), (PACE, left), (PACE, home),
                       (PACE, right), (PACE, home), (PACE, left), (PACE, home), (PACE, right), (PACE, home),
                       (6, {"stance": "wuji", "sink": 0.25, "arms": "hang"})))


def bounce() -> list[Step]:
    steps = [(0.0, "breathe", {"pattern": "natural"})]
    seq = [(3, {"stance": "wuji", "sink": 0.05, "arms": "hang", **CENTRE})]
    for _ in range(7):
        seq += [(1.6, {"rise": 1.0, **IN}), (0.6, {"rise": 0.0, **OUT})]  # up slowly on the inhale, down on the heels
    seq += [(3, {"sink": 0.25})]
    return steps + _seq(*seq)


BROCADES = (lift_sky, draw_bow, separate, owl, sway_tail, hold_feet, fists, bounce)


def ba_duan_jin() -> list[Step]:
    """The eight in order, from wuji to wuji."""
    out: list[Step] = []
    t = 0.0
    for form in BROCADES:
        steps = form()
        out += [(t + st, op, args) for st, op, args in steps]
        t += _length(steps) + 2.0  # a breath between the brocades
    out.append((t, "breathe", {"pattern": "natural"}))
    return out


def _settle() -> list[Step]:
    return [(0, "intend", {"stance": "wuji", "weight": 0.5, "sink": 0.25, "arms": "hang", **CENTRE, "over": 4}),
            (0, "breathe", {"pattern": "natural"}), (4, "tension", {"region": "all", "level": 0})]


def _embrace_tree() -> list[Step]:  # zhan zhuang: standing like a tree, holding a ball
    return [(0, "intend", {"stance": "wuji", "sink": 0.4, **CENTRE, "over": 4}), (4, "intend", {"arms": "embrace", "over": 5}),
            (9, "breathe", {"pattern": "abdominal"})]


FORMS = {
    "wuji": _settle(),
    "embrace_tree": _embrace_tree(),
    "lift_sky": lift_sky(),
    "draw_bow": draw_bow(),
    "separate": separate(),
    "owl": owl(),
    "sway_tail": sway_tail(),
    "hold_feet": hold_feet(),
    "fists": fists(),
    "bounce": bounce(),
    "ba_duan_jin": ba_duan_jin(),
}
