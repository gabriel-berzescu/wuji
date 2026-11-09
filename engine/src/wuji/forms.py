"""The forms: named sequences of intentions, with their breath.

Each step is (seconds after the form begins, operation, arguments), the operations those of the engine protocol (intend,
breathe, tension), "announce" {"text"}: what the form says as a brocade begins, or "say" {"text", "breath"}: what a
movement is. The engine calls the practitioner with each, a movement's joined to its breath ("In, the arms rise to hold up
the heavens."), so someone woken by the form's signals is present at every movement (Gabriel: "87 events is not too much
... all things must pass"). A movement speaks halfway through, when it is under way, so the felt sense that comes with the
call is the movement's own, not where the last one ended.

The breath leads the movement: each movement carries its breath phase ("breath": "inhale" | "exhale"), which begins with
it and lasts its slot: in while rising or opening, out while sinking, closing, or putting out force (the bow drawn, the
fist sent). A movement can be quicker than its breath (the heels drop in a moment while the out-breath goes on).

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


SPEAK = 0.5  # a movement speaks this far through itself: under way, the body already in it


def _seq(*moves: tuple[float, dict]) -> list[Step]:
    """Intentions one after another, each in its slot (seconds). The movement lasts the slot unless it gives its own
    "over"; its breath lasts the slot. A movement that says what it is speaks SPEAK of the way through."""
    out, t = [], 0.0
    for slot, args in moves:
        args = {"over": slot, **args}
        say = args.pop("say", None)
        if "breath" in args:
            args["breath_over"] = slot
        out.append((t, "intend", args))
        if say:
            out.append((t + SPEAK * args["over"], "say", {"text": say, "breath": args.get("breath")}))
        t += slot
    return out


def _paced(steps: list[Step]) -> list[Step]:
    """Belly breathing, led by the movements' own breath cues."""
    return [(0.0, "breathe", {"pattern": "abdominal"})] + steps


IN, OUT = {"breath": "inhale"}, {"breath": "exhale"}


def _announced(text: str, steps: list[Step]) -> list[Step]:
    return [(0.0, "announce", {"text": text})] + steps


def _length(steps: list[Step]) -> float:
    return max(t + max(args.get("over", 0.0), args.get("breath_over", 0.0)) for t, _, args in steps)


CENTRE = {"twist": 0.0, "side_bend": 0.0, "bend": 0.0, "rise": 0.0}


def lift_sky() -> list[Step]:
    up = {"arms": "lift_sky", "sink": 0.05, **IN, "say": "the arms rise to hold up the heavens"}
    down = {"arms": "press_down", "sink": 0.3, **OUT, "say": "the hands press down before the body"}
    return _announced("Two hands hold up the heavens: in as the arms rise, out as they press down; three times.",
                      _paced(_seq((3, {"stance": "wuji", "sink": 0.3, "arms": "hang", **CENTRE}), (PACE, up), (PACE, down),
                                  (PACE, up), (PACE, down), (PACE, up),
                                  (PACE, {"arms": "hang", "sink": 0.25, **OUT, "say": "the arms settle"}))))


def draw_bow() -> list[Step]:
    left = {"arm_left": "bow_extend", "arm_right": "bow_pull", "twist": 0.45, "sink": 0.6, **OUT,
            "say": "drawing the bow to the left, the eyes on the raised finger"}
    right = {"arm_right": "bow_extend", "arm_left": "bow_pull", "twist": -0.45, "sink": 0.6, **OUT,
             "say": "drawing the bow to the right, the eyes on the raised finger"}
    home = {"arms": "embrace", "twist": 0.0, "sink": 0.45, **IN, "say": "the hands come back before the chest"}
    return _announced("Drawing the bow to shoot the eagle: into horse stance; out as the bow is drawn, in returning; left, "
                      "right, twice.",
                      _paced(_seq((6, {"stance": "horse", "sink": 0.45, "arms": "embrace", **CENTRE}), (PACE, left),
                                  (PACE, home), (PACE, right), (PACE, home), (PACE, left), (PACE, home), (PACE, right),
                                  (PACE, home), (6, {"stance": "wuji", "sink": 0.25, "arms": "hang"}))))


def separate() -> list[Step]:
    left = {"arm_left": "lift_sky", "arm_right": "earth_press", "sink": 0.1, **IN,
            "say": "the left hand rises to heaven, the right presses the earth"}
    right = {"arm_right": "lift_sky", "arm_left": "earth_press", "sink": 0.1, **IN,
             "say": "the right hand rises to heaven, the left presses the earth"}
    home = {"arms": "embrace", "sink": 0.3, **OUT, "say": "the hands come back together"}
    return _announced("Separating heaven and earth: in as one hand rises and the other presses down, out returning.",
                      _paced(_seq((3, {"stance": "wuji", "sink": 0.3, "arms": "embrace", **CENTRE}), (PACE, left),
                                  (PACE, home), (PACE, right), (PACE, home), (PACE, left), (PACE, home), (PACE, right),
                                  (PACE, {"arms": "hang", "sink": 0.25, **OUT, "say": "the arms settle"}))))


def owl() -> list[Step]:
    left = {"twist": 0.8, "arms": "owl", **IN, "say": "turning to look back over the left shoulder"}
    right = {"twist": -0.8, "arms": "owl", **IN, "say": "turning to look back over the right shoulder"}
    home = {"twist": 0.0, "arms": "hang", **OUT, "say": "turning back to the front"}
    return _announced("The wise owl gazes backward: in as the body turns to look behind, out returning.",
                      _paced(_seq((3, {"stance": "wuji", "sink": 0.25, "arms": "hang", **CENTRE}), (PACE, left),
                                  (PACE, home), (PACE, right), (PACE, home), (PACE, left), (PACE, home), (PACE, right),
                                  (PACE, home))))


def sway_tail() -> list[Step]:
    left = {"side_bend": 0.8, "bend": 0.35, "twist": 0.3, "weight": 0.7, **OUT,
            "say": "leaning and turning to the left, the tailbone swinging the other way"}
    right = {"side_bend": -0.8, "bend": 0.35, "twist": -0.3, "weight": 0.3, **OUT,
             "say": "leaning and turning to the right, the tailbone swinging the other way"}
    # a deep horse stance, the hands resting on the thighs above the knees and the trunk inclined a little forward to
    # reach them (upright, the palms hang 15 cm short: they rest and bear only from about 15 degrees of incline)
    home = {"side_bend": 0.0, "bend": 0.15, "twist": 0.0, "weight": 0.5, **IN, "say": "back to the centre"}
    return _announced("Sway the head and shake the tail: horse stance, hands on the thighs; out leaning, in returning.",
                      _paced(_seq((6, {"stance": "horse", "sink": 0.75, "arms": "hands_thighs", **CENTRE, "bend": 0.15}),
                                  (PACE, left),
                                  (PACE, home), (PACE, right), (PACE, home), (PACE, left), (PACE, home), (PACE, right),
                                  (PACE, home), (6, {"stance": "wuji", "sink": 0.25, "arms": "hang", **CENTRE}))))


def hold_feet() -> list[Step]:
    up = {"arms": "lift_sky", "bend": 0.0, "sink": 0.05, **IN, "say": "rising, the arms reaching up"}
    # the full fold: the lumbar spine near the end of its range, where its ligaments carry the trunk and the back lets go
    fold = {"arms": "hang", "bend": 1.0, "sink": 0.1, **OUT, "say": "folding forward, the hands sliding down toward the feet"}
    return _announced("Two hands hold the feet: in reaching up, out folding down.",
                      _paced(_seq((3, {"stance": "wuji", "sink": 0.1, "arms": "hang", **CENTRE}), (PACE, up),
                                  (PACE + 1, fold), (PACE + 1, up), (PACE + 1, fold), (PACE + 1, up),
                                  (PACE, {"arms": "hang", "sink": 0.25, **OUT, "say": "the arms settle"}))))


def fists() -> list[Step]:
    left = {"arm_left": "punch", "arm_right": "fist_waist", **OUT, "say": "the left fist goes out, the eyes wide"}
    right = {"arm_right": "punch", "arm_left": "fist_waist", **OUT, "say": "the right fist goes out, the eyes wide"}
    home = {"arms": "fist_waist", **IN, "say": "the fist comes back to the waist"}
    return _announced("Clench the fists and glare: horse stance; out with each fist sent forward, in as it returns.",
                      _paced(_seq((6, {"stance": "horse", "sink": 0.55, "arms": "fist_waist", **CENTRE}), (PACE, left),
                                  (PACE, home), (PACE, right), (PACE, home), (PACE, left), (PACE, home), (PACE, right),
                                  (PACE, home), (6, {"stance": "wuji", "sink": 0.25, "arms": "hang"}))))


def bounce() -> list[Step]:
    # one bounce to a breath, like every other movement here: in rising slowly onto the toes, the crown lifting, a moment
    # at the top as the in-breath ends; out as the heels drop (quickly: a light jolt up through the body), the out-breath
    # going on while the body settles. (It was 1.6 s up and 0.6 s down on the natural breath, 27 a minute: Gabriel asked
    # why so fast, and it was not led by the breath.)
    up = {"rise": 1.0, "over": PACE - 0.5, **IN, "say": "up on the toes, the crown lifting"}
    drop = {"rise": 0.0, "over": 0.4, **OUT, "say": "the heels drop, a light jolt up through the body"}
    seq = [(3, {"stance": "wuji", "sink": 0.05, "arms": "hang", **CENTRE})]
    for _ in range(7):
        seq += [(PACE, up), (PACE, drop)]
    seq += [(3, {"sink": 0.25})]
    return _announced("Bouncing on the toes: in rising slowly onto the toes, out as the heels drop; seven times, a breath "
                      "each.", _paced(_seq(*seq)))


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
    return [(0, "announce", {"text": "Settling into wuji: feet hip-width, knees soft, everything released."}), (0, "intend", {"stance": "wuji", "weight": 0.5, "sink": 0.25, "arms": "hang", **CENTRE, "over": 4}),
            (0, "breathe", {"pattern": "natural"}), (4, "tension", {"region": "all", "level": 0})]


def _embrace_tree() -> list[Step]:  # zhan zhuang: standing like a tree, holding a ball
    return [(0, "announce", {"text": "Standing like a tree: sinking, the arms rounding to hold the ball, the breath in the belly."}),
            (0, "intend", {"stance": "wuji", "sink": 0.4, **CENTRE, "over": 4}), (4, "intend", {"arms": "embrace", "over": 5}),
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
