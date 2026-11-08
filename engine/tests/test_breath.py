"""The breath's physiology, checked against the numbers the module cites."""
import json
import statistics as st

import pytest

from wuji.breath import Breath

DT = 0.01


def run(b: Breath, secs: float, dt: float = DT, **kw) -> tuple[list[dict], list[dict]]:
    states, events = [], []
    for _ in range(int(round(secs / dt))):
        events += b.step(dt, **kw)
        states.append(b.state())
    return states, events


def breaths(states: list[dict]) -> list[list[dict]]:
    """Split a run into whole breaths, each starting at an inhale."""
    out, cur = [], None
    for prev, s in zip(states, states[1:]):
        if s["phase"] == "inhale" and prev["phase"] != "inhale":
            if cur:
                out.append(cur)
            cur = []
        if cur is not None:
            cur.append(s)
    return out


def end_of_inhale(states: list[dict]) -> list[dict]:
    return [a for a, b in zip(states, states[1:]) if a["phase"] == "inhale" and b["phase"] != "inhale"]


def swing(states: list[dict]) -> float:
    """Mean peak-to-trough heart rate per breath."""
    return st.mean(max(s["heart_bpm"] for s in br) - min(s["heart_bpm"] for s in br) for br in breaths(states))


def settle(pattern="natural", seed=0, warm=90.0, **kw) -> Breath:
    b = Breath(seed)
    b.set_pattern(pattern, **kw)
    run(b, warm)
    return b


# ---------------------------------------------------------------- natural breathing


def test_natural_tidal_rate_and_gases():
    b = settle()
    states, events = run(b, 240)
    brs = breaths(states)
    tidals = [br[0]["tidal_l"] for br in brs]
    rate = 60 * len(brs) / (len(states) * DT)
    assert 0.4 <= st.median(tidals) <= 0.65
    assert 11 <= rate <= 16
    co2 = [s["co2_mmhg"] for s in states]
    assert 37.5 <= st.mean(co2) <= 42.0
    assert 96.5 <= st.mean(s["o2_sat"] for s in states) <= 98.5
    assert 55 <= st.mean(s["heart_bpm"] for s in states) <= 75
    vols = [s["lung_volume_l"] for s in states]
    assert 2.4 <= min(vols) <= 3.0                      # FRC
    assert max(s["urge"] for s in states) < 0.1
    assert events == []                                 # events are rare: none in quiet breathing


def test_natural_breath_varies_but_is_not_chaotic():
    states, _ = run(settle(), 240)
    brs = breaths(states)
    tidals = [br[0]["tidal_l"] for br in brs]
    periods = [len(br) * DT for br in brs]
    cv_t = st.pstdev(tidals) / st.mean(tidals)
    cv_p = st.pstdev(periods) / st.mean(periods)
    assert 0.04 < cv_t < 0.5 and 0.04 < cv_p < 0.4     # not a metronome, not chaos


def test_quiet_diaphragm_and_partition():
    states, _ = run(settle(), 120)
    tops = [s for s in end_of_inhale(states) if s["tidal_l"] < 0.8]   # leave out sighs
    assert 0.6 <= st.median(s["diaphragm_descent_cm"] for s in tops) <= 2.0
    assert all(s["ribcage_share"] > 0.5 for s in tops)                 # upright: the rib cage leads quietly
    assert all(s["belly_cm"] > 0 and s["chest_cm"] > 0 for s in tops)


# ---------------------------------------------------------------- patterns


def test_paced_breath_follows_the_pacing():
    b = settle("abdominal", inhale=4, exhale=6, warm=60)
    states, _ = run(b, 120)
    brs = breaths(states)
    for br in brs:
        assert len(br) * DT == pytest.approx(10.0, abs=0.05)
        assert sum(s["phase"] == "inhale" for s in br) * DT == pytest.approx(4.0, abs=0.05)
    assert states[-1]["rate_bpm"] == pytest.approx(6.0)
    tidal = st.mean(br[0]["tidal_l"] for br in brs)
    assert 0.8 <= tidal <= 1.4                          # slow breathing: larger tides
    assert 37 <= st.mean(s["co2_mmhg"] for s in states) <= 41
    # smooth: flow never jumps between steps
    flows = [s["flow_lps"] for s in states]
    assert max(abs(a - b) for a, b in zip(flows, flows[1:])) < 0.05


def test_abdominal_belly_leads_reverse_draws_it_in():
    ab = end_of_inhale(run(settle("abdominal", inhale=4, exhale=6), 60)[0])
    rv = end_of_inhale(run(settle("reverse", inhale=4, exhale=6), 60)[0])
    nat = end_of_inhale(run(settle(), 60)[0])
    for s in ab:
        assert s["abdomen_share"] > 0.6 and s["belly_cm"] > 2.0
        assert s["diaphragm_descent_cm"] > 2.5
    for s in rv:
        assert s["abdomen_share"] < 0 and s["ribcage_share"] > 1.0
        assert s["belly_cm"] < -0.3 and s["chest_cm"] > 1.5
    # the diaphragm descends against drawn-in abdominals: belly pressure climbs
    assert st.mean(s["abdominal_pressure_cmh2o"] for s in rv) > st.mean(s["abdominal_pressure_cmh2o"] for s in ab) + 3
    assert st.mean(s["abdomen_share"] for s in nat) < st.mean(s["abdomen_share"] for s in ab)


def test_deep_belly_breath_descends_far():
    tops = end_of_inhale(run(settle("abdominal", inhale=6, exhale=9), 60)[0])
    assert all(s["tidal_l"] > 1.1 for s in tops)
    assert max(s["diaphragm_descent_cm"] for s in tops) > 3.5
    assert max(s["diaphragm"] for s in tops) > max(s["diaphragm"] for s in end_of_inhale(run(settle(), 30)[0])
                                                    if s["tidal_l"] < 0.8)


def test_belly_tension_shifts_breath_to_the_ribs():
    loose = end_of_inhale(run(settle("abdominal", inhale=4, exhale=6), 40)[0])
    b = settle("abdominal", inhale=4, exhale=6)
    run(b, 10, belly_tension=0.8)
    tight = end_of_inhale(run(b, 40, belly_tension=0.8)[0])
    mean = lambda xs, k: st.mean(s[k] for s in xs)
    assert mean(tight, "abdomen_share") < mean(loose, "abdomen_share") - 0.3
    assert mean(tight, "ribcage_share") > mean(loose, "ribcage_share") + 0.3
    assert mean(tight, "chest_cm") > mean(loose, "chest_cm")
    assert mean(tight, "belly_cm") < mean(loose, "belly_cm")
    assert mean(tight, "diaphragm_descent_cm") < mean(loose, "diaphragm_descent_cm")
    assert mean(tight, "abdominal_pressure_cmh2o") > mean(loose, "abdominal_pressure_cmh2o") + 8


# ---------------------------------------------------------------- holds and gases


def test_hold_raises_co2_and_urge_then_settles():
    b = settle()
    line = b.set_pattern("hold")
    assert "hold" in line.lower()
    states, events = [], []
    while not any(e["data"]["kind"] == "urge" for e in events):
        s, e = run(b, 1.0)
        states += s
        events += e
        assert len(states) * DT < 120, "no urge event"
    held = [s for s in states if s["phase"] == "hold"]
    assert held and held[-1]["co2_mmhg"] > held[0]["co2_mmhg"] + 4
    assert held[-1]["urge"] > held[0]["urge"]
    assert held[-1]["heart_bpm"] < held[0]["heart_bpm"] + 5   # no racing heart: a mild diving slowing
    assert [e["data"]["kind"] for e in events] == ["urge"]
    ev = events[0]
    assert ev["event"] == "breath" and 0.5 <= ev["data"]["urge"] < 1.0 and ev["data"]["text"]

    b.set_pattern("natural")                            # release
    assert b.state()["phase"] != "hold"
    states, events = run(b, 120)
    kinds = [e["data"]["kind"] for e in events]
    assert kinds == ["settled"]
    assert min(s["co2_mmhg"] for s in states) < 40      # the breath blows the CO2 off
    assert states[-1]["urge"] < 0.05


def test_co2_climbs_a_few_mmhg_per_minute_faster_with_effort():
    def held(effort):
        b = settle("natural", inhale=4, exhale=6, warm=60)     # about 1 L breaths
        run(b, 30, effort=effort)
        b.set_pattern("hold", hold=60)
        return [s for s in run(b, 62, effort=effort)[0] if s["phase"] == "hold"]

    def slope(states, a, z):                                    # mmHg per minute between a and z seconds
        return (states[int(z / DT)]["co2_mmhg"] - states[int(a / DT)]["co2_mmhg"]) / (z - a) * 60

    rest, work = held(0.0), held(0.6)
    assert len(rest) * DT == pytest.approx(60, abs=0.1)        # a minute's hold at rest, from a full slow breath
    assert rest[int(15 / DT)]["co2_mmhg"] > rest[0]["co2_mmhg"] + 2     # quick first rise toward venous CO2
    assert 2.0 <= slope(rest, 30, 59) <= 7.0                    # then a few mmHg a minute
    assert slope(work, 15, 30) > 1.4 * slope(rest, 15, 30)      # faster with effort
    assert len(work) < len(rest)                                # and the hold breaks sooner


def test_hold_cannot_pass_the_breaking_point():
    b = settle()
    b.set_pattern("hold")
    states, events = run(b, 150)
    assert any(s["phase"] != "hold" for s in states[int(100 / DT):])
    urges = [e for e in events if e["data"]["kind"] == "urge"]
    assert len(urges) == 2 and urges[1]["data"]["urge"] == pytest.approx(1.0)
    peak = max(s["co2_mmhg"] for s in states)
    assert 44 <= peak <= 56


def test_timed_hold_releases_itself():
    b = settle("natural", inhale=4, exhale=6, warm=40)
    b.set_pattern("hold", hold=10)
    states, _ = run(b, 30)
    held = sum(s["phase"] == "hold" for s in states) * DT
    assert held == pytest.approx(10, abs=0.1)
    assert states[-1]["pattern"] == "natural" and states[-1]["rate_bpm"] == pytest.approx(6)
    assert min(s["o2_sat"] for s in states) > 95        # a short hold barely touches oxygen


def test_hyperventilation_lowers_co2():
    b = settle("natural", inhale=1, exhale=1, warm=10)
    states, _ = run(b, 150)
    assert states[-1]["co2_mmhg"] < 35
    assert states[-1]["o2_sat"] >= 97.5
    assert any("light-headed" in l for l in b.felt()) or states[-1]["co2_mmhg"] >= 32


def test_effort_raises_breathing_and_heart():
    rest = run(settle(), 120)[0]
    b = settle()
    work = run(b, 180, effort=0.6)[0][-int(60 / DT):]
    ve = lambda xs: st.mean(s["rate_bpm"] * s["tidal_l"] for s in xs)
    assert ve(work) > 2 * ve(rest)
    assert st.mean(s["heart_bpm"] for s in work) > st.mean(s["heart_bpm"] for s in rest) + 20
    assert 36 <= st.mean(s["co2_mmhg"] for s in work) <= 44   # ventilation keeps pace with the CO2


# ---------------------------------------------------------------- nose


def test_nostril_temperatures_follow_the_breath():
    states, _ = run(settle(), 60)
    mid_in = [s for s in states if s["phase"] == "inhale" and s["flow_lps"] > 0.25]
    mid_out = [s for s in states if s["phase"] == "exhale" and s["flow_lps"] < -0.25]
    for side in ("left", "right"):
        assert st.mean(s["nostrils"][side]["temp_c"] for s in mid_in) < 26
        assert st.mean(s["nostrils"][side]["temp_c"] for s in mid_out) > 31
    for s in mid_in + mid_out:
        n = s["nostrils"]
        assert n["left"]["flow_lps"] + n["right"]["flow_lps"] == pytest.approx(s["flow_lps"], abs=2e-3)
        assert not s["mouth"]


def test_nasal_cycle_dominant_side_carries_more_flow():
    sides = set()
    for seed in range(10):
        states, _ = run(settle(seed=seed, warm=5), 20)
        n = states[-1]["nostrils"]
        dom, other = ("left", "right") if n["left"]["openness"] > n["right"]["openness"] else ("right", "left")
        sides.add(dom)
        flows = [(abs(s["nostrils"][dom]["flow_lps"]), abs(s["nostrils"][other]["flow_lps"]))
                 for s in states if abs(s["flow_lps"]) > 0.2]
        # a gentle cycle: the dominant side carries a few percent more of the air, not most of it
        ratio = sum(a for a, _ in flows) / sum(b for _, b in flows)
        assert 1.02 < ratio < 1.15
        # the cycle runs over hours: an hour barely changes it
        b = Breath(seed)
        o0 = b.state()["nostrils"][dom]["openness"]
        b.step(600.0)
        assert abs(b.state()["nostrils"][dom]["openness"] - o0) < 0.15
    assert sides == {"left", "right"}                   # the seed picks the side


def test_mouth_breathing_bypasses_the_nose():
    b = settle()
    line = b.set_pattern("natural", nose=False)
    assert "mouth" in line
    run(b, 10)
    states, _ = run(b, 30)
    assert all(s["mouth"] for s in states)
    assert all(s["nostrils"][k]["flow_lps"] == 0 for s in states for k in ("left", "right"))
    inh = [s for s in states if s["flow_lps"] > 0.25]
    assert inh and all(s["nostrils"]["left"]["temp_c"] > 28 for s in inh)   # no cool draught in the nose
    while b.state()["flow_lps"] < 0.2:
        b.step(DT)
    assert any("mouth" in l for l in b.felt())


# ---------------------------------------------------------------- heart


def test_heart_rises_on_inhale_falls_on_exhale():
    states, _ = run(settle("natural", inhale=4, exhale=6), 120)
    late_in = [s["heart_bpm"] for br in breaths(states) for s in br[300:400]]      # 3-4 s into the inhale
    late_out = [s["heart_bpm"] for br in breaths(states) for s in br[850:1000]]    # end of the exhale
    assert st.mean(late_in) > st.mean(late_out) + 5


def test_rsa_largest_near_six_breaths_a_minute():
    s6, _ = run(settle("natural", inhale=4, exhale=6), 120)
    s15, _ = run(settle("natural", inhale=1.6, exhale=2.4), 120)
    s3, _ = run(settle("natural", inhale=8, exhale=12), 160)
    w6, w15, w3 = swing(s6), swing(s15), swing(s3)
    assert w6 > 2 * w15
    assert 8 <= w6 <= 30 and 1.5 <= w15 <= 10
    # per litre of tide, the resonance itself: 6/min beats both faster and slower breathing
    per_l = lambda w, s: w / st.mean(x["tidal_l"] for x in s)
    assert per_l(w6, s6) > 1.3 * per_l(w15, s15)
    assert per_l(w6, s6) > per_l(w3, s3)


# ---------------------------------------------------------------- interface


def test_determinism_and_seeds():
    a, b, c = Breath(7), Breath(7), Breath(8)
    for _ in range(3000):
        a.step(DT, effort=0.1)
        b.step(DT, effort=0.1)
        c.step(DT, effort=0.1)
    assert a.state() == b.state()
    assert a.felt("full") == b.felt("full")
    assert a.state() != c.state()


def test_step_size_does_not_change_the_physiology():
    fine = run(settle(warm=10), 120, dt=0.01)[0]
    coarse = run(settle(warm=10), 120, dt=0.1)[0]
    assert st.mean(s["co2_mmhg"] for s in fine) == pytest.approx(st.mean(s["co2_mmhg"] for s in coarse), abs=1.0)


def test_state_is_json_and_complete():
    b = Breath()
    run(b, 5)
    s = json.loads(json.dumps(b.state()))
    keys = {"phase", "pattern", "rate_bpm", "tidal_l", "lung_volume_l", "flow_lps", "ribcage_share", "abdomen_share",
            "diaphragm", "diaphragm_descent_cm", "abdominal_pressure_cmh2o", "belly_cm", "chest_cm", "nostrils", "mouth",
            "co2_mmhg", "o2_sat", "urge", "heart_bpm"}
    assert keys <= s.keys()
    assert s["phase"] in ("inhale", "exhale", "pause", "hold")
    assert set(s["nostrils"]) == {"left", "right"}
    assert set(s["nostrils"]["left"]) == {"openness", "flow_lps", "temp_c"}
    assert isinstance(s["mouth"], bool)
    b.set_pattern("hold")
    run(b, 5)
    assert json.loads(json.dumps(b.state()))["pattern"] == "hold"


def test_set_pattern_lines_and_validation():
    b = Breath()
    for args in ({"pattern": "natural"}, {"pattern": "abdominal"}, {"pattern": "reverse", "inhale": 4, "exhale": 6},
                 {"pattern": "natural", "pause": 2}, {"pattern": "hold", "hold": 5}):
        line = b.set_pattern(**args)
        assert isinstance(line, str) and line and "\n" not in line
    with pytest.raises(ValueError):
        b.set_pattern("crane")
    with pytest.raises(ValueError):
        b.set_pattern("natural", inhale=-1)
    assert b.step(0) == []


def test_felt_lines_are_plain_and_sized():
    b = settle("abdominal", inhale=4, exhale=6, warm=30)
    for _ in range(40):
        run(b, 0.37)
        brief, full = b.felt(), b.felt("full")
        assert 2 <= len(brief) <= 3 and len(full) > len(brief)
        for line in full:
            words = line.lower().replace(",", " ").replace(".", " ").split()
            assert "you" not in words and "i" not in words and "your" not in words
            assert line[0].isupper() and line.endswith(".")
            line.encode("ascii")                        # safe on any console
    b.set_pattern("hold")
    run(b, 12)
    assert any("hold" in l.lower() for l in b.felt())


def test_settling_slows_and_deepens_the_breath_without_moving_co2():
    """Still and holding nothing, the natural breath slows toward 6-8 a minute and deepens over minutes, the same air
    reaching the alveoli (CO2 holds), oxygen use falls a little and the heart settles; moving again undoes it."""
    b = Breath(seed=4)
    awake_rates, awake_tides = [], []
    for i in range(int(240 / 0.05)):
        b.step(0.05, effort=0.06)
        if i * 0.05 > 120:
            awake_rates.append(b.state()["rate_bpm"]); awake_tides.append(b.state()["tidal_l"])
    awake = {"rate_bpm": st.mean(awake_rates), "tidal_l": st.mean(awake_tides)}
    rates, tides, co2, hr = [], [], [], []
    for i in range(int(480 / 0.05)):
        b.step(0.05, effort=0.06, settled=True)
        if i * 0.05 > 360:
            s = b.state()
            rates.append(s["rate_bpm"]); tides.append(s["tidal_l"]); co2.append(s["co2_mmhg"]); hr.append(s["heart_bpm"])
    assert b.state()["settled"] > 0.9
    assert st.mean(rates) < 9.5 < awake["rate_bpm"]
    assert st.mean(tides) > 1.4 * awake["tidal_l"]
    assert 37.0 < st.mean(co2) < 42.0
    assert any("slowed and deepened" in line for line in b.felt())
    for _ in range(int(60 / 0.05)):
        b.step(0.05, effort=0.06, settled=False)
    assert b.state()["settled"] < 0.05
