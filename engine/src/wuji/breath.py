"""The breath of Wuji: lungs, diaphragm, belly, nose, blood gases and the heart's answer to the breath.

A lumped model, stepped by the body engine at about 100 Hz. Pure Python (plus numpy's seeded generator), no OS calls.

Mechanics
    The respiratory system is one elastic, resistive compartment (the equation of motion of Rohrer and Otis):
    P_mus = E * V + R * dV/dt, with E = 10 cmH2O/L (compliance 0.1 L/cmH2O, stiffening near total lung capacity) and
    R about 3.5 cmH2O/(L/s) through the nose, of which the nose is roughly half (Ferris, Mead and Opie 1964), 1.8 through
    the mouth. The pattern generator lays down the volume a breath should follow (smooth half-cosine strokes; the natural
    exhale is skewed early, like passive recoil braked by the diaphragm); the equation of motion, run backwards, gives the
    muscle pressure that breath needs: positive is inspiratory (the diaphragm, and the rib-cage muscles), negative is
    active expiration (the abdominals). V is volume above FRC, FRC = 2.8 L standing, inspiratory capacity 3 L.
    The breath is split into rib cage and abdomen after Konno and Mead (1967): V = V_rc + V_ab. Upright, the rib cage
    takes about two thirds of a quiet breath (Sharp et al. 1975); `abdominal` breathing gives the belly about three
    quarters; `reverse` breathing draws the belly in (V_ab < 0) while the rib cage takes more than the whole breath.
    Abdominal tension stiffens the belly wall and shifts the breath to the rib cage. The diaphragm's descent is the volume
    it displaces (the belly's outward movement plus a share of the lower rib cage it lifts through the zone of
    apposition) over an effective piston area of 0.27 L/cm, which gives about 1.2 cm in a quiet breath and 7-10 cm in a
    deep belly breath (M-mode ultrasound: 1.8 cm quiet, 7 cm deep, Boussuges, Gole and Blanc 2009; Kantarci et al. 2004).
    Abdominal (gastric) pressure sits near 8 cmH2O standing (Cobb et al. 2005) and rises with every centimetre of descent,
    much more steeply against a held or drawn-in belly (Hodges and Gandevia 2000).

Rhythm
    Natural breathing: about 13.5 breaths/min, tidal 0.5 L, inspiratory time 40 % of the cycle, varying breath to breath
    (log-normal, AR(1) correlated, CV about 12 %; Tobin et al. 1983, 1988), with an occasional sigh (2-3x tidal, every
    several minutes; Vlemincx et al. 2010). Ventilation follows metabolism (effort) and the chemoreflex: about +25 % per
    mmHg of CO2 above 40, weaker below (CO2 response slope 1.5-2.5 L/min/mmHg, Read 1967); rate and depth share the
    increase (depth first). Paced breathing follows the given times exactly; its depth is what keeps CO2 near 40 at that
    rate, slightly more (people over-breathe a little when pacing, much more when pacing fast), and grows with air hunger
    above 40 mmHg. Slow breathing at 6/min therefore runs about 1 L per breath (Russo, Santarelli and O'Rourke 2017).

Nose
    Each nostril's conductance grows with the square of its openness; the nasal cycle (Kayser 1895; Hasegawa and Kern
    1977; Eccles 2000) makes one side dominant, chosen by the seed, swinging over a period of 3-6 hours, so within a
    session it barely moves. Effort decongests both sides. Above about 35 L/min the mouth opens as well (the oronasal
    switch, Niinimaa et al. 1980). Air at the nostril: near room temperature (22 C) on the inhale, 32-34 C on the exhale
    (Keck et al. 2000), drifting toward 30 C when still, with a sensor-like lag of 0.25 s.

Gas
    CO2: two stores, after the classic apnoea data (Eger and Severinghaus 1961): the lung (gas plus tissue, about 6 mL/mmHg)
    exchanging with the body's tissue and venous store (51 mL/mmHg), metabolic VCO2 0.22 L/min at rest. Fresh alveolar
    gas (inspired volume beyond the 0.15 L dead space) washes CO2 out (VA * PACO2 / 863 = VCO2, the alveolar ventilation
    equation), so a breath-hold raises arterial CO2 a few mmHg in the first 20 s toward the venous level, then about
    4 mmHg/min at rest, faster with effort, and over-breathing lowers it. O2: alveolar O2 as a mass balance (VO2 0.25 L/min,
    humidified inspired fraction), arterial saturation from the Severinghaus (1979) equation with a 5 mmHg A-a gradient,
    and a venous store that buffers the fall once saturation leaves the flat of the curve; the reported saturation lags
    the lungs by about 8 s, as the periphery sees it. About 97-98 % while breathing; in a hold the lung's own O2 is what
    lasts, so it holds above 95 % for the first half minute or so near FRC, a minute from a full slow breath, then falls
    (a mild diving response spares about 10 % of the oxygen use).
    The urge to breathe tracks arterial CO2 (with a hypoxic boost): it starts above 41.5 mmHg and reaches the breaking
    point near 53 mmHg (Parkes 2006). Late in a hold the diaphragm contracts involuntarily against the closed throat
    (Agostoni 1963). At the breaking point the body breathes on its own: a hold cannot be forced past it.

Heart
    Respiratory sinus arrhythmia (Hirsch and Bishop 1981): the heart rate rises with the inhale and falls with the exhale,
    from two paths driven by lung volume: fast vagal gating (a 1 s low-pass) and the baroreflex loop, a resonance at
    0.1 Hz (6 breaths/min; Vaschillo et al. 2002; Lehrer, Vaschillo and Vaschillo 2000; Song and Lehrer 2003). Per litre of
    tide the swing is about 6 bpm at 15/min and 15 bpm at 6/min; with the bigger tide of slow breathing the swing is far
    larger. Baseline 61-67 bpm (seeded), effort raises it and damps the arrhythmia, a hold brings a mild diving-reflex
    slowing (Foster and Sheel 2005), and a little slow noise keeps it alive.
    Hypocapnia below about 32 mmHg is felt as light-headedness and tingling (Laffey and Kavanagh 2002).

References
    Agostoni E. J Appl Physiol 1963;18:30-36. Boussuges A, Gole Y, Blanc P. Chest 2009;135:391-400. Cobb WS et al.
    J Surg Res 2005;129:231-235. Eccles R. Acta Otolaryngol 2000;120:580-595. Eger EI, Severinghaus JW. Anesthesiology
    1961;22:419-425. Ferris BG, Mead J, Opie LH. J Appl Physiol 1964;19:653-658. Foster GE, Sheel AW. Scand J Med Sci
    Sports 2005;15:3-12. Hasegawa M, Kern EB. Mayo Clin Proc 1977;52:28-34. Hirsch JA, Bishop B. Am J Physiol
    1981;241:H620-H629. Hodges PW, Gandevia SC. J Appl Physiol 2000;89:967-976. Kantarci F et al. J Ultrasound Med
    2004;23:255-260. Keck T et al. Laryngoscope 2000;110:651-654. Konno K, Mead J. J Appl Physiol 1967;22:407-422.
    Laffey JG, Kavanagh BP. N Engl J Med 2002;347:43-53. Lehrer PM, Vaschillo E, Vaschillo B. Appl Psychophysiol
    Biofeedback 2000;25:177-191. Niinimaa V et al. Respir Physiol 1980;42:61-71. Otis AB, Fenn WO, Rahn H. J Appl Physiol
    1950;2:592-607. Parkes MJ. Exp Physiol 2006;91:1-15. Read DJ. Australas Ann Med 1967;16:20-32. Russo MA, Santarelli DM,
    O'Rourke D. Breathe 2017;13:298-309. Severinghaus JW. J Appl Physiol 1979;46:599-602. Sharp JT et al. J Appl Physiol
    1975;39:608-618. Song HS, Lehrer PM. Appl Psychophysiol Biofeedback 2003;28:13-23. Tobin MJ et al. Chest 1983;84:202-205;
    J Appl Physiol 1988;65:309-317. Vaschillo E et al. Appl Psychophysiol Biofeedback 2002;27:1-27. Vlemincx E et al.
    Biol Psychol 2010;84:82-87.

Conventions: flows are positive inward (inhaling); belly_cm and chest_cm are displacements from the relaxed end of an
exhale, positive outward; o2_sat is in percent.
"""
from __future__ import annotations

import math

import numpy as np

PATTERNS = ("natural", "abdominal", "reverse", "hold")

# mechanics
FRC_L = 2.8                 # functional residual capacity, standing
IC_L = 3.0                  # inspiratory capacity: the most a breath can add above FRC
DEAD_L = 0.15               # anatomical dead space
E_RS = 10.0                 # respiratory-system elastance, cmH2O/L
R_NOSE, R_MOUTH, R_BOTH = 3.5, 1.8, 2.4   # airway resistance, cmH2O/(L/s)
DI_AREA = 0.27              # diaphragm piston: litres displaced per cm of descent
AB_AREA, RC_AREA = 0.30, 0.45   # litres per cm of belly / chest wall movement
PAB_REST = 8.0              # abdominal (gastric) pressure standing, cmH2O
PDI_REF = 40.0              # transdiaphragmatic pressure taken as full activation, cmH2O
SHARE_AB = {"natural": 0.35, "abdominal": 0.75, "reverse": -0.25}

# rhythm
F0, VT0 = 13.5, 0.5         # natural rate (/min) and tidal volume (L)
VA0 = (VT0 - DEAD_L) * F0   # alveolar ventilation at rest, L/min
SIGH_P = 1 / 70             # chance per natural breath

# gas
VCO2_0, VO2_0 = 0.22, 0.25  # L/min STPD at rest
CO2_K = 863.0               # alveolar ventilation equation constant
C_LUNG_TISSUE = 0.0025      # L/mmHg, lung tissue and capillary blood
C_TISSUE = 0.051            # L/mmHg, body CO2 store reached within minutes
VEN_GAP = 6.6               # venous minus arterial CO2 at rest, mmHg
Q0 = 5.0                    # cardiac output, L/min
HB_CAP = 0.2                # L O2 per L blood, fully saturated
VB = 4.0                    # blood volume the venous O2 store mixes through, L
BTPS = 1.21                 # STPD -> BTPS
FI_O2 = 0.2095 * 713 / 760  # inspired O2 fraction once humidified
AA_GRAD = 5.0               # alveolar-arterial O2 gradient, mmHg
ROOM_C, EXHALED_NOSE_C, EXHALED_MOUTH_C, STILL_C = 22.0, 33.0, 34.5, 30.0

# heart
W0 = 2 * math.pi * 0.1      # baroreflex resonance, rad/s
ZETA = 0.25
A_GATE, B_RES = 7.4, 8.7    # bpm per litre of tide: vagal gating, baroreflex resonance

H_MAX = 0.02                # internal step, s


def _sat(po2: float) -> float:
    """Haemoglobin O2 saturation (0..1) from PO2 in mmHg (Severinghaus 1979)."""
    p = max(po2, 1.0)
    return 1.0 / (23400.0 / (p ** 3 + 150.0 * p) + 1.0)


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _shape(s: float, skew: bool) -> float:
    """Fraction of a stroke done at fraction s of its time: a half cosine, zero flow at both ends.
    skew: the early-peaking flow of a passive exhale."""
    if skew:
        s = 1.0 - (1.0 - s) ** 1.5
    return 0.5 - 0.5 * math.cos(math.pi * s)


def _secs(x: float) -> str:
    return f"{x:g} s"


class Breath:
    def __init__(self, seed: int = 0) -> None:
        ss = np.random.SeedSequence(seed)
        r_body, r_breath, r_heart = (np.random.default_rng(s) for s in ss.spawn(3))
        self._rng_breath, self._rng_heart = r_breath, r_heart
        self.t = 0.0
        # the practitioner's intention
        self._pattern = "natural"
        self._paced: tuple[float, float, float] | None = None
        self._pause_extra = 0.0
        self._nose = True
        # holds
        self._hold_pending = False
        self._hold_for: float | None = None
        self._hold_t = 0.0
        self._broke = False
        self._urge_fired = False
        self._await_settle = False
        self._breaths_since = 0
        # this body
        self._hr0 = 64.0 + float(r_body.uniform(-3, 3))
        dominant = int(r_body.integers(2))            # 0 left, 1 right
        self._nc_phase = (0.0 if dominant == 0 else math.pi) + float(r_body.uniform(-0.5, 0.5))
        self._nc_omega = 2 * math.pi / (float(r_body.uniform(3, 6)) * 3600)
        self._nf = self._nv = 0.0                     # breath-to-breath noise
        # mechanics
        self._V = 0.0
        self._flow = 0.0
        self._p_mus = self._p_hold = 0.0
        self._s_ab = SHARE_AB["natural"]
        self._tension = 0.0
        self._effort = 0.0
        self._met = 1.0
        self._ve = VT0 * F0
        self._mech: dict = {}
        # gas
        self._PA = 40.0
        self._Pv = 40.0 + VEN_GAP
        self._Pa = self._Pc = 40.0
        f_a = FI_O2 - VO2_0 * BTPS / VA0
        self._A_O2 = f_a * FRC_L
        self._Sa = _sat(f_a * 760 - AA_GRAD)
        self._Sv = self._Sa - VO2_0 / (Q0 * HB_CAP)
        self._Sp = self._Sa
        self._v_in = 0.0
        self._urge = 0.0
        # heart
        self._vslow = VT0 / 2
        self._gate = self._z = self._zd = 0.0
        self._hr_noise = 0.0
        self._hr = self._hr0
        self._dhr = 0.0
        self._hr_hi = self._hr_lo = self._hr0
        self._swing = 0.0
        # nose
        self._temp = [STILL_C, STILL_C]
        self._open = (0.6, 0.6)
        # first breath
        self._plan: dict = {}
        self._phase = "inhale"
        self._ph_t = self._ph_T = 0.0
        self._v0 = self._v1 = 0.0
        self._start_breath()
        self._update_derived(0.0)

    # ---------------------------------------------------------------- intention

    def set_pattern(self, pattern: str = "natural", inhale: float | None = None, exhale: float | None = None,
                    pause: float | None = None, hold: float | None = None, nose: bool = True) -> str:
        """pattern: natural | abdominal | reverse | hold (see the protocol). Returns one plain line saying what changes.

        A paced breath needs inhale and/or exhale (a missing one mirrors the other); pause alone keeps the body's own
        rhythm with that pause after each exhale. `hold` holds after the current phase for `hold` seconds (None or 0:
        until another call releases it, or the body's breaking point); timings and nose are kept from before the hold."""
        if pattern not in PATTERNS:
            raise ValueError(f"unknown breath pattern '{pattern}' (natural, abdominal, reverse or hold)")
        for name, v, lo, hi in (("inhale", inhale, 0.5, 30), ("exhale", exhale, 0.5, 30), ("pause", pause, 0, 60),
                                ("hold", hold, 0, 600)):
            if v is not None and not (lo <= v <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi} seconds")

        if pattern == "hold":
            if self._phase == "hold":
                self._hold_for = (self._hold_t + hold) if hold else None
                return f"Holding on, {self._hold_words(hold)}."
            self._hold_for = hold if hold else None
            if self._phase == "pause":
                self._enter_hold()
                return f"Holding the breath now with the lungs empty, {self._hold_words(hold)}."
            self._hold_pending = True
            where = "this inhale, with the lungs full" if self._phase == "inhale" else "this exhale, with the lungs empty"
            return f"Holding the breath after {where}, {self._hold_words(hold)}."

        self._pattern = pattern
        self._nose = bool(nose)
        self._hold_pending = False
        if inhale is not None or exhale is not None:
            ti = inhale if inhale is not None else exhale
            te = exhale if exhale is not None else inhale
            self._paced = (float(ti), float(te), float(pause or 0.0))
            self._pause_extra = 0.0
        else:
            self._paced = None
            self._pause_extra = float(pause or 0.0)
        released = self._phase == "hold"
        if released:
            self._release_hold()

        route = "through the nose" if self._nose else "through the mouth"
        how = {"natural": "", "abdominal": ", the belly leading",
               "reverse": ", the belly drawing in on each inhale while the ribs widen"}[pattern]
        if self._paced:
            ti, te, tp = self._paced
            rate = 60 / (ti + te + tp)
            p = f", pausing {_secs(tp)}" if tp > 0 else ""
            line = (f"Breathing in for {_secs(ti)} and out for {_secs(te)}{p} ({rate:.3g} breaths a minute), "
                    f"{route}{how}")
        elif pattern == "natural":
            p = f", resting {_secs(self._pause_extra)} after each exhale" if self._pause_extra else ""
            line = f"The breath returns to its own rhythm, {route}{p}"
        else:
            p = f", resting {_secs(self._pause_extra)} after each exhale" if self._pause_extra else ""
            line = f"{'Reverse breathing' if pattern == 'reverse' else 'Belly breathing'} at the body's own pace, {route}{how}{p}"
        line += "; the hold is released." if released else "; the change begins with the next breath."
        return line[0].upper() + line[1:]

    @staticmethod
    def _hold_words(hold: float | None) -> str:
        return f"for {_secs(hold)}" if hold else "until released"

    # ---------------------------------------------------------------- time

    def step(self, dt: float, effort: float = 0.0, belly_tension: float = 0.0) -> list[dict]:
        """Advance dt seconds. effort: whole-body muscular effort 0..1 (raises metabolic CO2 and breathing drive).
        belly_tension: abdominal tension the practitioner holds, 0..1 (stiffens the abdomen, shifting the breath to the
        rib cage). Returns protocol events, rare: an urge rising during a hold, or the breath settling after it."""
        events: list[dict] = []
        if not dt or dt <= 0:
            return events
        effort = _clip(float(effort), 0.0, 1.0)
        tension = _clip(float(belly_tension), 0.0, 1.0)
        n = max(1, math.ceil(dt / H_MAX - 1e-9))
        h = dt / n
        for _ in range(n):
            self._advance(h, effort, tension, events)
        return events

    def _advance(self, h: float, effort: float, tension: float, events: list[dict]) -> None:
        self.t += h
        hm = h / 60.0
        self._effort += (effort - self._effort) * min(1.0, h / 4.0)
        self._met += (1.0 + 4.0 * effort - self._met) * min(1.0, h / 30.0)
        self._tension += (tension - self._tension) * min(1.0, h / 0.5)
        self._s_ab += (SHARE_AB[self._pattern] - self._s_ab) * min(1.0, h / 1.5)

        # -- the stroke
        v_old = self._V
        self._ph_t += h
        if self._phase in ("inhale", "exhale"):
            s = min(1.0, self._ph_t / self._ph_T)
            skew = self._phase == "exhale" and self._paced is None
            self._V = self._v0 + (self._v1 - self._v0) * _shape(s, skew)
        elif self._phase == "pause":
            self._V += -self._V * min(1.0, h / 0.5)
        else:
            self._hold_t += h
        dv = self._V - v_old
        self._flow = dv / h

        if self._phase == "hold":
            if self._urge >= 1.0:
                self._broke = True
                events.append(self._event("urge", "The hold breaks: the body takes its breath on its own."))
                self._release_hold()
            elif self._hold_for is not None and self._hold_t >= self._hold_for:
                self._release_hold()
        elif self._ph_t >= self._ph_T:
            self._next_phase()

        # -- gas exchange
        q = Q0 * (1.0 + 0.5 * (self._met - 1.0))
        vco2, vo2 = VCO2_0 * self._met, VO2_0 * self._met
        if self._phase == "hold":   # the diving response spares a little oxygen
            vo2 *= 1.0 - 0.1 * min(1.0, self._hold_t / 20.0)
        lung = FRC_L + self._V
        f_a = self._A_O2 / max(lung - dv, 0.5)
        if dv > 0:
            v_in = self._v_in + dv
            fresh = max(0.0, v_in - DEAD_L) - max(0.0, self._v_in - DEAD_L)
            self._v_in = v_in
            self._A_O2 += fresh * FI_O2 + (dv - fresh) * f_a
        else:
            fresh = 0.0
            self._A_O2 += dv * f_a
        g = (VCO2_0 / VEN_GAP) * q / Q0
        to_lung = g * (self._Pv - self._PA) * hm
        c_lung = lung / CO2_K + C_LUNG_TISSUE
        self._PA += (to_lung - fresh * self._PA / CO2_K) / c_lung
        self._Pv += (vco2 * hm - to_lung) / C_TISSUE
        self._Pa += (self._PA - self._Pa) * min(1.0, h / 4.0)
        self._Pc += (self._Pa - self._Pc) * min(1.0, h / 20.0)
        pao2 = self._A_O2 / lung * 760.0
        self._Sa = _sat(pao2 - AA_GRAD)
        uptake = q * HB_CAP * (self._Sa - self._Sv)
        self._A_O2 = max(0.0, self._A_O2 - uptake * BTPS * hm)
        self._Sv += (self._Sa - vo2 / (q * HB_CAP) - self._Sv) * min(1.0, hm * q / VB)
        self._Sv = _clip(self._Sv, 0.05, 1.0)
        self._Sp += (self._Sa - self._Sp) * min(1.0, h / 8.0)   # as the periphery and carotid bodies see it
        u_co2 = _clip((self._Pa - 41.5) / (53.0 - 41.5), 0.0, 1.0) ** 1.2
        u_o2 = _clip((0.92 - self._Sp) / 0.15, 0.0, 1.0)
        self._urge = _clip(u_co2 + u_o2, 0.0, 1.0)

        # -- events
        if self._phase == "hold":
            if not self._urge_fired and self._urge >= 0.5:
                self._urge_fired = self._await_settle = True
                events.append(self._event(
                    "urge", f"The urge to breathe rises, {self._hold_t:.0f} s into the hold; the chest wants air."))
        elif self._await_settle and self._breaths_since >= 3 and self._urge < 0.05 and self._Pc < 41.5:
            self._await_settle = False
            text = ("The hold broke at its limit; now the breath has settled again." if self._broke
                    else "The breath has settled again after the hold.")
            events.append(self._event("settled", text))

        # -- heart
        self._vslow += (self._V - self._vslow) * min(1.0, h / 30.0)
        u = self._V - self._vslow
        self._gate += (u - self._gate) * min(1.0, h / 1.0)
        self._zd += h * (W0 * W0 * (u - self._z) - 2 * ZETA * W0 * self._zd)
        self._z += h * self._zd
        rsa = (1.0 - 0.7 * self._effort) * (A_GATE * self._gate + B_RES * (2 * ZETA / W0) * self._zd)
        self._hr_noise += -self._hr_noise * h / 8.0 + 0.8 * math.sqrt(2 * h / 8.0) * float(self._rng_heart.standard_normal())
        base = self._hr0 + 70.0 * self._effort + 0.4 * max(0.0, self._Pa - 42.0) + 40.0 * max(0.0, 0.94 - self._Sp)
        if self._phase == "hold":
            base *= 1.0 - 0.08 * min(1.0, self._hold_t / 25.0)
        hr = base + rsa + self._hr_noise
        self._dhr += ((hr - self._hr) / h - self._dhr) * min(1.0, h / 0.5)
        self._hr = hr
        self._hr_hi, self._hr_lo = max(self._hr_hi, hr), min(self._hr_lo, hr)

        self._update_derived(h)

    def _update_derived(self, h: float) -> None:
        """Muscles, partition, pressures and nostrils from the present volume and flow."""
        tension = self._tension
        if self._phase == "hold":
            self._p_mus += (0.3 * self._p_hold - self._p_mus) * min(1.0, h / 2.0)
        else:
            nose_only, mouth_only = self._nasal_share() == 1.0, self._nasal_share() == 0.0
            r = R_NOSE if nose_only else R_MOUTH if mouth_only else R_BOTH
            self._p_mus = E_RS * (self._V + 0.04 * self._V ** 3) + r * self._flow
        p_insp = max(0.0, self._p_mus)
        p_exp = max(0.0, -self._p_mus - 0.5)
        s = self._s_ab
        s_eff = s * (1.0 - 0.75 * tension) if s > 0 else s
        v_ab = s_eff * self._V
        v_rc = self._V - v_ab
        descent = (max(v_ab, 0.0) + 0.4 * max(v_rc, 0.0)) / DI_AREA
        stiff = _clip(tension + 1.5 * max(0.0, -v_ab) + 0.05 * p_exp, 0.0, 1.0)
        rise = descent * (1.0 + 5.0 * stiff)
        pab = PAB_REST + 15.0 * stiff + rise + p_exp
        pdi = (0.6 + 0.3 * max(0.0, s_eff)) * p_insp + rise
        act = pdi / PDI_REF
        if self._phase == "hold" and self._urge > 0.6:
            period = 4.0 - 2.5 * (self._urge - 0.6) / 0.4
            act += 0.25 * (self._urge - 0.6) / 0.4 * max(0.0, math.sin(2 * math.pi * self._hold_t / period)) ** 4
        # nostrils
        phase = self._nc_phase + self._nc_omega * self.t
        swing = 0.22 * math.cos(phase)
        decong = 0.15 * self._effort
        o_l = _clip(0.6 + swing + decong, 0.1, 1.0)
        o_r = _clip(0.6 - swing + decong, 0.1, 1.0)
        self._open = (o_l, o_r)
        g_l, g_r = o_l * o_l, o_r * o_r
        nasal = self._flow * self._nasal_share()
        flows = (nasal * g_l / (g_l + g_r), nasal * g_r / (g_l + g_r))
        k = min(1.0, h / 0.25)
        for i, f in enumerate(flows):
            w = abs(f) / (abs(f) + 0.03)
            target = STILL_C + ((ROOM_C if f > 0 else EXHALED_NOSE_C) - STILL_C) * w
            self._temp[i] += (target - self._temp[i]) * k
        self._mech = dict(s_eff=s_eff, v_ab=v_ab, v_rc=v_rc, descent=descent, pab=pab, act=_clip(act, 0.0, 1.0),
                          flows=flows, p_exp=p_exp)

    def _nasal_share(self) -> float:
        if not self._nose:
            return 0.0
        return 1.0 if self._ve < 35.0 else 0.6

    # ---------------------------------------------------------------- phases

    def _ve_ratio(self, paced: bool) -> float:
        dp = self._Pc - 40.0
        chemo = 1.0 + (0.25 * dp if dp > 0 else (0.0 if paced else 0.06 * dp))
        return self._met * _clip(chemo, 0.5, 5.0)

    def _start_breath(self) -> None:
        rng = self._rng_breath
        rho = 0.5
        self._nf = rho * self._nf + math.sqrt(1 - rho * rho) * float(rng.standard_normal())
        self._nv = rho * self._nv + math.sqrt(1 - rho * rho) * float(rng.standard_normal())
        sigh = False
        if self._paced:
            ti, te, tp = self._paced
            rate = 60.0 / (ti + te + tp)
            r = self._ve_ratio(paced=True)
            over = 1.05 + 0.25 * max(0.0, rate - F0) / F0
            vt = (VA0 * r / rate + DEAD_L) * over
        else:
            r = self._ve_ratio(paced=False)
            rate = F0 * r ** 0.35 * math.exp(0.12 * self._nf)
            vt = VT0 * r ** 0.65 * math.exp(0.12 * self._nv)
            if self._pattern == "natural" and self._phase != "hold" and float(rng.random()) < SIGH_P:
                sigh = True
                vt *= 2.3
            cycle = 60.0 / rate
            ti = cycle * _clip(0.40 + 0.02 * (r - 1.0), 0.38, 0.48)
            tp = cycle * _clip(0.15 - 0.04 * (r - 1.0), 0.02, 0.15) + (1.0 if sigh else 0.0) + self._pause_extra
            te = cycle - ti - (tp - (1.0 if sigh else 0.0) - self._pause_extra)
            rate = 60.0 / (ti + te + tp)
        vt = _clip(vt, 0.25, IC_L * 0.95)
        self._plan = dict(ti=ti, te=te, tp=tp, vt=vt, rate=rate, sigh=sigh)
        self._ve = vt * rate
        self._breaths_since += 1
        self._swing = self._hr_hi - self._hr_lo
        self._hr_hi = self._hr_lo = self._hr
        self._v_in = 0.0
        self._begin("inhale", ti, vt)

    def _begin(self, phase: str, duration: float, v_end: float) -> None:
        self._phase = phase
        self._ph_t = 0.0
        self._ph_T = max(duration, 0.05)
        self._v0, self._v1 = self._V, v_end

    def _exhale_time(self) -> float:
        return self._paced[1] if self._paced else self._plan["te"]

    def _pause_time(self) -> float:
        return self._paced[2] if self._paced else self._plan["tp"]

    def _next_phase(self) -> None:
        over = self._ph_t - self._ph_T
        if self._phase == "inhale":
            if self._hold_pending:
                self._enter_hold()
                return
            self._begin("exhale", self._exhale_time(), 0.0)
        elif self._phase == "exhale":
            if self._hold_pending:
                self._enter_hold()
                return
            tp = self._pause_time()
            if tp > 0.05:
                self._begin("pause", tp, 0.0)
            else:
                self._start_breath()
        else:
            self._start_breath()
        self._ph_t = min(over, self._ph_T)

    def _enter_hold(self) -> None:
        self._hold_pending = False
        self._phase = "hold"
        self._hold_t = 0.0
        self._broke = False
        self._urge_fired = False
        self._await_settle = False
        self._p_hold = self._p_mus

    def _release_hold(self) -> None:
        self._hold_pending = False
        self._breaths_since = 0
        if self._V > 0.1:
            self._begin("exhale", 1.2 if self._broke else self._exhale_time(), 0.0)
        else:
            self._start_breath()

    def _event(self, kind: str, text: str) -> dict:
        return {"event": "breath", "data": {"kind": kind, "urge": round(self._urge, 3), "text": text}}

    # ---------------------------------------------------------------- reading

    def state(self) -> dict:
        """JSON-serialisable snapshot of the breath (see the module docstring for units and signs)."""
        m = self._mech
        o_l, o_r = self._open
        return {
            "phase": self._phase,
            "pattern": "hold" if (self._phase == "hold" or self._hold_pending) else self._pattern,
            "paced": bool(self._paced),
            "rate_bpm": round(self._plan["rate"], 2),
            "tidal_l": round(self._plan["vt"], 3),
            "lung_volume_l": round(FRC_L + self._V, 3),
            "flow_lps": round(self._flow, 3),
            "ribcage_share": round(1.0 - m["s_eff"], 3),
            "abdomen_share": round(m["s_eff"], 3),
            "diaphragm": round(m["act"], 3),
            "diaphragm_descent_cm": round(m["descent"], 2),
            "abdominal_pressure_cmh2o": round(m["pab"], 1),
            "belly_cm": round(m["v_ab"] / AB_AREA, 2),
            "chest_cm": round(m["v_rc"] / RC_AREA, 2),
            "nostrils": {
                "left": {"openness": round(o_l, 3), "flow_lps": round(m["flows"][0], 3), "temp_c": round(self._temp[0], 1)},
                "right": {"openness": round(o_r, 3), "flow_lps": round(m["flows"][1], 3), "temp_c": round(self._temp[1], 1)},
            },
            "mouth": self._nasal_share() < 1.0,
            "co2_mmhg": round(self._Pa, 1),
            "o2_sat": round(100.0 * self._Sp, 1),
            "urge": round(self._urge, 3),
            "heart_bpm": round(self._hr, 1),
            "hold_s": round(self._hold_t, 1) if self._phase == "hold" else 0.0,
        }

    def felt(self, detail: str = "brief") -> list[str]:
        """The breath as felt, in plain present-tense lines. brief: 2-3 lines; full: more (rates, volumes)."""
        m, p, ph = self._mech, self._plan, self._phase
        s = m["s_eff"]
        lines = [self._felt_motion(ph, p, s)]
        # air
        share = self._nasal_share()
        f_l, f_r = (abs(f) for f in m["flows"])
        side = "left" if f_l > f_r else "right"
        lop = abs(f_l - f_r) / max(f_l + f_r, 1e-9)
        where = f"mostly in the {side} nostril" if lop > 0.2 else "even in both nostrils"
        if ph in ("hold", "pause") or abs(self._flow) < 0.03:
            lines.append("No air moves at the nose." if share > 0 else "No air moves; the mouth rests open.")
        elif share == 0.0:
            lines.append("Cool air over the tongue, through the open mouth; the nose is quiet." if self._flow > 0
                         else "Warm air out through the open mouth; the nose is quiet.")
        else:
            mouth = " and the open mouth" if share < 1.0 else ""
            lines.append(f"Cool air on the inhale, {where}{mouth}." if self._flow > 0
                         else f"Warm air leaving the nose, {where.replace(' in ', ' through ')}{mouth}.")
        # third: urge, hypocapnia, or heart
        if self._urge >= 0.2:
            if ph == "hold":
                if self._urge >= 0.6:
                    lines.append("The urge to breathe is strong; the diaphragm tugs against the held breath.")
                else:
                    lines.append("The urge to breathe grows." if self._urge >= 0.4 else "A mild urge to breathe.")
            else:
                lines.append("A hunger for air; the breath runs deeper to answer it.")
        elif self._Pa < 32.0:
            lines.append("A little light-headed; the fingertips tingle.")
        elif ph == "hold":
            lines.append(f"Still and easy, {self._hold_t:.0f} s into the hold.")
        # the heart follows the breath with a lag, so its trend is told as it is, joined to the breath only when they agree
        elif self._dhr > 0.4:
            lines.append("The heart quickens with the in-breath." if ph == "inhale" else "The heart is quickening.")
        elif self._dhr < -0.4:
            lines.append("The heart slows with the out-breath." if ph in ("exhale", "pause") else "The heart is slowing.")
        else:
            lines.append(f"The heart beats steadily, about {self._hr:.0f} a minute.")
        if detail != "full":
            return lines
        rc = 100 * (1 - s)
        lines += [
            f"About {p['rate']:.1f} breaths a minute, {p['vt']:.2f} L each; the lungs hold {FRC_L + self._V:.2f} L now.",
            (f"The belly carries {100 * s:.0f}% of the breath and the ribs {rc:.0f}%" if s >= 0 else
             f"The ribs take {rc:.0f}% of the breath while the belly draws in by {-100 * s:.0f}%") + "; the diaphragm is "
            f"{m['descent']:.1f} cm down, the belly {m['v_ab'] / AB_AREA:+.1f} cm, the chest {m['v_rc'] / RC_AREA:+.1f} cm.",
            f"Pressure in the belly {m['pab']:.0f} cmH2O; the diaphragm works at {100 * m['act']:.0f}% of a strong effort.",
            f"Heart {self._hr:.0f} beats a minute, swinging about {self._swing:.0f} between in-breath and out-breath.",
            f"CO2 {self._Pa:.0f} mmHg, oxygen saturation {100 * self._Sp:.0f}%, urge to breathe {100 * self._urge:.0f}%.",
            f"The {'left' if self._open[0] > self._open[1] else 'right'} nostril is the more open one now "
            f"({self._open[0]:.2f} left, {self._open[1]:.2f} right).",
        ]
        return lines

    def _felt_motion(self, ph: str, p: dict, s: float) -> str:
        if ph == "hold":
            full = self._V > 0.1
            return (f"Holding the breath with the lungs {'full' if full else 'empty'}, {self._hold_t:.0f} s so far, "
                    f"the throat closed.")
        if ph == "pause":
            return "A pause at the bottom of the breath, the body still."
        route = "the nose" if self._nasal_share() == 1.0 else "the mouth" if self._nasal_share() == 0.0 else "nose and mouth"
        dur = self._ph_T
        adj = []
        if dur >= 3.5:
            adj.append("slow")
        elif dur <= 1.0:
            adj.append("quick")
        if p["vt"] >= 1.2:
            adj.append("deep")
        elif p["vt"] <= 0.35:
            adj.append("shallow")
        if p.get("sigh"):
            adj = ["sighing"]
        head = (", ".join(adj) + " " + ph) if adj else ph
        head = head[0].upper() + head[1:]
        if ph == "inhale":
            if s < 0:
                body = "the belly drawing in as the ribs widen"
            elif s > 0.6:
                body = "the belly swelling"
            elif s > 0.25:
                body = "belly and ribs rising together"
            else:
                body = "the ribs widening above a firm belly"
        else:
            if s < 0:
                body = "the belly releasing outward as the ribs settle"
            elif s > 0.6:
                body = "the belly sinking"
            elif s > 0.25:
                body = "belly and ribs settling together"
            else:
                body = "the ribs settling, the belly firm"
        return f"{head} through {route}, {body}."
