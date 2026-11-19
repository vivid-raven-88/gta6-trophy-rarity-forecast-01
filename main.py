"""gta6-trophy-rarity-forecast-01: dilution + adoption forecaster (Python 3.13, stdlib only).

Forecasts, for every GTA VI trophy, five quantiles of its Exophase earn rate
(percent of tracked players) at the case's target, scored by pinball loss on
the logit scale. Reads every <id>.in in the input directory, writes one
<id>.out per case: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.

METHOD (why this is not the naive persistence baseline)
-------------------------------------------------------
The house baselines say a trophy's rate stays where the last read put it
(persistence) or extends its recent logit trend (curve). Both miss the two
forces that actually move a launch-week earn rate:

1. DILUTION. A rate is earners / players_tracked. Exophase refreshes its
   counts on its own schedule, so the recorded players_tracked is stale
   between refreshes and jumps when it refreshes. On launch night it sat
   at 638 players for hours, then jumped to 3,868 in one read, and every
   rate but the one at 100% fell. A rate can fall a lot while nobody
   "un-earns" anything. So this program models earners and players
   separately and divides at the end, instead of extrapolating the rate.
2. ADOPTION. Earners for a trophy follow a saturating curve in time
   since launch: e(t) = E * (1 - exp(-t / tau)), where tau is the
   trophy's time constant, set by what the trophy asks for (story
   chapters are earned in hours, 100% completion in months). E is fit
   from the trophy's last observed earners, so the level comes from
   the data and only the shape comes from the prior. Two cohorts earn
   by the target: the players already tracked (E, saturating) and the
   players who join after the last read, who earn at the trophy's
   mature rate R = E / N_last over their mean age at the target (half
   the horizon). Without the second cohort a fast trophy collapses by
   the N_target/N_last factor on a long horizon although new joiners
   finish early chapters within a day or two of starting.

Players tracked N(t) is fit as a power law in time since launch
(log N vs log t, least squares on the distinct recorded values, slope
clamped to [0.2, 1.2]). When history is flat (no refresh ever seen),
the program falls back to N_target = N_last * (t_target/t_last)^0.8
times a staleness jump (1 + stale_hours/4): a tracker that has not
refreshed in hours is overdue one, and the target read will show it.

Spread: sigma = sqrt((0.5*sqrt(horizon_days))^2 + shock^2) in logit
units, where shock is 0.9 when history never showed a refresh and the
horizon is under a day (a refresh shock is the dominant uncertainty
there), 0.4 when history is flat but the horizon is long, else 0.15.
Quantiles use the standard normal z at the case's levels. The house
baseline's spread is the first term only, which is why its practice
loss is ~1.51 with 0% coverage: the refresh took every rate outside
its narrow 90-minute band.

ASSUMPTIONS (documented, not hidden)
------------------------------------
- Release is at 00:00 UTC on game.release_date; t is hours since then.
- A trophy at >= 99.5% stays near 100%: it survived a 6x dilution
  at 100% on launch night, so its earners scale with the tracked
  population (everyone who syncs has done it). Forecast near 99.9%.
- Class time constants (hours) are priors from GTA V / RDR2 trophy
  behaviour: story chapters 18-48h, final mission 72h, side strands
  96h, single missions 48h, grind/collectibles 400h, skill 200h,
  exploration/money 150h, misc one-offs 72h, 100%/platinum 1500h.
  They set the SHAPE only; the level E is fit per trophy from data.
- No network, no clock, no randomness: two runs write identical bytes.

The grader runs `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0);
outside it, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for
/input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# sibling module would import (none is shipped; this file is standalone).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Random-walk spread of a rate's move, logit units per sqrt day (same
# as the house baseline's SIGMA, so the first spread term matches it).
SIGMA = 0.5

# Time constants (hours) for the earners' saturating curve, by class.
TAU = {
    "platinum": 1500.0,
    "completion": 1500.0,   # 100% completion
    "story_early": 18.0,    # Chapter 1
    "story_early2": 24.0,   # Chapter 2
    "story_mid": 36.0,      # Chapter 3
    "story_mid2": 48.0,     # Chapter 4
    "story_final": 72.0,    # final mission
    "story_side": 96.0,     # final mission for a supporting character
    "mission": 48.0,        # a named story mission
    "sampler": 48.0,        # try each activity once
    "grind": 400.0,         # 250 stunts, 100 vehicles, 50 challenges, ...
    "grind_mid": 200.0,    # repeatable heists / skill grinds
    "skill": 200.0,         # gold medals, win at every location
    "explore": 150.0,       # all of Leonida
    "money": 150.0,         # earn / spend / buy big
    "misc": 72.0,           # one-off stunts, robberies, secrets
}


def classify(trophy: dict) -> str:
    """Class of a trophy from its tier, name and description."""
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    tier = trophy.get("tier", "")
    if tier == "platinum":
        return "platinum"
    if "100% completion" in text:
        return "completion"
    if "chapter 1" in text:
        return "story_early"
    if "chapter 2" in text:
        return "story_early2"
    if "chapter 3" in text:
        return "story_mid"
    if "chapter 4" in text:
        return "story_mid2"
    if "final mission for" in text:
        return "story_side"
    if "final mission" in text:
        return "story_final"
    if "try each type of activity" in text:
        return "sampler"
    if "gold medal in all" in text or "at every location" in text:
        return "skill"
    if "explore all of leonida" in text:
        return "explore"
    if any(k in text for k in ("250 stunts", "100 vehicles", "50 challenges",
                               "100 locations", "20 different races",
                               "3 collectible card sets", "8 mission strands",
                               "meditate at every spot", "discover 100")):
        return "grind"
    if "5 bank vaults" in text:
        return "grind_mid"
    if any(k in text for k in ("laundering", "spend $100,000",
                               "purchase all garage")):
        return "money"
    if "complete" in text and any(k in text for k in ("gone fishing", "breakfast run",
                                                       "cartel heist", "child's play",
                                                       "headliners", "vcia takedown",
                                                       "operation savage glory")):
        return "mission"
    return "misc"


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (as the starter does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def saturate(t_hours: float, tau: float) -> float:
    """Share of eventual earners who have earned by t_hours after launch."""
    if t_hours <= 0:
        return 0.0
    return 1.0 - math.exp(-t_hours / tau)


def forecast_players(history, t_last_h: float, t_target_h: float):
    """Forecast players_tracked at the target.

    Returns (n_target, n_distinct): n_target from a power-law fit on
    the distinct recorded counts, or a stale-tracker fallback.
    """
    points = []  # (t_hours, n) at each change of the recorded count
    seen_n = set()
    release = history[0].get("_release_midnight")
    for read in history:
        n = read.get("players_tracked")
        if n is None or n <= 0:
            continue
        t = (parse_time(read["slot"]) - release).total_seconds() / 3600.0
        if n not in seen_n:
            seen_n.add(n)
            points.append((t, float(n)))
    n_last = float(history[-1].get("players_tracked") or 0)
    if n_last <= 0:
        return 1000.0, len(points)
    if len(points) >= 2:
        xs, ys = [], []
        for t, n in points:
            if t > 0.5:  # need a sane log-time; drop pre-launch reads
                xs.append(math.log(t))
                ys.append(math.log(n))
        if len(xs) >= 2:
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2 for x in xs)
            alpha = num / den if den > 0 else 0.8
            alpha = min(1.2, max(0.2, alpha))
            log_c = my - alpha * mx
            n_target = math.exp(log_c + alpha * math.log(max(t_target_h, 0.5)))
            n_target = min(max(n_target, n_last), n_last * 50.0)
            return n_target, len(points)
    # Flat history: no refresh was ever seen. The recorded count is
    # stale; the target read will show the catch-up jump.
    stale_h = max(0.0, t_last_h - (points[0][0] if points else 0.0))
    growth = (max(t_target_h, 0.5) / max(t_last_h, 0.5)) ** 0.8
    jump = 1.0 + stale_h / 4.0
    return n_last * growth * jump, len(points)


def forecast_case(case: dict, name: str) -> dict:
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes", 0))
    horizon_days = horizon_min / 1440.0
    levels = case.get("levels") or LEVELS
    zvals = Z if levels == LEVELS else [0.0] * len(levels)

    release_raw = (case.get("game") or {}).get("release_date", "2026-11-19")
    release = datetime.fromisoformat(release_raw + "T00:00:00+00:00")
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    t_last_h = (at - release).total_seconds() / 3600.0
    t_target_h = (target - release).total_seconds() / 3600.0
    if t_last_h < 0:
        t_last_h = 0.0
    if t_target_h <= t_last_h:
        t_target_h = t_last_h + max(horizon_min / 60.0, 0.1)

    if not history:
        # No reads at all: fall back to small class priors, wide spread.
        rows = []
        for tr in trophies:
            tau = TAU[classify(tr)]
            med = 100.0 * 0.02 * saturate(t_target_h, tau)
            sigma = 1.5
            lg = logit(max(med, 0.01))
            rows.append({"id": tr["id"],
                         "q": [round4(clamp(inv_logit(lg + z * sigma))) for z in zvals]})
        return {"trophies": rows}

    for read in history:
        read["_release_midnight"] = release
    last = history[-1]
    n_last = float(last.get("players_tracked") or 0)
    n_target, n_distinct = forecast_players(history, t_last_h, t_target_h)
    flat_history = n_distinct <= 2

    # Refresh-shock spread: a tracker that never refreshed in the
    # history can jump at the target read; that dwarfs the random walk
    # on a short horizon and matters little on a long one.
    if flat_history and horizon_min < 1440:
        shock = 0.9
    elif flat_history:
        shock = 0.4
    else:
        shock = 0.15
    sigma_base = math.sqrt((SIGMA * math.sqrt(max(horizon_days, 0.0))) ** 2 + shock ** 2)

    rows = []
    for i, tr in enumerate(trophies):
        cls = classify(tr)
        tau = TAU[cls]
        # Last observed rate for this trophy (skip nulls backwards).
        rate_last = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                rate_last = float(rates[i])
                break
        if rate_last is None:
            rate_last = 0.0
        if rate_last >= 99.5:
            # Sticky 100%: survived a 6x dilution at 100% on launch
            # night, so its earners scale with the tracked population.
            lg = logit(99.9)
            sigma = 0.25
            rows.append({"id": tr["id"],
                         "q": [round4(clamp(inv_logit(lg + z * sigma))) for z in zvals]})
            continue
        # Earners at the last read, then the saturating adoption curve.
        # Two cohorts earn by the target. The players already tracked
        # contribute eventual * saturate(t_target, tau). The players
        # who join after the last read contribute too: they arrive
        # across the horizon (mean age half of it) and earn at the
        # trophy's mature rate R = eventual earners per tracked
        # player, capped at 1. Counting only the first cohort made
        # every fast trophy collapse by N_target/N_last on a long
        # horizon (chapter 1 at 96% fell to 35% on a synthetic final
        # case) although new joiners finish early chapters within a
        # day or two of starting.
        e_last = max(rate_last * max(n_last, 1.0) / 100.0, 0.5)
        s_last = saturate(t_last_h, tau)
        if s_last < 1e-6:
            s_last = max(t_last_h / tau, 1e-6)
        eventual = e_last / s_last
        e_old = eventual * saturate(t_target_h, tau)
        r_mature = min(eventual / max(n_last, 1.0), 0.999)
        age_new = max((t_target_h - t_last_h) / 2.0, 0.0)
        e_new = max(n_target - n_last, 0.0) * r_mature * saturate(age_new, tau)
        e_target = e_old + e_new
        med = clamp(100.0 * e_target / max(n_target, 1.0))
        # Rare trophies carry more relative uncertainty.
        sigma = sigma_base * (1.3 if rate_last < 1.0 else 1.0)
        lg = logit(med)
        rows.append({"id": tr["id"],
                     "q": [round4(clamp(inv_logit(lg + z * sigma))) for z in zvals]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
