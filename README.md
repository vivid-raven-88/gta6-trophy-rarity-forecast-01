# GTA VI Trophy Rarity Forecast — Week One

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vivid-raven-88/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vivid-raven-88/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

A forecaster for [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) — for every PlayStation 5 trophy of Grand Theft Auto VI, five quantiles of its Exophase earn rate at 2026-11-26T00:00:00Z, a week after launch.

## How it works

`main.py` (Python 3.13, standard library only) is not the naive persistence
baseline. An earn rate is earners divided by players tracked, and both move
for different reasons, so the program models them separately:

- **Dilution.** Exophase refreshes its counts on its own schedule, so the
  recorded players-tracked count is stale between refreshes and jumps when
  it refreshes. On launch night it sat at 638 players for hours, then
  jumped to 3,868 in one read, and every rate but the one at 100% fell.
  The program fits players-tracked as a power law in time since launch
  (log-log least squares on the distinct recorded values), and when
  history never showed a refresh it expects the catch-up jump at the
  target read instead of trusting the stale count.
- **Adoption.** Earners for a trophy follow a saturating curve in time
  since launch, `e(t) = E * (1 - exp(-t / tau))`, where `tau` is the
  trophy's time constant, set by what the trophy asks for: story
  chapters are earned in hours, 100% completion in months. `E` is fit
  from the trophy's last observed earners, so the level comes from the
  data and only the shape comes from the prior (GTA V / RDR2 trophy
  behaviour: chapters 18–48h, final mission 72h, side strands 96h,
  grind and collectibles 400h, skill 200h, exploration and money 150h,
  misc one-offs 72h, 100% / platinum 1500h).
- **Spread.** The pinball loss is scored on the logit scale, so the
  spread is in logit units: a random-walk term that widens with the
  horizon (the house baseline's term) plus a refresh-shock term that
  dominates on short horizons with a stale tracker — exactly the case
  where the house baseline scored ~1.51 with 0% coverage on the
  practice. A trophy at 100% is forecast sticky near 100%: it survived
  a 6x dilution at 100% on launch night.

## Running it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes one
`<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.
Deterministic: two runs write identical bytes. No network, no clock,
no randomness.

## License

MIT — see [LICENSE](LICENSE).
