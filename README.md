# Interference in Marketplace Experiments

**When every unit of inventory is unique, your A/B test is measuring the wrong thing.**

A simulation and analysis showing that user-level randomization overstates
treatment effects by ~270% in a supply-constrained marketplace, why that happens,
and what the unbiased design costs.

## The problem

In online used-vehicle retail, every listing is a single unit. Two shoppers
cannot buy the same VIN. When one buys, that vehicle disappears for everyone
else.

This breaks the assumption underneath every standard A/B test: that one user's
assignment doesn't affect another user's outcome. Under user-level randomization
in a market where inventory binds, a treatment that raises purchase intent partly
succeeds by **taking vehicles the control group would otherwise have bought**.

The measured gap between arms is then part real effect, part reallocation. The
experimentation platform reports it as if it were all real.

## Results

Simulation: 40 markets, 900 shoppers each, 60 vehicles per market (demand exceeds
supply), true treatment effect of +2.0pp on purchase *intent*.

```
GROUND TRUTH (ship to everyone vs no one)
Conversion effect      +0.0024
```

The policy effect is far smaller than the intent lift, because extra intent
cannot create vehicles that don't exist. Supply, not persuasion, is the binding
constraint.

```
DESIGN 1: USER-LEVEL RANDOMIZATION
Estimate               +0.0089
95% CI                 [+0.0049, +0.0129]
p-value                0.0000
Shoppers hitting a stockout: 5.7%

Bias vs ground truth   +0.0065  (+269% of the true effect)
```

Highly significant, tight interval, and wrong by a factor of nearly four. The
confidence interval doesn't come close to containing the truth — this is bias,
not noise, so more traffic makes the estimate *more* confidently wrong.

```
DESIGN 2: MARKET-LEVEL (CLUSTER) RANDOMIZATION
Estimate               +0.0044
95% CI                 [-0.0014, +0.0103]
p-value                0.1426
```

Nearly unbiased, and it honestly reports that 40 markets aren't enough to resolve
an effect this small. That's the correct conclusion. The first design's answer was
never more certain — it was just wrong in a way the standard error could not see.

```
WHAT THE UNBIASED DESIGN COSTS
Intra-cluster correlation (ICC)  0.0013
Design effect                    2.1x
Effective sample size            16,896 (from 36,000 shoppers)
Standard error ratio             1.5x vs user-level
```

Shoppers within a market share inventory and local demand conditions, so they
aren't independent observations. The design effect `1 + (m-1)·ICC` quantifies the
penalty: 36,000 shoppers carry the statistical weight of about 17,000.

## The actual decision

Not "which design is correct" — design 2 obviously is. It's a trade:

| | User-level | Market-level |
|---|---|---|
| Bias | Large, upward | Approximately none |
| Precision | Tight | ~1.5x wider |
| Cost | Standard traffic | Many more markets |
| Failure mode | Confidently wrong | Honestly inconclusive |

The right call depends on how much inventory actually binds. Run with
`--scarcity loose` (400 vehicles per market) and interference largely disappears
— user-level randomization becomes fine. **Diagnose the constraint before
choosing the design**, because after the test has run, the bias is invisible.

## Running it

```bash
pip install -r requirements.txt

python analyze.py                        # default: tight supply
python analyze.py --scarcity loose       # interference mostly vanishes
python analyze.py --markets 120          # more clusters, tighter cluster CI
```

## Why simulated data

The ground-truth effect is unobservable in a real experiment — you never get to
see both the all-treated and all-control worlds. Simulating lets each design be
scored against the truth, which is the only way to *demonstrate* bias rather than
assert it.

## Limitations

- **Two designs only.** Switchback (time-sliced) randomization is the other
  standard answer and isn't implemented here.
- **No demand spillover across markets.** Shoppers who can't find a vehicle
  locally sometimes look elsewhere; that would add a second interference channel.
- **Instant conversion.** Real vehicle purchases convert over weeks, so lagged
  outcomes and surrogate metrics matter.
- **ICC is simulation-driven.** Real intra-market correlation should be measured
  from historical data, not assumed.
- **No pricing response.** Constrained inventory would move prices, which feeds
  back into conversion.

## Files

| File | Purpose |
|---|---|
| `marketplace.py` | Market simulation with finite unique inventory |
| `analyze.py` | Both designs, ground-truth comparison, ICC and design effect |
| `requirements.txt` | numpy, pandas, scipy |
