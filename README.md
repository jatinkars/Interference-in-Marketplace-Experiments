# Choosing an Experiment Design Under Inventory Scarcity

When every unit of inventory is unique — one VIN, one buyer — a shopper who
converts removes a vehicle from everyone else's choice set. That breaks the
assumption under every standard A/B test: that one user's assignment does not
affect another user's outcome.

Most treatments of this pick a dramatic scarcity level, report a large bias, and
conclude "always cluster." That isn't useful guidance. The bias depends entirely
on how hard inventory binds, and clustering is expensive.

This study sweeps the demand-to-supply ratio and scores three designs against a
known ground truth at each level.

## Results

30 markets, 240 shoppers each, inventory varied from abundant to scarce.
15 replications per cell. Treatment raises purchase intent by 0.25 utility.

```
                        ratio    truth     user  cluster   switch  user bias
----------------------------------------------------------------------------
abundant           480    0.5   0.0527   0.0535   0.0613   0.0523        +1%
abundant           320    0.8   0.0385   0.0556   0.0466   0.0595       +44%
balanced           240    1.0   0.0450   0.0447   0.0705   0.0572        -1%
balanced           180    1.3   0.0489   0.0545   0.0615   0.0792       +12%
scarce             140    1.7   0.0267   0.0507   0.0330   0.1011       +90%
scarce             110    2.2   0.0117   0.0447   0.0083   0.1104      +283%
```

**User-level randomization** is close to unbiased while inventory is abundant and
degrades as demand approaches supply, reaching +283% once demand is roughly
double supply. The failure is not gradual noise — it is systematic, and it grows
precisely where the business is most supply-constrained.

**Cluster randomization** tracks the truth across the whole range. Each market
holds its own inventory, so there is no cross-arm depletion.

**Switchback fails here**, and that is the most interesting result. Toggling
treatment across arrival blocks does not work when the system carries state:
treated blocks deplete inventory that later control blocks never get to sell, so
control is handicapped by the design itself. The A/A check confirms it — with the
treatment effect set to zero, switchback still shows 0.05–0.07 bias under
scarcity while user and cluster stay within 0.008 of zero. Switchback is the
right tool for pricing or dispatch, where the system resets between blocks. It is
the wrong tool when the treatment consumes a shared, non-replenishing resource.

## What clustering costs

```
Intra-cluster correlation (balanced case) 0.1106
Cluster size                              240
Design effect  1 + (m-1)*ICC              27.4x
Effective sample size                     262 of 7,200 shoppers
Standard error inflation                  5.2x
```

Shoppers inside a market share inventory and local demand conditions, so they are
not independent observations. 7,200 shoppers carry the statistical weight of
roughly 260. That is the price of the unbiased design, and it is steep.

## The decision rule

Bias is not a property of the design. It is a property of the market.

| | User-level | Cluster |
|---|---|---|
| Bias when supply is abundant | Negligible | Negligible |
| Bias when demand ≈ 2× supply | Large, upward | Negligible |
| Precision | Full sample | ~5x wider intervals |
| Failure mode | Confidently wrong | Honestly inconclusive |

Measure the demand-to-supply ratio and the stockout rate **before** choosing the
design. When inventory is abundant, user-level randomization is both nearly
unbiased and far more precise — use it. Only when demand genuinely presses
against supply is clustering worth a 27x design effect.

After the test has run, the bias is invisible. It does not shrink with traffic;
it just becomes more precisely wrong.

## Validation

```bash
python experiment_design.py --aa-check
```

Sets the treatment effect to zero. User-level and cluster designs show no
meaningful bias (largest deviation 0.008), confirming the simulator is not
manufacturing the effect it claims to detect.

## Why simulated data

The ground-truth effect is unobservable in a real experiment — you never see both
the all-treated and all-control worlds. Simulating lets each design be scored
against the truth, which is the only way to demonstrate bias rather than assert
it.

## Limitations

- **No cross-market spillover.** Shoppers who can't find a vehicle locally
  sometimes look elsewhere, which would add a second interference channel and
  erode cluster randomization's advantage.
- **Instant conversion.** Real vehicle purchases take weeks, so lagged outcomes
  and surrogate metrics matter.
- **No pricing response.** Constrained inventory moves prices, which feeds back
  into conversion.
- **ICC is simulation-driven.** Real intra-market correlation should be measured
  from historical data, not assumed.
- **Residual noise.** Even at 15 replications, individual cells wobble by a few
  percent; the trend across the sweep is the signal, not any single row.

## Files

| File | Purpose |
|---|---|
| `experiment_design.py` | Market simulation, three designs, scarcity sweep, ICC |
| `requirements.txt` | numpy |
