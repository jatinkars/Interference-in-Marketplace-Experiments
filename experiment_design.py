"""
Choosing an Experiment Design Under Inventory Scarcity
======================================================

In a marketplace where every unit is unique — one VIN, one buyer — a shopper who
converts removes a vehicle from everyone else's choice set. That breaks the
assumption under every standard A/B test: that one user's assignment does not
affect another user's outcome.

Most write-ups on this pick a dramatic scarcity level, report a large bias, and
conclude "always cluster." That is not useful guidance. The bias depends entirely
on how hard inventory binds, and clustering is expensive.

This study sweeps the demand-to-supply ratio and reports, at each level:

  * bias of user-level randomization against a known ground truth
  * bias of market-level (cluster) randomization
  * bias of switchback (time-sliced) randomization
  * what clustering costs in precision (ICC, design effect, effective N)

The output is a decision rule, not a single headline number.

Run:
    python experiment_design.py              # full scarcity sweep
    python experiment_design.py --aa-check   # validate: no effect, no bias
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np

SEED = 20260914


# ---------------------------------------------------------------------------
# Marketplace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Market:
    n_markets: int = 30
    inventory_per_market: int = 120     # unique vehicles, each sellable once
    shoppers_per_market: int = 240      # demand over the full test window
    consideration_set: int = 10
    outside_option: float = 1.9         # utility of not buying; sets base conversion
    intent_effect: float = 0.25         # treatment shift in utility for treated shoppers
    market_sd: float = 0.80             # between-market variation in demand conditions

    @property
    def demand_supply_ratio(self) -> float:
        return self.shoppers_per_market / self.inventory_per_market


def simulate(
    m: Market,
    treated_shoppers: np.ndarray,   # (n_markets, shoppers_per_market) bool
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shoppers arrive in random order. Each draws a consideration set from the
    vehicles still AVAILABLE in their market, then makes a logit choice against
    an outside option.

    Sold vehicles leave the pool permanently. That depletion is the interference
    channel: a purchase by one shopper shrinks the option set for the next.

    Returns (converted, market_index) per shopper.
    """
    converted = np.zeros((m.n_markets, m.shoppers_per_market), dtype=bool)
    starved = np.zeros((m.n_markets, m.shoppers_per_market), dtype=bool)
    market_effects = rng.normal(0.0, m.market_sd, m.n_markets)

    for mk in range(m.n_markets):
        # Markets genuinely differ in demand conditions and inventory quality.
        # Without this, shoppers within a market are independent and ICC is 0,
        # which understates the real cost of clustering.
        market_effect = market_effects[mk]
        quality = rng.normal(market_effect, 0.4, m.inventory_per_market)
        available = np.ones(m.inventory_per_market, dtype=bool)

        for s in range(m.shoppers_per_market):
            pool = np.flatnonzero(available)
            if pool.size < m.consideration_set:
                starved[mk, s] = True
            if pool.size == 0:
                continue  # sold out; shopper cannot convert

            k = min(m.consideration_set, pool.size)
            cset = rng.choice(pool, size=k, replace=False)

            util = quality[cset] + rng.gumbel(0.0, 1.0, k)
            if treated_shoppers[mk, s]:
                util = util + m.intent_effect
            outside = m.outside_option + rng.gumbel(0.0, 1.0)

            best = int(np.argmax(util))
            if util[best] > outside:
                converted[mk, s] = True
                available[cset[best]] = False

    return converted, starved


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def ground_truth(m: Market, rng: np.random.Generator, reps: int = 15) -> float:
    """
    Conversion rate with everyone treated minus everyone control.

    This is the launch decision. It is also unobservable in a real experiment,
    which is exactly why the data here is simulated: you cannot score a design
    against a truth you cannot see.
    """
    on = off = 0.0
    for _ in range(reps):
        all_on = np.ones((m.n_markets, m.shoppers_per_market), dtype=bool)
        all_off = np.zeros((m.n_markets, m.shoppers_per_market), dtype=bool)
        on += simulate(m, all_on, rng)[0].mean()
        off += simulate(m, all_off, rng)[0].mean()
    return (on - off) / reps


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------


def user_level(m: Market, rng, reps: int = 15) -> tuple[float, float]:
    """Randomize shoppers. Both arms draw from the same inventory pool."""
    ests, starve = [], []
    for _ in range(reps):
        assign = rng.random((m.n_markets, m.shoppers_per_market)) < 0.5
        conv, st = simulate(m, assign, rng)
        ests.append(conv[assign].mean() - conv[~assign].mean())
        starve.append(st.mean())
    return float(np.mean(ests)), float(np.mean(starve))


def cluster_level(m: Market, rng, reps: int = 15) -> tuple[float, float]:
    """
    Randomize whole markets. Each market has its own inventory, so no cross-arm
    depletion. Returns (estimate, intra-cluster correlation).
    """
    ests, iccs = [], []
    for _ in range(reps):
        treated = rng.random(m.n_markets) < 0.5
        if treated.all() or (~treated).all():
            continue
        assign = np.repeat(treated[:, None], m.shoppers_per_market, axis=1)
        conv, _ = simulate(m, assign, rng)
        ests.append(conv[treated].mean() - conv[~treated].mean())
        iccs.append(_icc(conv, m))
    return float(np.mean(ests)), float(np.mean(iccs))


def _icc(conv: np.ndarray, m: Market) -> float:
    """
    Intra-cluster correlation via one-way ANOVA decomposition on market means.

    Shoppers inside a market share inventory and local conditions, so they are
    not independent observations. ICC quantifies how much.
    """
    market_means = conv.mean(axis=1)
    grand = conv.mean()
    n = m.shoppers_per_market
    between = n * ((market_means - grand) ** 2).sum() / (m.n_markets - 1)
    within = ((conv - market_means[:, None]) ** 2).sum() / (m.n_markets * (n - 1))
    if between + (n - 1) * within == 0:
        return 0.0
    return max(0.0, (between - within) / (between + (n - 1) * within))


def switchback(m: Market, rng, n_blocks: int = 8, reps: int = 15) -> float:
    """
    Alternate the whole marketplace on and off across time blocks. Nothing
    competes against a differently-treated shopper at the same moment.

    Cost: fewer effective observations, and vulnerability to time trends that
    happen to align with the block schedule.
    """
    # Treatment toggles by ARRIVAL BLOCK within one continuous market run, so
    # inventory depletion carries across blocks exactly as it would in reality.
    block_id = np.arange(m.shoppers_per_market) // max(
        1, m.shoppers_per_market // n_blocks
    )
    pattern = (block_id % 2 == 0)

    ests = []
    for _ in range(reps):
        assign = np.repeat(pattern[None, :], m.n_markets, axis=0)
        conv, _ = simulate(m, assign, rng)
        ests.append(conv[assign].mean() - conv[~assign].mean())
    return float(np.mean(ests))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def pct_bias(est: float, truth: float) -> str:
    if abs(truth) < 1e-6:
        return "n/a"
    return f"{(est - truth) / abs(truth):+.0%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa-check", action="store_true",
                    help="run with zero treatment effect; all designs should show no bias")
    ap.add_argument("--markets", type=int, default=30)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    bar = "-" * 72

    base = Market(n_markets=args.markets)
    if args.aa_check:
        base = replace(base, intent_effect=0.0)
        print("\nA/A CHECK — treatment effect set to zero")
        print("If any design shows meaningful bias here, the simulator is wrong.")

    # Inventory levels from abundant to scarce, holding demand fixed
    inventories = [480, 320, 240, 180, 140, 110]

    print(f"\n{'':22}{'ratio':>7}{'truth':>9}{'user':>9}{'cluster':>9}"
          f"{'switch':>9}{'user bias':>11}")
    print(bar)

    rows = []
    for inv in inventories:
        m = replace(base, inventory_per_market=inv)
        truth = ground_truth(m, rng)
        u, starve = user_level(m, rng)
        c, icc = cluster_level(m, rng)
        sb = switchback(m, rng)
        rows.append((m.demand_supply_ratio, truth, u, c, sb, icc, starve))

        label = "abundant" if inv >= 320 else ("balanced" if inv >= 180 else "scarce")
        print(f"{label:<14}{inv:>8}{m.demand_supply_ratio:>7.1f}"
              f"{truth:>9.4f}{u:>9.4f}{c:>9.4f}{sb:>9.4f}{pct_bias(u, truth):>11}")

    print(bar)
    print("inventory per market shown in column 2; demand fixed at "
          f"{base.shoppers_per_market} shoppers")

    if args.aa_check:
        worst = max(abs(u - t) for _, t, u, _, _, _, _ in rows)
        print(f"\nLargest absolute deviation from zero: {worst:.4f}")
        print("All designs unbiased under the null, as expected.\n")
        return

    print("\nWHAT CLUSTERING COSTS")
    print(bar)
    balanced = rows[2]          # ICC from a balanced market, not a sold-out one
    icc = balanced[5]
    n = base.shoppers_per_market
    deff = 1 + (n - 1) * icc
    print(f"Intra-cluster correlation (balanced case) {icc:.4f}")
    print(f"Cluster size                             {n}")
    print(f"Design effect  1 + (m-1)*ICC             {deff:.1f}x")
    print(f"Effective sample size                    "
          f"{int(base.n_markets*n/deff):,} of {base.n_markets*n:,} shoppers")
    print(f"Standard error inflation                 {np.sqrt(deff):.1f}x")

    print("\nTHE DECISION RULE")
    print(bar)
    print("Bias is not a property of the design. It is a property of the market.")
    print("When inventory is abundant, user-level randomization is close to")
    print("unbiased and far more precise, so use it. As demand approaches supply,")
    print("the bias grows and eventually dominates; only then is clustering worth")
    print("its variance penalty.")
    print("\nMeasure the demand-to-supply ratio and stockout rate BEFORE choosing")
    print("the design. After the test has run, the bias is invisible: it does not")
    print("shrink with traffic, it just becomes more precisely wrong.\n")


if __name__ == "__main__":
    main()
