# Use cases

Where the MUO/MTO framework adds value over intention-to-treat (ITT), a plain
LATE, or a conditional average treatment effect (CATE) model — and where it
does not apply.

A note on notation before the use cases: `mteo`'s ATT, ATU, and ATE are
**Kowalski's uniform-weight MTE integrals over the U_D intervals** — ATT
averages the marginal treatment effect MTE(p) over `(0, p_I]` with weight
`1/p_I`, ATU over `(p_C, 1]` with weight `1/(1-p_C)`, and ATE over `(0, 1]`
with weight `1`. These are the parameters the separate-approach estimator
actually targets and reports an identification status for. They are **not**
the classical conditional-on-treatment-status effects: `mteo`'s ATU is not
`E[MTE(U) | D=0]`, because never-takers are not uniformly distributed in
`U_D` — they occupy `(p_I, 1]` but the classical ATU conditions on the whole
`D=0` population (always- and never-takers are disjoint from compliers, but
the *within-group* distribution of `U_D` need not be uniform once you
condition on an observed group rather than integrate over an interval).
Read every ATT/ATU/ATE from this package as "the MTE-reweighted average over
this margin," not as "the average effect on people who are/aren't treated."

## Diagnosing selection into treatment

The untreated-outcome selection test (`result.tests.selection`) is
identified from LATE assumptions alone — no shape restriction, no
extrapolation. It compares `E[Y_U | complier]` to `E[Y_U | never taker]`:
the untreated outcome of the units the instrument *does* move into treatment
versus the untreated outcome of the units it never moves. Because `Y_U` is
the untreated potential outcome, treatment cannot be driving any difference
you find — a nonzero statistic is baseline selection, full stop.

In plain terms, this answers "do the units the instrument moves differ, before
any treatment touches them, from the units it never moves?" That is a question
about who selects into treatment, not about what treatment does to anyone. ITT
cannot answer it: ITT averages over compliers and never-movers together and
mixes selection with effect. A CATE model fit on observed treatment status
faces the reverse problem — it can show treated and untreated units differ on
outcomes, but because take-up is endogenous (driven by the same `U_D` that may
also predict the outcome), it cannot separate "treatment changed their
trajectory" from "the units that take up were always going to look different."
The untreated-outcome test isolates the second piece cleanly, using only the
randomized instrument.

## Expansion decisions via `predict_late`

`result.predict_late(p_lo, p_hi)` reweights the fitted MTE curve to the
subpopulation with `U_D` in `(p_lo, p_hi]` — a different margin than the
LATE the RCT identified. This is the tool for "if we expand treatment to
reach more units, what effect should we expect on the newly-reached
group?" The newly-reached group is not a random sample of current
compliers: it is drawn from `U_D` just above the current `p_I`, i.e. units
that resisted take-up at the current level of encouragement and would only
take up under a stronger push. Kowalski's own motivating exercise — using
one experiment's identified MTE curve to predict outcomes for a
population defined by a different program's take-up margin (the paper's
Oregon/Massachusetts reconciliation) — is the precedent for this kind of
out-of-sample-margin reweighting.

Two caveats travel with every `predict_late` call away from the identified
complier range. First, `mteo` only extrapolates when you ask for it
(`extrapolation="linear"` at `fit()` time, or implicitly whenever `p_lo`/`p_hi`
fall outside the fitted curve's identified span) — the returned `Estimand`
reports `status="extrapolated"` so you can tell the difference from a
point-identified prediction. Second, any prediction outside the complier
support depends on AA.1 (linear MUO/MTO) holding on the new margin, which is
untestable there by construction; the overidentification test only checks
linearity on the range strata make identified, not beyond it. Treat expansion
predictions as "our best linear extrapolation, flagged as such," not as
another point estimate with the same footing as the LATE.

## Interpreting OLS-vs-IV gaps

A common diagnostic question: why does a naive OLS/regression comparison of
treated vs. untreated units show a much bigger (or smaller, or
sign-flipped) gap than the IV/LATE estimate? The gap between the two has two
possible sources, and the MUO/MTO decomposition is what separates them.
**Selection heterogeneity**: treated and untreated units differ at
baseline (captured by the untreated-outcome selection test above), so the
naive comparison bakes in a selection effect that IV, by design, purges.
**Effect heterogeneity**: the treatment effect itself varies across the
population (captured by `result.tests.mte_slope`, the treated-vs-untreated
selection contrast, and the BMW Δ₁=Δ₀ external-validity test), so even a
selection-free estimate at one margin (the LATE) need not generalize to the
ATE or to a different margin.

Running both tests tells you which story to reach for. A large selection
statistic with a flat MTE slope says the OLS/IV gap is almost entirely who
takes up, not what treatment does to them — the LATE likely generalizes
reasonably to nearby margins. A near-zero selection statistic with a large
MTE slope says the reverse: selection isn't the story, but the effect itself
is heterogeneous, and the LATE from this experiment may not represent the ATE
or a different program's margin. Seeing both large is the case that most needs
the full curve rather than any single scalar summary.

## When NOT to use this

The framework needs a real first stage: a **binary randomized instrument**
with a detectable take-up gap `p_I - p_C`. `mteo.fit` raises if `p_I <= p_C`
(monotonicity violated as coded) and warns via `WeakInstrumentWarning` when
the gap is thin (`summary()` also surfaces this warning). With a thin gap,
complier quantities are estimated from very little identifying variation and
any AA.1 extrapolation leans hard on functional form assumed, not observed —
treat point estimates from a thin-gap fit with real skepticism regardless of
what the reported SEs say.

This is not an observational-data tool: v1 has no exclusion-based local-IV,
control-function, or unconfoundedness mode for endogenous (non-randomized)
treatment assignment. If take-up wasn't randomly encouraged, or the
"instrument" is only quasi-random, this package's identification arguments
do not apply as written. Finally, watch for outcomes measured only for
treated units — if `Y` is missing whenever `D=0` (e.g. a usage survey given
only to people who took up the program), the untreated cell means this
framework relies on cannot be computed at all, and no amount of strata or
bootstrap reps fixes a missing-by-construction outcome; that data limitation
has to be solved before this framework's identification results are usable.
