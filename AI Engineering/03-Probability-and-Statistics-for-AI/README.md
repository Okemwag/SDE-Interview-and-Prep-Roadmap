# Probability and Statistics for AI — Deep Track

Every ML model is a statistical claim: "given data sampled this way, this function's outputs approximate this conditional distribution." Senior engineers get paid for knowing when that claim breaks — when accuracy flatters a useless model, when an offline win is a sampling artifact, when a p-value is theater, when probabilities are rankings in disguise. With an actuarial background you already own the underlying math; this track's job is to wire each concept into the exact place it bites in production ML systems, with full derivations, runnable simulations, incident-style war stories, and senior-level interview drills.

This folder replaces the single-file overview with six deep sub-guides. Each one derives the math line by line, verifies it with complete NumPy/SciPy simulations, and connects it to production failure modes you will actually be asked about in senior interviews.

Part of the [Senior AI Engineer Roadmap](../00-Senior-AI-Engineer-Roadmap.md) — Phase 1.

---

## The Six Sub-Guides

| # | Guide | What it covers | Why it matters in production |
| --- | --- | --- | --- |
| 1 | [Probability Foundations and Distributions](./01-Probability-Foundations-and-Distributions.md) | Random variables, PMF/PDF/CDF, expectation/variance/covariance with proofs, the distribution zoo mapped to ML homes, CLT by simulation, joint/marginal/conditional, correlation vs independence | Choosing losses and links, recognizing fat tails, knowing when averages are trustworthy |
| 2 | [Bayes and Bayesian Reasoning](./02-Bayes-and-Bayesian-Reasoning.md) | Bayes' theorem derived, base-rate fallacy fully worked, sequential updating, conjugate pairs (Beta-Binomial CTR), credible vs confidence intervals, Bayesian A/B testing, naive Bayes from scratch, shrinkage and cold start | Alert precision at low prevalence, small-sample rates, exploration, hierarchical pooling |
| 3 | [Estimation, MLE, and Loss Functions](./03-Estimation-MLE-and-Loss-Functions.md) | Estimator properties, MLE derived in general, Gaussian→MSE, Bernoulli→BCE, multinomial→softmax loss, MAP→regularization (L2/L1 priors), bias-variance decomposition, log-sum-exp | Every standard loss is MLE under a noise assumption — pick the wrong assumption, get the wrong model |
| 4 | [Statistical Inference and Testing](./04-Statistical-Inference-and-Testing.md) | Sampling distributions, confidence intervals, bootstrap from scratch (AUC CI), hypothesis testing mechanics, power and sample size, multiple comparisons (Bonferroni, BH), p-hacking, paired model comparisons, McNemar | Knowing whether a metric delta is signal or noise before you ship |
| 5 | [Calibration and Uncertainty](./05-Calibration-and-Uncertainty.md) | Calibration formally, reliability diagrams, ECE/MCE, why modern models miscalibrate, Platt/isotonic/temperature scaling implemented, proper scoring rules, aleatoric vs epistemic, conformal prediction, review routing | Uncalibrated probabilities silently poison every expected-value decision downstream |
| 6 | [Causality and A/B Testing](./06-Causality-and-AB-Testing.md) | Confounding/selection/collider bias with numbers, Simpson's paradox worked, ML feedback loops, randomized experiments, A/B testing end to end, CUPED, sequential pitfalls, offline-online gaps, diff-in-diff | The senior-level material: why models trained on their own outputs rot, and why offline wins vanish online |

## Suggested Order

Work through them in numeric order — each builds on the previous:

1. **Foundations first** (Guide 1): the distribution zoo and expectation algebra are the vocabulary everything else uses.
2. **Bayes** (Guide 2): conditional probability is the skeleton of both ML training and production decision-making.
3. **MLE and losses** (Guide 3): connects Guides 1-2 to the losses you actually train with. This is the highest-leverage interview material.
4. **Inference and testing** (Guide 4): how to know whether any measured improvement is real.
5. **Calibration** (Guide 5): what your model's probabilities actually mean, and how to fix them when they lie.
6. **Causality and A/B** (Guide 6): the capstone — moving from "the model predicts" to "the intervention works."

If you are cramming for an interview: Guides 3 → 2 → 6 cover the most commonly probed senior material (loss derivations, base rates, and offline-online gaps).

## How to Study Each Guide

- Run every code block — each is self-contained and seeded, with expected output in comments. If your numbers differ wildly, investigate; that is the point.
- Reproduce each derivation on paper before reading the next section.
- Do the Interview Drills cold, out loud, including the follow-up chains — the follow-ups are where senior loops are won or lost.

Back to the [AI Engineering track index](../README.md). The original single-file overview remains at [../03-Probability-and-Statistics-for-AI.md](../03-Probability-and-Statistics-for-AI.md).
