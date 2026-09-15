# Applicability experiment: research basis

I used two focused Exa searches, requesting ten result slots in total, and inspected selected sections from three primary works. Several search results were duplicate versions of SelectiveNet. This is a targeted extension of the earlier literature review, not ten independent papers or three complete paper readings.

Selective prediction measures error together with the fraction of queries a model accepts. SelectiveNet jointly learns prediction and rejection; GG-GUARD-001 instead holds the existing Gaussian generator fixed and learns a separate experimental selection function. I use the paper's risk/coverage distinction, without claiming to reproduce its architecture or results. I inspected the publisher abstract and indexed method excerpts. [Geifman and El-Yaniv, 2019](https://proceedings.mlr.press/v97/geifman19a.html).

Conformal Risk Control requires a bounded monotone loss and exchangeable loss functions for its basic guarantee. The error fraction *conditional on acceptance* can increase or decrease when a threshold changes, so that theorem does not directly certify the selective risk used here. I inspected the introduction, algorithm, assumptions and Theorem 1. The experiment therefore labels threshold selection as empirical calibration and tests the reserved families without a distribution-free guarantee. [Angelopoulos et al., Conformal Risk Control](https://arxiv.org/html/2208.02814).

Conformal Prediction Beyond Exchangeability studies how violations of exchangeability affect predictive coverage and develops modifications for those settings. I inspected its abstract and introductory assumptions. It does not establish validity for arbitrary unseen mechanisms in this experiment. Keeping worlds together during splitting and resampling avoids counting correlated query rows as independent experiments; it does not eliminate family shift. [Barber et al., 2023](https://arxiv.org/abs/2202.13415).

## Decisions frozen before the final cohort

- Fit a 16-input, 16-hidden-unit validity network with factual prediction-outcome labels. Calibrate its probability and choose its acceptance threshold on two additional, disjoint world banks.
- Compare it with always predict, a residual/uncertainty guard and a distance-to-observations guard. Use always abstain as the zero-coverage reference. Tune fixed-guard thresholds with the same threshold-selection outcomes.
- Evaluate error among accepted queries, acceptance, false acceptance/rejection, latent prediction MSE, Brier score, log loss, reliability bins and risk/coverage curves. Pair and bootstrap worlds within each specified family.
- Charge the network's extra teaching data and fitting. Freeze one optimizer seed and disclose the lack of an initialization-robustness panel.
- Include an observational-alias diagnostic: a local exception can be zero at every acquired observation and large at an unmeasured query. No confidence score can identify that exception from the unchanged prefix alone.

## Source-packet alignment

The second packet's `docs/AUTONOMOUS_COOKBOOK.md`, W05 and W08, requires adequate uncertainty, positive and counterexample training, learned scopes compared with engineered scopes, and common-owner transfer before claiming integrated understanding. W08's stated failure response is to retain a conditional tool and inspect guard labels and distillation targets. This experiment uses observed sensor outcomes rather than the old model's answers as correctness labels. It tests a component gate before considering a shared-owner revision.

The packet remains preserved under archive SHA-256 `4e9b829f67aca7bbb543b0c03eff022b019b67698a0f27f899fbd7a8186e814f`. The original architecture, common owner, graph contracts, earlier models and failed experiments remain intact.
