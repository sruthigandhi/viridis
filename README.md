# Viridis: Kentucky Farmland Resilience

Viridis is a county-level early-warning system designed to identify Kentucky counties at elevated risk of substantial future agricultural-land decline.

## Research question

Can we identify Kentucky counties at risk of future farmland decline early enough to target interventions that help keep productive agricultural land in use?

## Prediction target

For each county and origin year:

\[
\Delta A_{i,t\rightarrow t+5}
=
\frac{A_{i,t+5}-A_{i,t}}{A_{i,t}}\times100
\]

A county is labeled `high_decline = 1` when its future agricultural-land change falls in the bottom 25% of counties within that origin year.

This is a proxy for substantial future agricultural-land decline. It is NOT treated as a direct measurement of abandonment.

## Validation design

The primary evaluation is temporal:

- 2012 features → 2017 outcome
- 2017 features → 2022 outcome

The model must only use information available at the prediction/origin year.

## Leakage rules

The following may NEVER be model predictors:

- future agricultural acreage
- future acreage change
- `high_decline`
- any feature calculated using the future period
- any transformation of the target

## Current model

The initial clean model intentionally excludes satellite NDVI so that the predictive model can be evaluated independently of Earth Engine.

Future extensions may add additional predictors, but they must pass the same temporal and leakage checks.

## Important

The old `land` repository contains experimental models and earlier feature engineering. Those artifacts are not used by this clean rebuild.
