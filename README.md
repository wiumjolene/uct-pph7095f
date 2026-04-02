# Marital Satisfaction Analysis (Longitudinal Study)

## Overview

This project investigates patterns of marital satisfaction over time
using longitudinal data. The primary objective is to assess how
satisfaction changes as marriage duration increases, while accounting
for individual-level variability and other covariates.

The analysis is implemented in Python and documented using Quarto,
enabling a fully reproducible and transparent analytical workflow.

------------------------------------------------------------------------

## Objectives

-   Examine the trajectory of marital satisfaction over time\
-   Evaluate the effect of demographic and relationship variables\
-   Compare alternative model specifications for longitudinal data\
-   Identify the most appropriate correlation structure for repeated
    observations

------------------------------------------------------------------------

## Data

The dataset consists of repeated observations of individuals over time,
including:

-   Marital satisfaction (outcome variable)\
-   Time (months of marriage)\
-   Demographic variables (e.g., sex, age at marriage)\
-   Relationship characteristics (e.g., cohabitation status)\
-   Individual identifier for longitudinal tracking

------------------------------------------------------------------------

## Methods

### 1. Linear Mixed Effects Models (LME)

-   Implemented using Python (statsmodels)\
-   Random intercept models\
-   Random slope models (time varying by individual)\
-   Model comparison using AIC, BIC, and log-likelihood

### 2. Generalized Estimating Equations (GEE)

-   Implemented using Python (statsmodels)\
-   Exchangeable and AR(1) correlation structures\
-   Robust standard errors\
-   Population-averaged interpretation

------------------------------------------------------------------------

## Tools & Technologies

-   Python (pandas, numpy, statsmodels, scipy, matplotlib)\
-   Quarto for reproducible reporting\
-   Markdown and LaTeX for document formatting\
-   Version-controlled workflow for reproducibility

------------------------------------------------------------------------

## Key Findings

-   Marital satisfaction shows a consistent decline over time\
-   Time is a statistically significant predictor of satisfaction\
-   More complex random-effects structures did not improve model fit due
    to convergence issues\
-   Simpler models provided more stable and interpretable results

------------------------------------------------------------------------

## Model Selection

The preferred model was selected based on: - Goodness-of-fit statistics
(AIC, BIC, log-likelihood)\
- Model convergence\
- Interpretability of parameters

------------------------------------------------------------------------

## Project Structure

    ├── data/               
    ├── code/               
    ├── manuscript/         
    ├── output/             
    └── README.md           

------------------------------------------------------------------------

## Reproducibility

The analysis is fully reproducible using the Quarto document:

    quarto render manuscript.qmd

To render alternative formats:

    quarto render manuscript.qmd --to pdf
    quarto render manuscript.qmd --to markdown

------------------------------------------------------------------------

## Notes

-   Longitudinal structure was carefully accounted for in all models\
-   Model convergence was considered in selecting the final
    specification\
-   Results should be interpreted as average trends over time

------------------------------------------------------------------------