import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Autoregressive, Unstructured
from statsmodels.genmod.families import Gaussian

import config


DATA_PATH = config.DATA_PATH

def load_data(data_path=DATA_PATH):
    """Load dataset and sample 70% of individuals while preserving full profiles."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)

    # Set seed
    np.random.seed(1987090901)

    # Sample 70% of unique IDs
    unique_ids = df["id"].unique()
    sample_size = int(0.7 * len(unique_ids))
    sampled_ids = np.random.choice(unique_ids, size=sample_size, replace=False)

    # Keep full longitudinal profiles
    df = df[df["id"].isin(sampled_ids)].copy()

    # Keep only relevant columns
    columns = [
        "id",
        "obs_time",
        "satisfaction",
        "sex",
        "age_marriage",
        "cohab",
        "income",
        "hw_all",
    ]
    df = df[columns]

    return df


# ###########################################################################
# PART 1: Descriptive data analysis
# ###########################################################################
# -----------------------------
# Question 1: Baseline characteristics
# -----------------------------
def summarize_cont(df, var, by="sex", digits=2):
    """Summarize continuous variables as mean (SD)."""
    out = df.groupby(by)[var].agg(["mean", "std"])
    return out.apply(
        lambda row: f"{row['mean']:.{digits}f} ({row['std']:.{digits}f})",
        axis=1,
    )


def summarize_cat(df, var, by="sex", digits=1):
    """Summarize categorical variables as n (%)."""
    counts = pd.crosstab(df[var], df[by], dropna=False)
    perc = (
        pd.crosstab(df[var], df[by], normalize="columns", dropna=False) * 100
    )
    return counts.astype(int).astype(str) + " (" + perc.round(digits).astype(str) + "%)"


def add_categorical_block(table_rows, df, var, block_name, by="sex"):
    """Append categorical variable block to Table 1."""
    cat_summary = summarize_cat(df, var, by=by)

    # Header row
    header_row = {"Variable": block_name}
    for col in cat_summary.columns:
        header_row[str(col)] = ""
    table_rows.append(header_row)

    # Category rows
    for level in cat_summary.index:
        row = {"Variable": f"  {level}"}
        for col in cat_summary.columns:
            row[str(col)] = cat_summary.loc[level, col]
        table_rows.append(row)


def table1_descriptive(df):
    """Create Table 1: baseline characteristics stratified by sex."""

    # Baseline dataset (first observation per individual)
    df_baseline = (
        df.sort_values(["id", "obs_time"])
        .groupby("id", as_index=False)
        .first()
    )

    # Ensure consistent column order for sex
    sex_levels = sorted(df_baseline["sex"].dropna().unique())

    table1_rows = []

    # Continuous variables
    continuous_vars = [
        ("Age at marriage", "age_marriage"),
        ("Income", "income"),
    ]

    for label, var in continuous_vars:
        summary = summarize_cont(df_baseline, var)
        row = {"Variable": label}
        for sex in sex_levels:
            row[str(sex)] = summary.get(sex, "")
        table1_rows.append(row)

    # Categorical variables
    add_categorical_block(table1_rows, df_baseline, "cohab", "Cohabitation")
    add_categorical_block(
        table1_rows, df_baseline, "hw_all", "Housework allocation"
    )

    return pd.DataFrame(table1_rows)


# -----------------------------
# Question 2: Mean longitudinal profile of satisfaction 
#             measurements by household  workload
# -----------------------------
def plot_mean_profile(df, n_bins=20):
    plt.close("all")

    plot_df = df[["obs_time", "hw_all", "satisfaction"]].dropna().copy()
    plot_df["time_bin"] = pd.cut(plot_df["obs_time"], bins=n_bins)

    summary = (
        plot_df.groupby(["time_bin", "hw_all"], observed=False)
        .agg(
            mean_satisfaction=("satisfaction", "mean"),
            sd_satisfaction=("satisfaction", "std"),
            n=("satisfaction", "size"),
        )
        .reset_index()
    )

    summary["time_mid"] = summary["time_bin"].apply(lambda x: x.mid)
    summary["se"] = summary["sd_satisfaction"] / np.sqrt(summary["n"])
    summary["lower"] = summary["mean_satisfaction"] - 1.96 * summary["se"]
    summary["upper"] = summary["mean_satisfaction"] + 1.96 * summary["se"]

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for hw in sorted(summary["hw_all"].dropna().unique()):
        subset = summary[summary["hw_all"] == hw].sort_values("time_mid")
        ax.plot(
            subset["time_mid"],
            subset["mean_satisfaction"],
            linewidth=2,
            label=f"{hw}",
        )
        ax.fill_between(
            subset["time_mid"],
            subset["lower"],
            subset["upper"],
            alpha=0.2,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Mean satisfaction")
    ax.set_title("Mean longitudinal profile of satisfaction by household workload")
    ax.legend(title="Household workload")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()


# -----------------------------
# Question 3: Spaghetti plot of satisfaction 
#             measurements by prematital cohabitation
# -----------------------------
def plot_spaghetti(df):

    ids = df["id"].unique()
    plot_df = df[df["id"].isin(ids)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.5), sharey=True)

    for i, cohab_value in enumerate(sorted(plot_df["cohab"].dropna().unique())):
        subset = plot_df[plot_df["cohab"] == cohab_value]

        for pid in subset["id"].unique():
            person = subset[subset["id"] == pid].sort_values("obs_time")

            axes[i].plot(
                person["obs_time"],
                person["satisfaction"],
                alpha=0.2,
                linewidth=1
            )

        axes[i].set_title(f"Cohab: {cohab_value}")
        axes[i].set_xlabel("Time")

    axes[0].set_ylabel("Satisfaction")

    plt.tight_layout()



# ###########################################################################
# PART 2: Generalized estimating equations (GEEs)
# ###########################################################################
def fit_gee_models(df):
    dat = df.dropna().copy()

    dat = dat.sort_values(["id", "obs_time"]).reset_index(drop=True)

    # visit order within person
    dat["time_index"] = dat.groupby("id").cumcount().astype(int)

    formula = (
        "satisfaction ~ time_index + sex + age_marriage + "
        "cohab + income + hw_all"
    )

    # AR(1): all visits
    gee_ar1 = smf.gee(
        formula=formula,
        groups="id",
        time="time_index",
        data=dat,
        family=Gaussian(),
        cov_struct=Autoregressive(grid=True),
    ).fit()

    # Unstructured: restricted visits
    dat_un = dat[dat["time_index"] <= 3].copy()

    gee_un = smf.gee(
        formula=formula,
        groups="id",
        time="time_index",
        data=dat_un,
        family=Gaussian(),
        cov_struct=Unstructured(),
    ).fit()

    return gee_ar1, gee_un


def tidy_gee(result, model_name):
    """Extract coefficients, SE, and 95% CI from a fitted GEE model."""

    params = result.params
    se = result.bse
    ci = result.conf_int()

    out = pd.DataFrame({
        "Variable": params.index,
        "Estimate": params.values,
        "SE": se.values,
        "CI Lower": ci[0].values,
        "CI Upper": ci[1].values,
    })

    # Format nicely
    out["Estimate"] = out["Estimate"].round(3)
    out["SE"] = out["SE"].round(3)
    out["95% CI"] = (
        out["CI Lower"].round(3).astype(str)
        + " to "
        + out["CI Upper"].round(3).astype(str)
    )

    out = out.drop(columns=["CI Lower", "CI Upper"])
    out["Model"] = model_name

    return out


def table2_gee(df):
    gee_ar1, gee_un = fit_gee_models(df)

    t1 = tidy_gee(gee_ar1, "GEE 1: AR(1)")
    t2 = tidy_gee(gee_un, "GEE 2: Unstructured")

    table2 = pd.concat([t1, t2], ignore_index=True)

    # Optional: nicer ordering
    table2 = table2[
        ["Model", "Variable", "Estimate", "SE", "95% CI"]
    ]

    return table2


# ###########################################################################
# PART 3: Linear mixed effects models (LMEs)
# ###########################################################################
def fit_lme_models(df):
    dat = df[
        [
            "id",
            "obs_time",
            "satisfaction",
            "sex",
            "age_marriage",
            "cohab",
            "income",
            "hw_all",
        ]
    ].dropna().copy()

    dat = dat.sort_values(["id", "obs_time"]).reset_index(drop=True)

    # Use within-person time index
    dat["time_index"] = dat.groupby("id").cumcount().astype(int)

    formula = (
        "satisfaction ~ time_index + sex + age_marriage + "
        "cohab + income + hw_all"
    )

    # LME 1: Random intercept only
    lme1 = smf.mixedlm(
        formula=formula,
        data=dat,
        groups=dat["id"],   # random intercept
    ).fit()

    # LME 2: Random intercept + random slope for time
    lme2 = smf.mixedlm(
        formula=formula,
        data=dat,
        groups=dat["id"],
        re_formula="~time_index",  # adds random slope
    ).fit()

    return lme1, lme2


def tidy_lme(result, model_name):
    params = result.params
    se = result.bse
    ci = result.conf_int()

    out = pd.DataFrame({
        "Variable": params.index,
        "Estimate": params.values,
        "SE": se.values,
        "CI Lower": ci[0].values,
        "CI Upper": ci[1].values,
    })

    out["Estimate"] = out["Estimate"].round(3)
    out["SE"] = out["SE"].round(3)
    out["95% CI"] = (
        out["CI Lower"].round(3).astype(str)
        + " to "
        + out["CI Upper"].round(3).astype(str)
    )

    out = out.drop(columns=["CI Lower", "CI Upper"])
    out["Model"] = model_name

    return out


def table3_lme(df):
    lme1, lme2 = fit_lme_models(df)

    t1 = tidy_lme(lme1, "LME 1: Random intercept")
    t2 = tidy_lme(lme2, "LME 2: Random intercept + slope")

    table = pd.concat([t1, t2], ignore_index=True)

    table = table[
        ["Model", "Variable", "Estimate", "SE", "95% CI"]
    ]

    return table


