import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.display import Markdown, display
from statsmodels.genmod.cov_struct import Autoregressive, Unstructured, Independence
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

    # Set obs_time as integer 
    df["obs_time"] = df["obs_time"].astype(int)

    # Sort df (visit order within person) and create visit variable
    df = df.sort_values(["id", "obs_time"]).reset_index(drop=True)
    df["visit"] = df.groupby("id").cumcount()

    # Keep relevant columns
    columns = [
        "id",
        "obs_time",
        "satisfaction",
        "sex",
        "age_marriage",
        "cohab",
        "income",
        "hw_all",
        "visit",
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
        row = {"Variable": f" > {level}"}
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
        row = {"Variable": f"{label} "}
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

    ax.set_xlabel("Time (Months since marriage)")
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

    plt.style.use("seaborn-v0_8")
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

        if cohab_value == 0:
            cohab_label = "No premarital cohabitation"
        else:            
            cohab_label = "Premarital cohabitation"

        # axes[i].set_title(f"Cohabitate: {cohab_value}")
        axes[i].set_title(f"{cohab_label}")
        axes[i].set_xlabel("Time (Months since marriage)")

    axes[0].set_ylabel("Satisfaction")

    plt.tight_layout()


# ###########################################################################
# PART 2: Generalized estimating equations (GEEs)
# ###########################################################################
def fit_gee_models(df):
    dat = df.dropna().copy()

    formula = (
        "satisfaction ~ obs_time + sex + age_marriage + "
        "cohab + income + hw_all"
    )

    # AR(1): all visits
    gee_ar1 = smf.gee(
        formula=formula,
        groups="id",
        time="obs_time",
        data=dat,
        family=Gaussian(),
        cov_struct=Autoregressive(grid=True),
    ).fit()

    # Unstructured: restricted visits
    dat_un = dat[dat["visit"] <= 14].copy()
    gee_un = smf.gee(
        formula=formula,
        groups="id",
        time="visit",  # visit order, but obs_time is used in the formula
        data=dat_un,
        family=Gaussian(),
        cov_struct=Unstructured(), 
    ).fit()

    return gee_ar1, gee_un


def format_model(result):
    df_out = pd.DataFrame({
        "Variable": result.params.index,
        "Estimate (SE)": [
            f"{est:.3f} ({se:.3f})"
            for est, se in zip(result.params, result.bse)
        ],
        "95% CI": [
            f"{l:.3f} to {u:.3f}"
            for l, u in result.conf_int().values
        ],
    })
    return df_out.set_index("Variable")


def question2_pretty(df):
    gee_ar1, gee_un = fit_gee_models(df)

    ar1 = format_model(gee_ar1)
    un = format_model(gee_un)

    # Rename inner columns
    ar1.columns = ["Estimate (SE)", "95% CI"]
    un.columns = ["Estimate (SE)", "95% CI"]

    # Create MultiIndex columns
    ar1.columns = pd.MultiIndex.from_product(
        [["AR(1)"], ar1.columns]
    )
    un.columns = pd.MultiIndex.from_product(
        [["Unstructured"], un.columns]
    )

    table = pd.concat([ar1, un], axis=1)

    # Reset index to bring Variable back
    table = table.reset_index()

    # Rename variables
    rename_map = {
        "Intercept": "Intercept",
        "sex[T.Male]": "Male (vs Female)",
        "obs_time": "Time",
        "age_marriage": "Age at marriage",
        "cohab": "Premarital cohabitation",
        "income": "Income",
        "hw_all": "Household workload",
    }

    table["Variable"] = table["Variable"].replace(rename_map)

    return table


# ###########################################################################
# PART 3: Linear mixed effects models (LMEs)
# ###########################################################################
def fit_lme_models(df):
    dat = df.dropna().copy()

    dat = dat.sort_values(["id", "obs_time"]).reset_index(drop=True)

    formula = (
        "satisfaction ~ obs_time + sex + age_marriage + "
        "cohab + income + hw_all"
    )

    # LME 1: Random intercept only
    lme1 = smf.mixedlm(
        formula=formula,
        data=dat,
        groups=dat["id"],   # random intercept
    ).fit(reml=False)

    # LME 2: Random intercept + random slope for time
    lme2 = smf.mixedlm(
        formula=formula,
        data=dat,
        groups=dat["id"],
        re_formula="~visit",  # add random slope
    ).fit(reml=False)

    return lme1, lme2


def format_lme_fixed(result):
    """Return fixed effects only from an LME model."""
    fe_params = result.fe_params
    fe_se = result.bse_fe
    fe_ci = result.conf_int().loc[fe_params.index]

    df_out = pd.DataFrame({
        "Variable": fe_params.index,
        "Estimate (SE)": [
            f"{est:.3f} ({se:.3f})"
            for est, se in zip(fe_params, fe_se)
        ],
        "95% CI": [
            f"{l:.3f} to {u:.3f}"
            for l, u in fe_ci.values
        ],
    })
    return df_out.set_index("Variable")


def question3_pretty(df):
    lme1, lme2 = fit_lme_models(df)

    t1 = format_lme_fixed(lme1)
    t2 = format_lme_fixed(lme2)

    t1.columns = ["Estimate (SE)", "95% CI"]
    t2.columns = ["Estimate (SE)", "95% CI"]

    t1.columns = pd.MultiIndex.from_product(
        [["LME 1: Random intercept"], t1.columns]
    )
    t2.columns = pd.MultiIndex.from_product(
        [["LME 2: Random intercept + slope"], t2.columns]
    )

    table = pd.concat([t1, t2], axis=1).reset_index()

    rename_map = {
        "Intercept": "Intercept",
        "sex[T.Male]": "Male (vs Female)",
        "time_index": "Time",
        "obs_time": "Time",
        "age_marriage": "Age at marriage",
        "cohab": "Premarital cohabitation",
        "income": "Income",
        "hw_all": "Household workload",
    }

    table["Variable"] = table["Variable"].replace(rename_map)

    return table


def lme_fit_stats_text(lme1, lme2):
    # lme1, lme2 = fit_lme_models(df)
    return display(Markdown(
    f"""
*Note:* Fixed effects are presented as estimate (SE) with 95% confidence intervals.  
*Model fit statistics:*  
> LME 1 AIC: {lme1.aic:.2f}, BIC: {lme1.bic:.2f}, Log-likelihood: {lme1.llf:.2f};  
> LME 2 AIC: {lme2.aic:.2f}, BIC: {lme2.bic:.2f}, Log-likelihood: {lme2.llf:.2f}.
"""
))


# ###########################################################################
# PART 4: Model validation
# ###########################################################################
def plot_residuals(model):
    fitted = model.fittedvalues
    residuals = model.resid

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(6, 4))
    plt.scatter(fitted, residuals, alpha=0.5)

    plt.axhline(0, linestyle="--")

    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted Values")

    plt.tight_layout()


def plot_random_effects(model):
    # Extract random intercepts
    re = pd.DataFrame(model.random_effects).T

    intercepts = re.iloc[:, 0]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    res = stats.probplot(intercepts, dist="norm", plot=ax)

    # Change colors
    ax.get_lines()[0].set_color("teal")   # data points
    ax.get_lines()[1].set_color("black")  # reference line

    ax.set_title("QQ Plot of Random Intercepts")

    plt.tight_layout()