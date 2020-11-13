# # Exploration of the crash severity information in CAS data
#
# In this notebook, we will explore the severity of crashes, as it will be the
# target of our predictive models.

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sb

from crash_prediction import cas_data

# set seaborn default style
sb.set()

# But first, we ensure we have the data or download it if needed

dset_path = Path("..") / "data" / "cas_dataset.csv"
if not dset_path.exists():
    dset_path.parent.mkdir(parents=True, exist_ok=True)
    cas_data.download(dset_path)

# and load it.

dset = pd.read_csv(dset_path)
dset.head()

# The CAS dataset has 4 features that can be associated with the crash severity:
#
# - `crashSeverity`, severity of a crash, determined by the worst injury
#    sustained in the crash at time of entry,
# - `fatalCount`, count of the number of fatal casualties associated with this
#   crash,
# - `minorInjuryCount`, count of the number of minor injuries associated with
#   this crash,
# - `seriousInjuryCount`, count of the number of serious injuries associated
#   with this crash.

severity_features = [
    "fatalCount",
    "seriousInjuryCount",
    "minorInjuryCount",
    "crashSeverity",
]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for ax, feat in zip(axes.flat, severity_features):
    counts = dset[feat].value_counts(dropna=False)
    counts.plot.bar(ylabel="# crashes", title=feat, ax=ax)
    ax.set(yscale="log")
fig.tight_layout()

# To check the geographical distribution, we will focus on Auckland and replace
# discrete levels of `crashSeverity` with number to ease plotting.

dset_auckland = dset[dset["X"].between(174.7, 174.9) & dset["Y"].between(-37, -36.8)]

mapping = {
    "Non-Injury Crash": 1,
    "Minor Crash": 2,
    "Serious Crash": 3,
    "Fatal Crash": 4,
}
dset_auckland = dset_auckland.replace({"crashSeverity": mapping})

# Given the data set imbalance, we plot the local maxima to better see the
# location of more severe car crashes.

fig, axes = plt.subplots(2, 2, figsize=(15, 15))
for ax, feat in zip(axes.flat, severity_features):
    dset_auckland.plot.hexbin(
        "X",
        "Y",
        feat,
        gridsize=500,
        reduce_C_function=np.max,
        cmap="BuPu",
        title=feat,
        ax=ax,
        sharex=False,
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
fig.tight_layout()

# Few remarks coming from these plots:
#
# - fatal counts are (hopefully) very low,
# - crashes with serious injuries are also very sparse,
# - crashes with minor injuries are denser and seem to follow major axes,
# - the crash severity feature looks like the most homogeneous feature, yet
#   highlighting some roads more than others.
#
# The crash severity is probably a good go-to target, as it's quite
# interpretable and actionable. The corresponding ML problem is a supervised
# multi-class prediction problem.

# To simplify the problem, we can also just try to predict if a crash is going
# to involve an injury (minor, severe or fatal) or none. Here is how it would
# look like in Auckland

dset_auckland["injuryCrash"] = (dset_auckland["crashSeverity"] > 1) * 1.0
dset_auckland.plot.hexbin(
    "X",
    "Y",
    "injuryCrash",
    gridsize=500,
    cmap="BuPu",
    title="Crash with injury",
    sharex=False,
    figsize=(10, 10),
)

# Interestingly, the major axes do not pop up as saliently here, as we are
# averaging instead of taking the local maxima.

# This brings us to to the another question: is the fraction of crash with
# injuries constant fraction of the number of crashes in an area? This would
# imply that a simple binomial model can model locally binned data.

# We first discretize space into 0.01° wide cells and count the total number of
# crashes in each cell as well as the number of crashes with injuries.

# +
dset["X_bin"] = pd.cut(
    dset["X"], pd.interval_range(dset.X.min(), dset.X.max(), freq=0.01)
)
dset["Y_bin"] = pd.cut(
    dset["Y"], pd.interval_range(dset.Y.min(), dset.Y.max(), freq=0.01)
)

counts = (
    dset.groupby(["X_bin", "Y_bin"], observed=True).size().reset_index(name="crash")
)

injury_counts = (
    dset.groupby(["X_bin", "Y_bin"], observed=True)
    .apply(lambda x: (x["crashSeverity"] != "Non-Injury Crash").sum())
    .reset_index(name="injury")
)

counts = counts.merge(injury_counts)
# -

# For each number of crashes in cells, we can check the fraction of crashes with
# injuries. Here we see that cells with 1 or few crashes have a nearly 50/50
# chance of injuries, compared to cells with a larger number of accidents, where
# it goes down to about 20%.

injury_fraction = counts.groupby("crash").apply(
    lambda x: x["injury"].sum() / x["crash"].sum()
)
ax = injury_fraction.plot(style=".", ylabel="fraction of injuries", figsize=(10, 7))
ax.set_xscale("log")

# Then we can also check how good is a binomial distribution at modeling binned
# data, using it to derive a 95% predictive interval.

ratio = counts["injury"].sum() / counts["crash"].sum()
xs = np.arange(1, counts["crash"].max() + 1)
pred_intervals = st.binom(xs, ratio).ppf([[0.025], [0.975]])

# +
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

counts.plot.scatter(x="crash", y="injury", alpha=0.3, c="b", s=2, ax=axes[0])
axes[0].fill_between(
    xs,
    pred_intervals[0],
    pred_intervals[1],
    alpha=0.3,
    color="r",
    label="95% equal-tail interval for binomial",
)
axes[0].legend()

counts.plot.scatter(x="crash", y="injury", alpha=0.3, c="b", s=2, ax=axes[1])
axes[1].fill_between(
    xs,
    pred_intervals[0],
    pred_intervals[1],
    alpha=0.3,
    color="r",
    label="95% equal-tail interval for binomial",
)
axes[1].legend()
axes[1].set_xscale("log")
axes[1].set_yscale("log")
# -

# The predictive interval seems to have a poor coverage, overshooting the high
# counts regions and being to narrow for the regions with hundreds of crashes.
# We can compute the empirical coverage of these interval to check this.

counts["covered"] = counts["injury"].between(
    pred_intervals[0, counts["crash"] - 1], pred_intervals[1, counts["crash"] - 1]
)
print(f"95% predictive interval has {counts['covered'].mean() * 100:.2f}%.")

print("95% predictive interval coverage per quartile of crash counts:")
mask = counts["crash"] > 1
counts[mask].groupby(pd.qcut(counts.loc[mask, "crash"], 4))["covered"].mean()

# So it turns out that on a macro scale, the coverage of this simple model is
# quite good, but if we split by number of crashes, the coverage isn't so good
# anymore for the cells with higher number of crashes.
#
# Hence, including the number of crashes in a vicinity could be an relevant
# predictor for the probability of crash with injury.

# ---
# ## Original computing environment

# !date -R

# !uname -a

# !pip freeze
