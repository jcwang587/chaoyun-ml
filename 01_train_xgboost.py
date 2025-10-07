import warnings

import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import r2_score

from plot_utils import plot_scatter

warnings.filterwarnings("ignore")

# Set font to Arial Bold
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"

# Load data
data = pd.read_csv("data/processed/data.csv")
X = data.drop(["Faradic Effecency", "NH3 Yield"], axis=1)
y1 = data["Faradic Effecency"]
y2 = data["NH3 Yield"]

# use full data to train xgboost
model1 = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model2 = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
# fit and print r2 score for both targets
model1.fit(X, y1)
y1_pred = model1.predict(X)
print(r2_score(y1, y1_pred))

model2.fit(X, y2)
y2_pred = model2.predict(X)
print(r2_score(y2, y2_pred))

# plot scatter plot for both targets
df_faradic = pd.DataFrame({"true": y1, "pred": y1_pred})
df_nh3 = pd.DataFrame({"true": y2, "pred": y2_pred})
plot_scatter(
    df_faradic,
    xlabel="Experimental Faradic Efficiency",
    ylabel="Predicted Faradic Efficiency",
    out_png="figure/faradic_efficiency_xgboost.png",
)
plot_scatter(
    df_nh3,
    xlabel="Experimental NH3 Yield",
    ylabel="Predicted NH3 Yield",
    out_png="figure/nh3_yield_xgboost.png",
)

# plot shap values for both targets
plt.figure()
explainer = shap.Explainer(model1, X)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, show=False)
cb_ax = plt.gcf().axes[-1]  # colorbar axis is always the last one
cb_ax.set_ylabel("Feature value", fontweight="bold", fontfamily="Arial")
plt.xlabel("SHAP value", fontweight="bold", fontfamily="Arial")
# Add frame to the plot
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color("black")
# Move y-axis tick labels closer to the axis and add outward ticks
ax.tick_params(axis="y", pad=-15)
plt.tight_layout()
plt.savefig("figure/shap_values_xgboost_faradic.png", dpi=300, bbox_inches="tight")
plt.close()

# plot shap values for both targets
plt.figure()
explainer = shap.Explainer(model2, X)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, show=False)
cb_ax = plt.gcf().axes[-1]  # colorbar axis is always the last one
cb_ax.set_ylabel("Feature value", fontweight="bold", fontfamily="Arial")
plt.xlabel("SHAP value", fontweight="bold", fontfamily="Arial")
# Add frame to the plot
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color("black")
# Move y-axis tick labels closer to the axis and add outward ticks
ax.tick_params(axis="y", pad=-15)
plt.tight_layout()
plt.savefig("figure/shap_values_xgboost_nh3.png", dpi=300, bbox_inches="tight")
plt.close()
