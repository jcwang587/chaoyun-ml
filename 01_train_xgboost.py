import warnings

import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

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
plt.scatter(y1, y1_pred)
plt.xlabel("True Faradic Efficiency")
plt.ylabel("Predicted Faradic Efficiency")
plt.title("Faradic Efficiency")
plt.savefig("figure/faradic_efficiency_xgboost.png")
plt.close()

plt.scatter(y2, y2_pred)
plt.xlabel("True NH3 Yield")
plt.ylabel("Predicted NH3 Yield")
plt.title("NH3 Yield")
plt.savefig("figure/nh3_yield_xgboost.png")
plt.close()

# plot shap values for both targets
explainer = shap.Explainer(model1, X)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, show=False)
plt.savefig("figure/shap_values_xgboost_faradic.png")
plt.close()

# plot shap values for both targets
explainer = shap.Explainer(model2, X)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, show=False)
plt.savefig("figure/shap_values_xgboost_nh3.png")
plt.close()
