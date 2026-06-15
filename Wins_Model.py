import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('refined_feature_data.csv')

#removed features due to Multicollinearity -  'ERA','HR','E','Attendance','Allstar_Num','Batting_Avg'

X = df[['Total_Salary_Adjusted','Pitcher_Batter_Ratio','Max_Salary_Adjusted',
        'Stdev_Salary_Adjusted','Pitcher_Salary_Adjusted','Batter_Salary_Adjusted']]

Y = df[['W']]

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

model = make_pipeline(StandardScaler(), LinearRegression())

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(X_scaled)

X_OLS = sm.add_constant(X_scaled)
ols_model = sm.OLS(Y, X_OLS).fit()
print(f'\nStatsmodels OLS Summary')
print(ols_model.summary())

kf = KFold(n_splits=6, random_state=0, shuffle=True)

scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring = 'r2')

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print(f"\nLinear Regression Model Results")
print(f"Mean Absolute Error (MAE): {metrics.mean_absolute_error(Y_test, y_pred):.2f}")
print(f"Mean Squared Error (MSE): {metrics.mean_squared_error(Y_test, y_pred):.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(Y_test, y_pred)):.2f}")
print(f"R-squared (R^2) score: {model.score(X_test, Y_test):.2f}")

print(f"\nCross-Validated Linear Regression Model Results")
print(f"R-squared scores for each fold: {scores}")
print(f"Mean R-squared across all folds: {np.mean(scores):.2f}")
print(f"Standard deviation of R-squared: {np.std(scores):.2f}")