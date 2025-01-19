import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# 1) Generate synthetic regression data
#    - n_samples=100, n_features=5, with 3 truly informative features
X, y = make_regression(
    n_samples=100, 
    n_features=5, 
    n_informative=3, 
    noise=10, 
    random_state=42
)

# 2) Create a new column that is *strongly correlated* to the first column (index 0)
#    This simulates having redundant or nearly duplicate features.
#    We'll call this correlated column X_corr.
X_corr = X[:, [0]] + 0.05 * np.random.randn(100, 1)  # small random noise

# 3) Append that correlated column to our existing feature matrix
#    Now we have 6 features in total: original 5 + 1 correlated
X = np.hstack([X, X_corr])

# 4) Fit an Ordinary Linear Regression model
lin_model = LinearRegression()
lin_model.fit(X, y)

# 5) Fit a Lasso model (with alpha=0.5 just as an example)
lasso_model = Lasso(alpha=1)
lasso_model.fit(X, y)

# 6) Evaluate both models on the *same* training data
lin_preds = lin_model.predict(X)
lasso_preds = lasso_model.predict(X)

lin_r2 = r2_score(y, lin_preds)
lasso_r2 = r2_score(y, lasso_preds)

lin_mse = mean_squared_error(y, lin_preds)
lasso_mse = mean_squared_error(y, lasso_preds)

# 7) Print out the coefficients
print("=== Ordinary Linear Regression ===")
print("Coefficients:", lin_model.coef_)
print("Intercept:", lin_model.intercept_)
print(f"R2 Score: {lin_r2:.3f}")
print(f"MSE: {lin_mse:.3f}")

print("\n=== Lasso Regression (alpha=0.5) ===")
print("Coefficients:", lasso_model.coef_)
print("Intercept:", lasso_model.intercept_)
print(f"R2 Score: {lasso_r2:.3f}")
print(f"MSE: {lasso_mse:.3f}")
