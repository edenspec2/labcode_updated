import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from adjustText import adjust_text

# 1. Generate Synthetic Data
np.random.seed(42)  # For reproducibility
n_samples = 100

# Independent variable
X = np.linspace(0, 10, n_samples)
# Dependent variable with some noise
y = 3 * X + 7 + np.random.normal(0, 10, n_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Measured': X,
    'y': y
})

# Optional: Assign labels to each data point (e.g., "Point 1", "Point 2", ...)
# data['Label'] = [f'Point {i+1}' for i in range(n_samples)]

# 2. Fit Linear Regression Model using Statsmodels
# Add a constant term for the intercept
X_const = sm.add_constant(data['Measured'])
model = sm.OLS(data['y'], X_const).fit()

# 3. Calculate Prediction Intervals
predictions = model.get_prediction(X_const)
pred_summary = predictions.summary_frame(alpha=0.10)  # 90% prediction interval

# Add prediction intervals to the DataFrame
data['Predicted'] = pred_summary['mean']
data['pi_lower'] = pred_summary['obs_ci_lower']
data['pi_upper'] = pred_summary['obs_ci_upper']

# 4. Plotting using Matplotlib
plt.figure(figsize=(12, 8))

# Scatter Plot of Measured vs. Predicted
scatter = plt.scatter(
    data['Measured'], 
    data['Predicted'], 
    color='blue', 
    edgecolor='white', 
    s=100
)

# Plot Regression Line
plt.plot(
    data['Measured'], 
    data['Predicted'], 
    color='red', 
    linestyle='-', 
    linewidth=2, 
    label='Regression Line'
)

# Plot Prediction Interval Bounds as Lines
plt.plot(
    data['Measured'], 
    data['pi_lower'], 
    color='gray', 
    linestyle='--', 
    linewidth=1, 
    label='90% Prediction Interval'
)
plt.plot(
    data['Measured'], 
    data['pi_upper'], 
    color='gray', 
    linestyle='--', 
    linewidth=1
)

# # Add Labels to Data Points
# texts = []
# for _, row in data.iterrows():
#     texts.append(plt.text(
#         row['Measured'], 
#         row['Predicted'], 
#         row['Label'], 
#         fontsize=9,
#         ha='center', 
#         va='bottom',
#         color='darkgreen'
#     ))

# # Adjust Text to Prevent Overlapping
# adjust_text(
#     texts,
#     arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
#     expand_points=(1.2, 1.2),
#     force_text=0.2,
#     force_points=0.2,
#     lim=1000
# )

# Customize Plot Aesthetics
plt.xlabel('Measured', fontsize=14, fontweight='bold')
plt.ylabel('Predicted', fontsize=14, fontweight='bold')
plt.title('Regression Analysis with 90% Prediction Intervals', fontsize=16, fontweight='bold')

# Create Custom Legend Handles
# import matplotlib.lines as mlines
# regression_line_handle = mlines.Line2D([], [], color='red', linestyle='-', linewidth=2, label='Regression Line')
# prediction_interval_handle = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1, label='90% Prediction Interval')
# data_points_handle = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Data Points')

# Add Legend
# plt.legend(handles=[data_points_handle, regression_line_handle, prediction_interval_handle], loc='upper left')

# Set Aspect Ratio to 1
plt.gca().set_aspect('equal', adjustable='box')

# Optional: Set Limits Based on Prediction Intervals
plt.xlim(data['Measured'].min() - 1, data['Measured'].max() + 1)
plt.ylim(data['pi_lower'].min() - 10, data['pi_upper'].max() + 10)

# Remove Top and Right Spines for a Cleaner Look
sns.despine()

# Show Plot
plt.tight_layout()
plt.show()
