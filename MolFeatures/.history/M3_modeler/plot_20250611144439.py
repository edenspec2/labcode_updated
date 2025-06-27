import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from tkinter import filedialog, messagebox
from tkinter.simpledialog import askstring
import os
import statsmodels.api as sm
import tkinter as tk
import traceback
from tkinter import ttk
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
try:
    from .modeling import fit_and_evaluate_single_combination_regression
except ImportError as e:
    from modeling import fit_and_evaluate_single_combination_regression

def show_table_window(title, df):
    """
    Creates a new Tkinter window to display the DataFrame in a Treeview widget.

    Parameters:
        title (str): The title of the window.
        df (pd.DataFrame): The DataFrame to display.
    """
    # Create a new top-level window
    window = tk.Toplevel()
    window.title(title)

    # Set window size (optional)
    window.geometry("800x600")

    # Create a frame for the Treeview and scrollbar
    frame = ttk.Frame(window)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create the Treeview
    tree = ttk.Treeview(frame, columns=list(df.columns), show='headings')

    # Define headings and column properties
    for col in df.columns:
        tree.heading(col, text=col)
        # Optionally, set column width and alignment
        tree.column(col, anchor=tk.CENTER, width=100)

    # Insert data into the Treeview
    for _, row in df.iterrows():
        tree.insert('', tk.END, values=list(row))

    # Add a vertical scrollbar
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Optional: Add a horizontal scrollbar
    scrollbar_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(xscroll=scrollbar_x.set)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)


def get_valid_integer(prompt, default_value):
            while True:
                value_str = askstring("Input", prompt)
                try:
                    return int(value_str)
                except (ValueError, TypeError):
                    # If the input is not valid, return the default value
                    print(f"Using default value: {default_value}")
                    return default_value
                

def fit_and_evaluate_single_combination_classification(model, combination, accuracy_threshold=0.7, return_probabilities=False):
        selected_features = model.features_df[list(combination)]
        X = selected_features.to_numpy()
        y = model.target_vector.to_numpy()

        # Fit the model
        model.fit(X, y)

        # Evaluate the model
        evaluation_results = model.evaluate(X, y)
      
        # Check if accuracy is above the threshold
        if evaluation_results['accuracy'] > accuracy_threshold:
            avg_accuracy, avg_f1,avg_r2 = model.cross_validation(X, y) ## , avg_auc
            evaluation_results['avg_accuracy'] = avg_accuracy
            evaluation_results['avg_f1_score'] = avg_f1
            evaluation_results['avg_r2'] = avg_r2
            # evaluation_results['avg_auc'] = avg_auc

        results={
            'combination': combination,
            'scores': evaluation_results,
            'models': model
        }

        if return_probabilities:

            probabilities = model.model.predict_proba(X)
            # Creating a DataFrame for probabilities
            prob_df = pd.DataFrame(probabilities, columns=[f'Prob_Class_{i+1}' for i in range(probabilities.shape[1])])
            prob_df['Predicted_Class'] = model.model.predict(X)
            prob_df['True_Class'] = y

            return results, prob_df

        return results


def set_q2_plot_settings(ax, lower_bound, upper_bound, fontsize=15):
    bounds_array = np.array([lower_bound, upper_bound])
    ax.plot(bounds_array, bounds_array, 'k--', linewidth=2)  # black dashed line
    ax.set_xlabel('Measured', fontsize=fontsize)  # Assuming 'Measured' is the label you want
    ax.set_ylabel('Predicted', fontsize=fontsize)
    ax.set_ylim(bounds_array)
    ax.set_xlim(bounds_array)
    ax.grid(True)  # Adding a grid

def build_regression_equation(formula, coefficients, r_squared):
    """
    Build a regression equation string with proper LaTeX formatting.
    """
    intercept = coefficients[0]  # Intercept
    feature_coeffs = coefficients[1:]

    # Escape underscores in feature names
    safe_formula = [str(name).replace("_", r"\_") for name in formula]

    equation_terms = []

    for i, coef in enumerate(feature_coeffs):
        sign = "+" if coef >= 0 else "-"
        equation_terms.append(f" {sign} {abs(coef):.2f}·{safe_formula[i]}")

    # Add intercept at the end
    sign_intercept = "+" if intercept >= 0 else "-"
    equation_terms.append(f" {sign_intercept} {abs(intercept):.2f}")

    # Build the LaTeX equation
    equation = f'$y = {"".join(equation_terms).strip()}$\n$R^2 = {r_squared:.2f}$'

    return equation

## might change in the future to plot confidence intervals as dotted lines calculated from the covariance matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
 

def generate_q2_scatter_plot(
    y,
    y_pred,
    labels,
    folds_df,
    formula,
    coefficients,
    r=None,
    lower_bound=None,
    upper_bound=None,
    figsize=(16, 6),
    fontsize=12,
    scatter_color='#2ca02c',
    band_color='cadetblue',
    identity_color='#1f77b4',
    palette='deep',
    dpi=300
):
    """
    Plots Predicted vs Measured with a smooth 90% confidence/prediction band
    around the regression line using seaborn's regplot, plus identity line,
    fixed margin lines, point labels, and Q² metrics.
    """
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    labels = np.asarray(labels)

    data = pd.DataFrame({
        'Measured':  y,
        'Predicted': y_pred,
        'Labels':    labels
    })

    sns.set_theme(style='whitegrid', palette=palette)
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter
    sns.scatterplot(
        data=data,
        x='Measured',
        y='Predicted',
        hue='Labels',
        palette=palette,
        edgecolor='w',
        s=100,
        ax=ax,
        legend=False
    )

    # Smooth regression band (90% CI) around regressed fit
    sns.regplot(
        data=data,
        x='Measured',
        y='Predicted',
        scatter=False,
        ci=90,
        line_kws={'color': band_color, 'linewidth': 0},
        ax=ax
    )

    # Identity line y = x and margin lines
    mn = min(data['Measured'].min(), data['Predicted'].min())
    mx = max(data['Measured'].max(), data['Predicted'].max())

    x_ideal = np.linspace(mn, mx, 100)
    dev_line_color = 'grey'

    # Plot margin lines: identity, ±0.25, ±0.5
    ax.plot(x_ideal, x_ideal, linestyle='--', color=identity_color, linewidth=2, label='Identity (y=x)')
    ax.plot(x_ideal, x_ideal + 0.25, linestyle='--', color=dev_line_color, linewidth=1, label='y=x+0.25')
    ax.plot(x_ideal, x_ideal - 0.25, linestyle='--', color=dev_line_color, linewidth=1, label='y=x-0.25')
    ax.plot(x_ideal, x_ideal + 0.50, linestyle=':', color=dev_line_color, linewidth=1, label='y=x+0.50')
    ax.plot(x_ideal, x_ideal - 0.50, linestyle=':', color=dev_line_color, linewidth=1, label='y=x-0.50')

    # Equation & Pearson r
    corr = r if r is not None else np.corrcoef(y, y_pred)[0,1]
    eqn = build_regression_equation(formula, coefficients, corr)
    ax.text(
        0.05, 0.95,
        f"{eqn}\nPearson r = {corr:.2f}",
        transform=ax.transAxes,
        fontsize=fontsize,
        va='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    # Q² metrics
    if not folds_df.empty:
        q = folds_df.iloc[0]
        q_txt = (
            f"3-fold Q²: {q['Q2_3_Fold']:.2f}\n"
            f"5-fold Q²: {q['Q2_5_Fold']:.2f}\n"
            f"LOOCV Q²: {q['Q2_LOOCV']:.2f}"
        )
        ax.text(0.05, 0.80, q_txt, transform=ax.transAxes,
                fontsize=fontsize, va='top', bbox=dict(facecolor='white', alpha=0.8))

    # Labels
    texts = []
    for _, row in data.iterrows():
        texts.append(
            ax.text(row['Measured'], row['Predicted'], row['Labels'],
                    fontsize=fontsize-2, ha='center', va='bottom', color='gray')
        )
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax.set_xlabel('Measured', fontsize=fontsize+2)
    ax.set_ylabel('Predicted', fontsize=fontsize+2)
    ax.set_title('Predicted vs Measured with Smooth Regression Band', fontsize=fontsize+4)

    # Optional: show a legend for the margin lines
    handles, labels_ = ax.get_legend_handles_labels()
    margin_labels = ['Identity (y=x)', 'y=x+0.25', 'y=x-0.25', 'y=x+0.50', 'y=x-0.50']
    display_handles = [h for h, l in zip(handles, labels_) if l in margin_labels]
    display_labels = [l for l in labels_ if l in margin_labels]
    if display_handles:
        ax.legend(display_handles, display_labels, loc='lower right', frameon=True)

    plt.tight_layout()
    plt.savefig(f'model_plot_{formula}.png', dpi=dpi)
    # plt.show()

    return fig


### chance to use sns.regplot to plot the regression line and confidence intervals


# def generate_q2_scatter_plot(
#     y, 
#     y_pred, 
#     labels, 
#     folds_df, 
#     formula, 
#     coefficients, 
#     r, 
#     lower_bound=None, 
#     upper_bound=None, 
#     figsize=(10, 10), 
#     fontsize=12, 
#     scatter_color='#2ca02c',  # A more vibrant color
#     regression_color='#d62728',  # Distinct color for regression line
#     palette='deep',
#     dpi=300
# ):
#     """
#     Generates a beautiful and state-of-the-art scatter plot with regression analysis.

#     Parameters:
#         y (array-like): Measured values.
#         y_pred (array-like): Predicted values.
#         labels (array-like): Labels for each data point.
#         folds_df (pd.DataFrame): DataFrame containing Q2 metrics.
#         formula (str): Regression formula.
#         coefficients (dict): Regression coefficients.
#         r (float): Pearson correlation coefficient.
#         lower_bound (array-like, optional): [x_min, y_min].
#         upper_bound (array-like, optional): [x_max, y_max].
#         figsize (tuple, optional): Figure size in inches.
#         fontsize (int, optional): Base font size.
#         scatter_color (str, optional): Color of scatter points.
#         regression_color (str, optional): Color of regression line.
#         palette (str, optional): Seaborn color palette.
#         dpi (int, optional): Resolution of saved figure.

#     Returns:
#         plt.Figure: The matplotlib figure object.
#     """
    
#     # Convert inputs to numpy arrays for easier handling
#     y = np.array(y)
#     y_pred = np.array(y_pred)
#     labels = np.array(labels)
    
#     # Validate y and y_pred
#     if np.isnan(y).any() or np.isnan(y_pred).any():
#         raise ValueError("Input data 'y' or 'y_pred' contains NaN values.")
#     if np.isinf(y).any() or np.isinf(y_pred).any():
#         raise ValueError("Input data 'y' or 'y_pred' contains Inf values.")
    
#     # Validate lower_bound and upper_bound if provided
#     if lower_bound is not None:
#         lower_bound = np.array(lower_bound)
#         if lower_bound.shape != (2,):
#             raise ValueError("lower_bound must be an array-like with two elements: [x_min, y_min].")
#         if not np.isfinite(lower_bound).all():
#             raise ValueError("lower_bound contains NaN or Inf values.")
    
#     if upper_bound is not None:
#         upper_bound = np.array(upper_bound)
#         if upper_bound.shape != (2,):
#             raise ValueError("upper_bound must be an array-like with two elements: [x_max, y_max].")
#         if not np.isfinite(upper_bound).all():
#             raise ValueError("upper_bound contains NaN or Inf values.")
    
#     # Create a DataFrame for seaborn usage
#     data = pd.DataFrame({
#         'Measured': y,
#         'Predicted': y_pred,
#         'Labels': labels
#     })
    
#     # Check if data is sufficient for plotting
#     if data.empty:
#         raise ValueError("No data available to plot.")
    
#     # Set Seaborn theme
#     sns.set_theme(style="whitegrid", palette=palette)
    
#     # Initialize the matplotlib figure
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Scatter plot
#     scatter = sns.scatterplot(
#         data=data, 
#         x='Measured', 
#         y='Predicted', 
#         hue='Labels',
#         palette=palette,
#         edgecolor='w',
#         s=100,  # Increased size for better visibility
#         ax=ax,
#         legend=False  # Disable legend to reduce clutter
#     )
    
#     # Regression line
#     sns.regplot(
#         data=data, 
#         x='Measured', 
#         y='Predicted', 
#         scatter=False,
#         color=regression_color,
#         line_kws={'linewidth': 2, 'alpha': 0.8},
#         ax=ax
#     )
    
#     # Compute and display Pearson correlation if not provided
#     if r is None:
#         pearson_r = np.corrcoef(y, y_pred)[0,1]
#     else:
#         pearson_r = r
    
#     equation = build_regression_equation(formula, coefficients, pearson_r)
    
#     # Add regression equation and Pearson r
#     text_box = (
#         f"{equation}\n"
#         f"Pearson r = {pearson_r:.2f}"
#     )
#     ax.text(
#         0.05, 0.95, text_box, 
#         transform=ax.transAxes,
#         fontsize=fontsize,
#         verticalalignment='top',
#         bbox=dict(boxstyle="round,pad=0.5", 
#                   edgecolor='gray', 
#                   facecolor='white', 
#                   alpha=0.8)
#     )
    
#     # Set axis limits if bounds are provided
#     if lower_bound is not None and upper_bound is not None:
#         ax.set_xlim(lower_bound[0], upper_bound[0])
#         ax.set_ylim(lower_bound[1], upper_bound[1])
#     else:
#         # Set limits based on data with some padding
#         buffer_x = (data['Measured'].max() - data['Measured'].min()) * 0.05
#         buffer_y = (data['Predicted'].max() - data['Predicted'].min()) * 0.05
#         ax.set_xlim(data['Measured'].min() - buffer_x, data['Measured'].max() + buffer_x)
#         ax.set_ylim(data['Predicted'].min() - buffer_y, data['Predicted'].max() + buffer_y)
    
#     # Annotations using adjustText to prevent overlap
#     texts = []
#     for _, row in data.iterrows():
#         texts.append(ax.text(
#             row['Measured'], 
#             row['Predicted'], 
#             row['Labels'],
#             fontsize=fontsize-2,
#             ha='center', 
#             va='bottom',
#             color='gray'
#         ))
#     adjust_text(texts, 
#                 arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
#                 ax=ax,
#                 expand_points=(1.2, 1.2),
#                 force_points=0.2,
#                 force_text=0.2)
    
#     # Plot the Q2 metrics if available
#     if not folds_df.empty:
#         # Assuming folds_df has only one relevant row
#         row = folds_df.iloc[0]
#         q2_text = (
#             f"3-fold Q²: {row.get('Q2_3_Fold', np.nan):.2f}\n"
#             f"5-fold Q²: {row.get('Q2_5_Fold', np.nan):.2f}\n"
#             f"LOOCV Q²: {row.get('Q2_LOOCV', np.nan):.2f}"
#         )
#         ax.text(
#             0.05, 0.80, q2_text, 
#             transform=ax.transAxes,
#             fontsize=fontsize,
#             verticalalignment='top',
#             bbox=dict(boxstyle="round,pad=0.5", 
#                       edgecolor='gray', 
#                       facecolor='white', 
#                       alpha=0.8)
#         )
    
#     # Customize labels and title
#     ax.set_xlabel("Measured", fontsize=fontsize+2, fontweight='bold')
#     ax.set_ylabel("Predicted", fontsize=fontsize+2, fontweight='bold')
#     ax.set_title('Regression Analysis with Labels', fontsize=fontsize+4, fontweight='bold', pad=15)
    
#     # Improve layout
#     plt.tight_layout()
    
#     # Save the plot with high resolution
#     plt.savefig(f'model_plot_{formula}.png', dpi=dpi, bbox_inches='tight')
    
#     # Show the plot
#     plt.show()
    
#     return fig



import matplotlib.pyplot as plt

def plot_probabilities(probabilities_df, sample_names):
    df = probabilities_df.copy()
    
    # Rename if needed
    if 'prediction' in df.columns:
        df.rename(columns={'prediction': 'Predicted_Class'}, inplace=True)
    if 'True_Class' in df.columns:
        df.rename(columns={'True_Class': 'Actual_Class'}, inplace=True)
    
    # Identify probability columns by prefix
    prob_cols = [col for col in df.columns if col.startswith('Prob_Class_')]
    if not prob_cols:
        raise ValueError("No probability columns found (expecting columns like 'Prob_Class_1', ...)")
    
    # Ensure classes are integers for building column names
    df['Actual_Class'] = df['Actual_Class'].astype(int)
    df['Predicted_Class'] = df['Predicted_Class'].astype(str)
    
    # Compute rankings across the probability columns
    rankings = df[prob_cols].rank(axis=1, ascending=False, method='min')
    
    # Helper to get the rank of the true class
    def _true_rank(row):
        prob_col = f"Prob_Class_{row['Actual_Class']}"
        if prob_col not in rankings.columns:
            raise KeyError(f"Expected column {prob_col} in probabilities, got {prob_cols}")
        return int(rankings.at[row.name, prob_col])
    
    df['Rank'] = df.apply(_true_rank, axis=1)
    
    # Color‐code by rank (1=green, 2=yellow, 3=red, >3=gray)
    color_map = {1: 'green', 2: 'yellow', 3: 'red'}
    df['Color_Code'] = df['Rank'].map(color_map).fillna('gray')
    
    # Build sample labels from your list
    if len(sample_names) != len(df):
        raise ValueError("sample_names length must match number of rows in probabilities_df")
    df['Label'] = [
        f"{name} (Pred: {pred}, Actual: {act})"
        for name, pred, act in zip(
            sample_names,
            df['Predicted_Class'].astype(str),
            df['Actual_Class'].astype(str)
        )
    ]
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df[prob_cols].astype(float),
        cmap='Blues',
        annot=True,
        fmt=".2f",
        cbar_kws={'label': 'Probability'}
    )

    ax = plt.gca()
    # Y‐ticks: use your sample labels
    ax.set_yticks(np.arange(0.5, len(df), 1))
    ax.set_yticklabels(df['Label'], rotation=0, fontsize=10)
    for label, color in zip(ax.get_yticklabels(), df['Color_Code']):
        label.set_color(color)
    
    # X‐ticks: strip the prefix for readability
    classes = [col.replace('Prob_Class_', '') for col in prob_cols]
    ax.set_xticklabels(classes, rotation=45, ha='right')
    
    plt.title('Probability Heatmap with Prediction vs. Actual')
    plt.xlabel('Class')
    plt.ylabel('Samples')
    plt.tight_layout()
    plt.show()

from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

def plot_enhanced_confusion_matrix(cm, classes, precision, recall, accuracy, figsize=(10, 10), annot_fontsize=12):
    """
    Plot an enhanced confusion matrix with precision, recall, and accuracy.
    
    The confusion matrix will include an extra row and column for total samples and precision.
    
    Parameters:
    - cm: Confusion matrix (as a 2D array)
    - classes: List of class names
    - precision: List of precision values for each class
    - recall: List of recall values for each class
    - accuracy: Overall accuracy value
    - figsize: Size of the figure
    - annot_fontsize: Font size of the annotations inside the matrix
    """
    # Number of classes
    n = len(classes)

    # Expand the confusion matrix to 4x4
    cm_expanded = np.zeros((n + 1, n + 1))

    # Fill the confusion matrix values into the expanded matrix
    cm_expanded[:n, :n] = cm

    # Calculate row totals (true class totals) and column totals (predicted class totals)
    row_totals = cm.sum(axis=1)  # Row totals for the right column
    col_totals = cm.sum(axis=0)  # Column totals for the bottom row

    # Fill the last row with column totals and precision percentages
    cm_expanded[n, :n] = col_totals

    # Fill the last column with row totals and percentages of the total dataset
    cm_expanded[:n, n] = row_totals

    # Add overall total and accuracy in the bottom-right corner
    cm_expanded[n, n] = cm.sum()

    # Create a DataFrame for plotting
    classes_with_total = classes + ['Total']
    cm_df = pd.DataFrame(cm_expanded, index=classes_with_total, columns=classes_with_total)

    # Create annotations for the confusion matrix cells
    annot = np.empty_like(cm_expanded).astype(str)

    # Fill in the confusion matrix cells with counts and percentages (diagonal and off-diagonal)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{int(cm_expanded[i, j])}\n({cm_expanded[i, j] / row_totals[i] * 100:.1f}%)" if row_totals[i] != 0 else f"{int(cm_expanded[i, j])}\n(0%)"

    # Fill the last row with precision percentages
    for i in range(n):
        annot[n, i] = f"{int(col_totals[i])}\n({precision[i] * 100:.1f}%)"

    # Fill the last column with row totals and percentage of total dataset
    for i in range(n):
        annot[i, n] = f"{int(row_totals[i])}\n({row_totals[i] / cm.sum() * 100:.1f}%)"

    # Add the total and accuracy in the bottom-right corner
    annot[n, n] = f"Total: {int(cm.sum())}\nAcc: {accuracy * 100:.1f}%"

    # Plot the expanded confusion matrix
    plt.figure(figsize=figsize)

    # Create a mask for diagonal cells (true positives)
    mask_diag = np.zeros_like(cm_expanded, dtype=bool)
    np.fill_diagonal(mask_diag, True)

    # Create a mask for the bottom-right cell (total and accuracy)
    mask_acc = np.zeros_like(cm_expanded, dtype=bool)
    mask_acc[-1, -1] = True

    # Mask for off-diagonal cells
    mask_false = np.ones_like(cm_expanded, dtype=bool)
    np.fill_diagonal(mask_false, False)
    mask_false[-1, :] = False  # Do not mask the totals row
    mask_false[:, -1] = False  # Do not mask the totals column

    # Mask for total row (yellow gradient)
    mask_total_row = np.zeros_like(cm_expanded, dtype=bool)
    mask_total_row[-1, :-1] = True  # The entire last row except the bottom-right corner

    # Mask for total column (grey gradient)
    mask_total_column = np.zeros_like(cm_expanded, dtype=bool)
    mask_total_column[:-1, -1] = True  # The entire last column except the bottom-right corner

    # Plot true positive cells (diagonal) using green gradient (Greens cmap)
    sns.heatmap(cm_df, annot=annot, annot_kws={"size": annot_fontsize}, fmt='', mask=~mask_diag, cmap="Greens", vmin=0, vmax=np.max(cm), cbar=False, square=True, linewidths=1, linecolor='black')

    # Plot false predictions (off-diagonal) using red gradient (Reds cmap)
    sns.heatmap(cm_df, annot=annot, annot_kws={"size": annot_fontsize}, fmt='', mask=~mask_false, cmap="Reds", vmin=0, vmax=np.max(cm), cbar=False, square=True, linewidths=1, linecolor='black')

    # Plot total row (yellow gradient)
    sns.heatmap(cm_df, annot=annot, annot_kws={"size": annot_fontsize}, fmt='', mask=~mask_total_row, cmap="YlOrBr", vmin=0, vmax=np.max(cm), cbar=False, square=True, linewidths=1, linecolor='black')

    # Plot total column (grey gradient)
    sns.heatmap(cm_df, annot=annot, annot_kws={"size": annot_fontsize}, fmt='', mask=~mask_total_column, cmap="Greys", vmin=0, vmax=np.max(cm), cbar=False, square=True, linewidths=1, linecolor='black')

    # Plot total samples and precision/accuracy using blue gradient (Blues cmap)
    sns.heatmap(cm_df, annot=annot, annot_kws={"size": annot_fontsize}, fmt='', mask=~mask_acc, cmap="Blues", vmin=0, vmax=np.max(cm), cbar=False, square=True, linewidths=1, linecolor='black')

    # Add a color legend for each section
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', edgecolor='black', label='True Positives (Diagonal)'),
                       Patch(facecolor='red', edgecolor='black', label='False Positives/Negatives'),
                       Patch(facecolor='yellow', edgecolor='black', label='Precision'),
                       Patch(facecolor='grey', edgecolor='black', label='Total Column (Grey)'),
                       Patch(facecolor='blue', edgecolor='black', label='Overall Total and Accuracy')]
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1), title='Legend')

    plt.title('Enhanced Confusion Matrix with Totals, Precision, and Accuracy', fontsize=16)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.show()





def print_models_classification_table(results , app=None):
    formulas=[result['combination'] for result in results]
    accuracy=[result['scores']['accuracy'] for result in results]
    precision=[result['scores']['precision'] for result in results]
    recall=[result['scores']['recall'] for result in results]
    f1=[result['scores']['f1_score'] for result in results]
    mcfaden=[result['scores']['mc_fadden_r2'] for result in results]
    model_ids=[i for i in range(len(results))]
    models=[result['models'] for result in results]
    avg_accuracy=[result['scores'].get('avg_accuracy', float('-inf')) for result in results]
    avg_f1=[result['scores'].get('avg_f1_score', float('-inf')) for result in results]
    # avg_auc=[result['scores'].get('avg_auc', float('-inf')) for result in results]
    # Create a DataFrame from the inputs
    df = pd.DataFrame({
        'formula': formulas,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcfaden': mcfaden,
        'avg_accuracy': avg_accuracy,
        'avg_f1': avg_f1,
        #'avg_auc': avg_auc,

        'Model_id': model_ids
    })
    df.sort_values(by='avg_accuracy', ascending=False, inplace=True)
    # Set the index to range from 1 to n (1-based indexing)
    df.index = range(1, len(df) + 1)
    if app:
        app.show_result(df.to_markdown(index=False, tablefmt="pipe"))
        messagebox.showinfo('Models List: ', df.to_markdown(index=False, tablefmt="pipe"))  
        
    else:
        print(df.to_markdown(index=False, tablefmt="pipe"))
        

    try:
        df.to_csv('models_classification_table.csv', index=False)
    except:
        print('could not save the table')

    
    while True:
        if app:
            selected_model = get_valid_integer('Select a model number: default is 0', 0)
        else:
            try:
                selected_model = int(input("Select a model number (or -1 to exit): "))
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue
            
        if selected_model == -1:
            print("Exiting model selection.")
            break

        try:
            model = models[selected_model]
        except IndexError:
            print("Invalid model number. Please try again.")
            continue

        _, probablities_df = fit_and_evaluate_single_combination_classification(model, formulas[selected_model], return_probabilities=True)
        X=model.features_df[list(formulas[selected_model])]
        # x=pd.DataFrame(X, columns=formulas[selected_model])
        vif_df = model._compute_vif(X)
        samples_names=model.molecule_names
        plot_probabilities(probablities_df, samples_names)
        print_models_vif_table(vif_df)
        # Print the confusion matrix
        y_pred = model.predict(model.features_df[list(formulas[selected_model])].to_numpy())
        y_true = model.target_vector.to_numpy()
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix\n")
        class_names = np.unique(y_true)
        class_names = [f'Class_{i}' for i in class_names]

        model_precision = precision_score(y_true, y_pred, average=None)
        model_recall = recall_score(y_true, y_pred, average=None)
        model_accuracy = accuracy_score(y_true, y_pred)

        # Save text file with the results
        with open('classification_results.txt', 'a') as f:
            f.write(f"Models List\n\n{df.to_markdown(index=False, tablefmt='pipe')}\n\n Probabilities\n\n{probablities_df.to_markdown(tablefmt='pipe')}\n\nConfusion Matrix\n\n{cm}\n\nPrecision\n\n{model_precision}\n\nRecall\n\n{model_recall}\n\nAccuracy\n\n{model_accuracy}\n\n")
            print('Results saved to classification_results.txt in {}'.format(os.getcwd()))

        plot_enhanced_confusion_matrix(cm, class_names, model_precision, model_recall, model_accuracy)

        # Ask the user if they want to select another model or exit
        if not app:
            cont = input("Do you want to select another model? (y/n): ").strip().lower()
            if cont != 'y':
                print("Exiting model selection.")
                break
        else:
            cont=messagebox.askyesno('Continue','Do you want to select another model?')
            if not cont:
                break



def print_models_vif_table(results, app=None):
    if app:
        app.show_result('\n\n\n')
        app.show_result('VIF Table\n')
        app.show_result('---\n')
        app.show_result(results.to_markdown(index=False, tablefmt="pipe"))
    else:
        print('\n\n\n')
        print('VIF Table\n')
        print('---\n')
        print(results.to_markdown(index=False, tablefmt="pipe"))


def print_models_regression_table(results, app=None):

    formulas=[result['combination'] for result in results]
    r_squared=[result['scores']['r2'] for result in results]
    q_squared=[result['scores'].get('Q2', float('-inf')) for result in results]
    mae=[result['scores'].get('MAE', float('-inf')) for result in results]
    model_ids=[i for i in range(len(results))]
    intercepts=[result['intercept'] for result in results]
    model_coefficients=[result['coefficients'] for result in results]
   
    models=[result['models'] for result in results]

    # Create a DataFrame from the inputs
    df = pd.DataFrame({
        'formula': formulas,
        'R.sq': r_squared,
        'Q.sq': q_squared,
        'MAE': mae,
        'Model_id': model_ids
    })

    # Sort the DataFrame by Q.sq (descending) for a similar order
    df = df.sort_values(by='Q.sq', ascending=False)
    
    # Set the index to range from 1 to n (1-based indexing)
    df.index = range(1, len(df) + 1)
    
    # Print the DataFrame as a markdown-like table
    
    # Q2_3, MAE_3 = model.calculate_q2_and_mae(X, y, n_splits=3)
    # Q2_5, MAE_5 = model.calculate_q2_and_mae(X, y, n_splits=5)
    


    while True:

        if app:
            messagebox.showinfo('Models List:',df.to_markdown(index=False, tablefmt="pipe"))
            print(df.to_markdown(index=False, tablefmt="pipe"))
            selected_model = get_valid_integer('Select a model number: default is 0', 0)
            show_table_window('Models List:',df)
        else:
            print(df.to_markdown(index=False, tablefmt="pipe"))
            try:
                selected_model = int(input("Select a model number (or -1 to exit): "))
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue
            
        if selected_model == -1:
            print("Exiting model selection.")
            break

        try:
            model = models[selected_model]
        except IndexError:
            print("Invalid model number. Please try again.")
            continue

        features = list(formulas[selected_model])
        X = model.features_df[features]
        vif_df = model._compute_vif(X)
       
        X = model.features_df[features].to_numpy()
        y = model.target_vector.to_numpy()
        model.fit(X, y)
        pred, lwr, upr = model.predict(X, return_interval=True)
        coef_df = model.get_covariace_matrix(features)

        x_min, y_min = y.min(), y.min()
        x_max, y_max = pred.max(), pred.max()
        padding_x = (x_max - x_min) * 0.05
        padding_y = (y_max - y_min) * 0.05
        lwr = [x_min - padding_x, y_min - padding_y]
        upr = [x_max + padding_x, y_max + padding_y]

        # Debug statements
        print(f"lower_bound (lwr): {lwr}")
        print(f"upper_bound (upr): {upr}")

        # Validate bounds
        if not (isinstance(lwr, (list, tuple, np.ndarray)) and len(lwr) == 2 and np.all(np.isfinite(lwr))):
            print("Invalid lower_bound. Using default.")
            lwr = None
        if not (isinstance(upr, (list, tuple, np.ndarray)) and len(upr) == 2 and np.all(np.isfinite(upr))):
            print("Invalid upper_bound. Using default.")
            upr = None
        
        if app:
            app.show_result('\nModel Coefficients\n')
            app.show_result(coef_df.to_markdown(tablefmt="pipe"))
            print_models_vif_table(vif_df, app)
        else:
            print("\nModel Coefficients\n")
            print(coef_df.to_markdown(tablefmt="pipe"))
            print(f"\nSelected Model: {formulas[selected_model]}\n")
            print_models_vif_table(vif_df)
        
        Q2_3, MAE_3 , rmsd_3 = model.calculate_q2_and_mae(X, y, n_splits=3)
        Q2_5, MAE_5, rmsd_5 = model.calculate_q2_and_mae(X, y, n_splits=5)
        ## LOOCV
        Q2_loo, MAE_loo , rmsd_loo = model.calculate_q2_and_mae(X, y, n_splits=1)
        
        if app:
            app.show_result(f'\n\n Model Picked: {selected_model}_{formulas[selected_model]}\n')
            app.show_result(pd.DataFrame({'Q2_3_Fold': [Q2_3], 'MAE': [MAE_3], 'RMSD':[rmsd_3]}).to_markdown(tablefmt="pipe", index=False))
            app.show_result(pd.DataFrame({'Q2_5_Fold':[Q2_5], 'MAE': [MAE_5], 'RMSD':[rmsd_5]}).to_markdown(tablefmt="pipe", index=False))
            app.show_result(pd.DataFrame({'Q2_LOOCV':[Q2_loo], 'MAE': [MAE_loo], 'RMSD':[rmsd_loo]}).to_markdown(tablefmt="pipe", index=False))
        else:
            print("\n3-fold CV\n")
            print(pd.DataFrame({'Q2_3_Fold': [Q2_3], 'MAE': [MAE_3]}).to_markdown(tablefmt="pipe", index=False))
            print("\n5-fold CV\n")
            print(pd.DataFrame({'Q2_5_Fold':[Q2_5], 'MAE': [MAE_5]}).to_markdown(tablefmt="pipe", index=False))
            print("\nLOOCV\n")
            print(pd.DataFrame({'Q2_LOOCV':[Q2_loo], 'MAE': [MAE_loo]}).to_markdown(tablefmt="pipe", index=False))
        
        # Create a text file with the results
        with open('regression_results.txt', 'a') as f:
            f.write(f"Models list {df.to_markdown(index=False, tablefmt='pipe')} \n\n Model Coefficients\n\n{coef_df.to_markdown(tablefmt='pipe')}\n\n3-fold CV\n\n{pd.DataFrame({'Q2': [Q2_3], 'MAE': [MAE_3]}).to_markdown(tablefmt='pipe', index=False)}\n\n5-fold CV\n\n{pd.DataFrame({'Q2':[Q2_5], 'MAE': [MAE_5]}).to_markdown(tablefmt='pipe', index=False)}\n\n")
            print('Results saved to regression_results.txt in {}'.format(os.getcwd()))
        ## make a 3 5 loocv table to plot
        folds_df=pd.DataFrame({'Q2_3_Fold': [Q2_3], 'MAE': [MAE_3],'RMSD':[rmsd_3],'Q2_5_Fold':[Q2_5], 'MAE': [MAE_5],'RMSD':[rmsd_5],'Q2_LOOCV':[Q2_loo], 'MAE': [MAE_loo],'RMSD':[rmsd_loo]})
        r=r_squared[selected_model]
        # Generate and display the Q2 scatter plot
        
        _ = generate_q2_scatter_plot(y, pred, model.molecule_names,folds_df ,features,coef_df['Estimate'] ,r,X, lwr, upr)

        # Ask the user if they want to select another model or exit
        if not app:
            cont = input("Do you want to select another model? (y/n): ").strip().lower()
            if cont != 'y':
                print("Exiting model selection.")
                break
        else:
            cont=messagebox.askyesno('Continue','Do you want to select another model?')
            if not cont:
                break



def generate_and_display_q2_scatter_plot(model, features, app=None):
    """
    Computes extra calculations (fitting, predictions, CV metrics, coefficient estimates, 
    and axis bounds) and then generates the Q2 scatter plot.

    Parameters:
        model: The regression model instance.
        features (list): List of feature names (columns in model.features_df) to use.
        app (optional): An application interface to display results (if provided).

    Returns:
        The return value of generate_q2_scatter_plot.
    """
    print("Starting generate_and_display_q2_scatter_plot...")
    
    # Extract features and target values
    try:
        
        print("Extracting features from model.features_df...")
        X = model.features_df[features].to_numpy()
        y = model.target_vector.to_numpy()
       
    except Exception as e:
        print("Error extracting features:", e)
        return
    
   
    try:

        result = fit_and_evaluate_single_combination_regression(model, features)
        pred = result['predictions']
        # print y and pred side by side
        # prediction_print = pd.DataFrame({'y': y, 'pred': pred})
        print('Training Set Results:', result['scores'])
        # print("Predictions generated successfully:\n", prediction_print)
        # print("Model fitted and predictions generated successfully.")
    except Exception as e:
        print("Error during model fitting/prediction:", e)
        return

    # Retrieve coefficient estimates
    

    # Compute cross-validation metrics
    try:
        print("Calculating cross-validation metrics for 3-fold CV...")
        Q2_3, MAE_3, rmsd_3 = model.calculate_q2_and_mae(X, y, n_splits=3)
        print("3-fold CV metrics: Q2: {}, MAE: {}, RMSD: {}".format(Q2_3, MAE_3, rmsd_3))
        
        print("Calculating cross-validation metrics for 5-fold CV...")
        Q2_5, MAE_5, rmsd_5 = model.calculate_q2_and_mae(X, y, n_splits=5)
        print("5-fold CV metrics: Q2: {}, MAE: {}, RMSD: {}".format(Q2_5, MAE_5, rmsd_5))
        
        print("Calculating cross-validation metrics for LOOCV...")
        Q2_loo, MAE_loo, rmsd_loo = model.calculate_q2_and_mae(X, y, n_splits=1)
        print("LOOCV metrics: Q2: {}, MAE: {}, RMSD: {}".format(Q2_loo, MAE_loo, rmsd_loo))


        leftout_pred = None
        # print models attributes and params
        
        try:
          
            if model.leftout_samples is not None and len(model.leftout_samples) > 0:
                print("Calculating left-out samples prediction and metrics...")

                X_left = model.leftout_samples[features]  # DataFrame shape (n_leftout, 4)
                X_left = X_left.reindex()
                y_left = model.leftout_target_vector  # Series shape (n_leftout,)
               

                # 3) call your predictor; let it add constant & reorder itself
                try:
                    leftout_pred = model.predict_for_leftout(X_left, y=y_left, calc_interval=False)
                    print("Left-out samples prediction completed.", leftout_pred)
                    if isinstance(leftout_pred, tuple):
                        y_pred = np.array(leftout_pred[0]).ravel()  # Use only predictions, not errors
                    else:
                        y_pred = np.array(leftout_pred).ravel()
                        
                    print(f"Successfully predicted left-out samples: {leftout_pred}")
                except Exception as e:
                    print("Error predicting left-out samples:", e)
                    leftout_pred = None  # Ensure variable exists if exception occurs

                if leftout_pred is not None:
                    y_true = np.array(model.leftout_target_vector).ravel()
                    names  = list(model.molecule_names_predict)

                    # 2) build DataFrame
                    prediction_df = pd.DataFrame({
                        'Molecule':   names,
                        'Actual':     y_true,
                        'Predicted':  y_pred
                    })

                    # 3) compute absolute percent error (always positive, avoids division by zero)
                    prediction_df['Error in %'] = np.where(
                        prediction_df['Actual'] != 0,
                        np.abs(prediction_df['Actual'] - prediction_df['Predicted']) / np.abs(prediction_df['Actual']) * 100,
                        np.nan
                    )

                    print(prediction_df)
                    # Calculate and print R2 as well
                    r2_leftout = r2_score(y_true, y_pred)
                    mae_leftout = mean_absolute_error(y_true, y_pred)
                    print(f"R² for left-out predictions: {r2_leftout:.4f}")
                    print(f"MAE for left-out predictions: {mae_leftout:.4f}")

                        
                else:
                    print("No left-out predictions available; skipping result table.")
        except Exception as e:
            print("Error:", e)
            print("No left-out samples available; skipping result table.")
            pass

        # Prepare a folds DataFrame with CV results
        folds_df = pd.DataFrame({
            'Q2_3_Fold': [Q2_3],
            'MAE_3': [MAE_3],
            'RMSD_3': [rmsd_3],
            'Q2_5_Fold': [Q2_5],
            'MAE_5': [MAE_5],
            'RMSD_5': [rmsd_5],
            'Q2_LOOCV': [Q2_loo],
            'MAE_LOOCV': [MAE_loo],
            'RMSD_LOOCV': [rmsd_loo]
        })
        print("Folds DataFrame prepared:")
        print(folds_df)
    except Exception as e:
        
        print(traceback.format_exc())
        print("Error calculating cross-validation metrics:", e)
        return

    # Compute axis bounds for plotting
    try:
        print("Calculating fixed margin lines and axis bounds…")
        # Set axis bounds
        plot_scale_start = -3.0
        plot_scale_end = 0.6
        x_ideal = np.linspace(plot_scale_start, plot_scale_end, 100)

        # Fixed margin lines (identity ±0.25, ±0.50)
        margin_lines = {
            "identity":     x_ideal,                           # y = x
            "+0.25":        x_ideal + 0.25,                    # y = x + 0.25
            "-0.25":        x_ideal - 0.25,                    # y = x - 0.25
            "+0.50":        x_ideal + 0.50,                    # y = x + 0.50
            "-0.50":        x_ideal - 0.50                     # y = x - 0.50
        }

    except Exception as e:
        print("Error calculating margin lines:", e)
        return

    try:
        vif_df = model._compute_vif(model.features_df[features])
        print_models_vif_table(vif_df)
    except Exception as e:
        print("Error calculating VIF:", e)
        return

    # Compute R^2 (or r-squared) as a measure of correlation
    try:
        print("Calculating R^2 value...")
        r = np.corrcoef(y, pred)[0, 1]**2
        mae = mean_absolute_error(y, pred)
        print(f"R^2 value: {r:.4f}, MAE: {mae:.4f}")
        print("R^2 calculated:", r)
    except Exception as e:
        print("Error calculating R^2:", e)
        return
    
    try:
        y_i,upr,lwr = model.predict(X, return_interval=True)
        print("Retrieving coefficient estimates...")
        coef_df = model.get_covariance_matrix(features)
        print("Coefficient estimates retrieved:")
        print(coef_df.head())
    except Exception as e:
        print("Error retrieving coefficient estimates:", e)
        return
    
    try:
        print("Calling generate_q2_scatter_plot with computed values...")
        # Remove pi_lower and pi_upper if you are not calculating prediction intervals
        plot_output = generate_q2_scatter_plot(
            y, pred, model.molecule_names, folds_df, features, coef_df['Estimate'], r
        )
        print("Plot generated successfully.")
    except Exception as e:
        print("Error in generate_q2_scatter_plot:", e)
        return

    print("Finished generate_and_display_q2_scatter_plot.")
    return 

