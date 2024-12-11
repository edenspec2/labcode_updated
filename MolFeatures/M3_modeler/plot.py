import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from tkinter import filedialog, messagebox
from tkinter.simpledialog import askstring
import os

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

## might change in the future to plot confidence intervals as dotted lines calculated from the covariance matrix
def generate_q2_scatter_plot(y, y_pred, labels, formula , lower_bound=None, upper_bound=None, figsize=(10, 10), fontsize=12, scatter_color='black'):
    # Create a DataFrame for seaborn usage
    data = pd.DataFrame({
        'Measured': y,
        'Predicted': y_pred,
        'Labels': labels
    })
    
    # Initialize the plot with seaborn's set style and context
    sns.set(style="whitegrid", context="notebook", rc={"figure.figsize": figsize})

    min_bound = np.min(lower_bound)
    max_bound = np.max(upper_bound)

    # Create the scatter plot with regression line using sns.lmplot
    plot = sns.lmplot(x='Measured', y='Predicted', data=data,
                      height=figsize[0]/2.54, aspect=figsize[0]/figsize[1], # height in inches and aspect ratio
                      scatter_kws={'s': 50, 'color': scatter_color},
                      line_kws={'color': 'black', 'lw': 2},
                      ci=95) # ci=None to not display the confidence interval

    # Adjusting the axes limits if bounds are provided
    
    
    # plot.ax.set_ylim(min_bound, max_bound)
    # plot.ax.set_xlim(np.min(y), np.max(y))

    # Adding annotations
    for i, row in data.iterrows():
        plot.ax.annotate(row['Labels'], (row['Measured'], row['Predicted']), 
                         textcoords="offset points", xytext=(5,5), ha='center', fontsize=fontsize)

    # Additional customization options directly with seaborn
    plot.set_axis_labels("Measured", "Predicted", fontsize=fontsize, weight='bold')
    plot.fig.suptitle('Regression Analysis with Labels', fontsize=fontsize+2, weight='bold')
    plt.show()
    ## save the plot
    plot.savefig(f'model_plot_{formula}.png')



    return plot

import matplotlib.pyplot as plt

def plot_probabilities(probabilities_df):
    df = probabilities_df.copy()
    
    # Ensure 'actual' and 'prediction' columns are present and rename them
    if 'prediction' in df.columns:
        df.rename(columns={'prediction': 'Predicted_Class'}, inplace=True)
    if 'True_Class' in df.columns:
        df.rename(columns={'True_Class': 'Actual_Class'}, inplace=True)
    
    # Get the probability columns (assuming they are class labels)
    prob_cols = [col for col in df.columns if col not in ['Predicted_Class', 'Actual_Class']]
    
    # Ensure class labels are strings for consistent access
    df['Actual_Class'] = df['Actual_Class'].astype(str)
    df['Predicted_Class'] = df['Predicted_Class'].astype(str)
    prob_cols = [str(col) for col in prob_cols]
    
    # Compute the rankings of the probabilities for each sample
    rankings = df[prob_cols].rank(axis=1, ascending=False, method='min')
    
    # Get the rank of the actual class's probability for each sample
    df['Rank'] = df.apply(lambda row: rankings.loc[row.name, row['Actual_Class']], axis=1)
    df['Rank'] = df['Rank'].astype(int)
    
    # Map the Rank to colors as per your requirement
    color_map = {1: 'green', 2: 'yellow', 3: 'red'}
    df['Color_Code'] = df['Rank'].map(color_map)
    
    # Create labels for each sample, indicating predicted and actual classes
    df['Labels'] = df.apply(lambda row: f"Sample_{row.name} (Pred: {row['Predicted_Class']}, Actual: {row['Actual_Class']})", axis=1)
    
    # Plot heatmap of probabilities
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(df[prob_cols].astype(float),
                          cmap='Blues', annot=True, fmt=".2f", cbar_kws={'label': 'Probability (%)'})
    
    # Set the y-axis labels to include sample info and color them based on correctness
    plt.yticks(ticks=np.arange(0.5, len(df.index), 1), labels=df['Labels'], rotation=0, fontsize=10)
    for ytick, color in zip(plt.gca().get_yticklabels(), df['Color_Code']):
        ytick.set_color(color)
    
    plt.title('Probability Heatmap with Prediction Classes')
    plt.xlabel('Classes')
    plt.ylabel('Samples with Predictions')
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
        plot_probabilities(probablities_df)

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
    
    
    
    if app:
        messagebox.showinfo("3-fold CV",pd.DataFrame({'Q2': [Q2_3], 'MAE': [MAE_3]}).to_markdown(tablefmt="pipe", index=False))
        messagebox.showinfo("5-fold CV",pd.DataFrame({'Q2':[Q2_5], 'MAE': [MAE_5]}).to_markdown(tablefmt="pipe", index=False))
        # messagebox.showinfo('Models List:',df.to_markdown(index=False, tablefmt="pipe"))
        
    else:
        print(df.to_markdown(index=False, tablefmt="pipe"))
        
    ## Need the actual model list to calculate CV3 CV5

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

        features = list(formulas[selected_model])
        X = model.features_df[features].to_numpy()
        y = model.target_vector.to_numpy()
        model.fit(X, y)
        pred, lwr, upr = model.predict(X, calc_covariance_matrix=True)
        coef_df = model.get_covariace_matrix(features)
        
        if app:
            app.show_result('\nModel Coefficients\n')
            app.show_result(coef_df.to_markdown(tablefmt="pipe"))
        else:
            print("\nModel Coefficients\n")
            print(coef_df.to_markdown(tablefmt="pipe"))
            print(f"\nSelected Model: {formulas[selected_model]}\n")
        
        Q2_3, MAE_3 = model.calculate_q2_and_mae(X, y, n_splits=3)
        Q2_5, MAE_5 = model.calculate_q2_and_mae(X, y, n_splits=5)
        
        if app:
            app.show_result(f'\n\n Model Picked: {selected_model}_{formulas[selected_model]}\n')
            app.show_result(pd.DataFrame({'Q2': [Q2_3], 'MAE': [MAE_3]}).to_markdown(tablefmt="pipe", index=False))
            app.show_result(pd.DataFrame({'Q2':[Q2_5], 'MAE': [MAE_5]}).to_markdown(tablefmt="pipe", index=False))
        else:
            print("\n3-fold CV\n")
            print(pd.DataFrame({'Q2': [Q2_3], 'MAE': [MAE_3]}).to_markdown(tablefmt="pipe", index=False))
            print("\n5-fold CV\n")
            print(pd.DataFrame({'Q2':[Q2_5], 'MAE': [MAE_5]}).to_markdown(tablefmt="pipe", index=False))
        
        # Create a text file with the results
        with open('regression_results.txt', 'a') as f:
            f.write(f"Models list {df.to_markdown(index=False, tablefmt='pipe')} \n\n Model Coefficients\n\n{coef_df.to_markdown(tablefmt='pipe')}\n\n3-fold CV\n\n{pd.DataFrame({'Q2': [Q2_3], 'MAE': [MAE_3]}).to_markdown(tablefmt='pipe', index=False)}\n\n5-fold CV\n\n{pd.DataFrame({'Q2':[Q2_5], 'MAE': [MAE_5]}).to_markdown(tablefmt='pipe', index=False)}\n\n")
            print('Results saved to regression_results.txt in {}'.format(os.getcwd()))

        # Generate and display the Q2 scatter plot
        _ = generate_q2_scatter_plot(y, pred, model.molecule_names, features, lwr, upr)

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




#### have not been tried

def print_models_regression_table(results, app=None, auto_select_first_model=True):

    formulas = [result['combination'] for result in results]
    r_squared = [result['scores']['r2'] for result in results]
    q_squared = [result['scores'].get('Q2', float('-inf')) for result in results]
    mae = [result['scores'].get('MAE', float('-inf')) for result in results]
    model_ids = [i for i in range(len(results))]
    intercepts = [result['intercept'] for result in results]
    model_coefficients = [result['coefficients'] for result in results]
    models = [result['models'] for result in results]

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
    if app:
        messagebox.showinfo("3-fold CV", pd.DataFrame({'Q2': [Q2_3], 'MAE': [MAE_3]}).to_markdown(tablefmt="pipe", index=False))
        messagebox.showinfo("5-fold CV", pd.DataFrame({'Q2': [Q2_5], 'MAE': [MAE_5]}).to_markdown(tablefmt="pipe", index=False))
    else:
        print(df.to_markdown(index=False, tablefmt="pipe"))

    # Automatically select the first model if `auto_select_first_model` is True or run in nohup
    selected_model = 0 if auto_select_first_model else None

    while True:
        if auto_select_first_model or app:
            selected_model = selected_model if selected_model is not None else get_valid_integer('Select a model number: default is 0', 0)
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

        features = list(formulas[selected_model])
        X = model.features_df[features].to_numpy()
        y = model.target_vector.to_numpy()
        model.fit(X, y)
        pred, lwr, upr = model.predict(X, calc_covariance_matrix=True)
        coef_df = model.get_covariace_matrix(features)

        if app:
            app.show_result('\nModel Coefficients\n')
            app.show_result(coef_df.to_markdown(tablefmt="pipe"))
        else:
            print("\nModel Coefficients\n")
            print(coef_df.to_markdown(tablefmt="pipe"))
            print(f"\nSelected Model: {formulas[selected_model]}\n")

        Q2_3, MAE_3 = model.calculate_q2_and_mae(X, y, n_splits=3)
        Q2_5, MAE_5 = model.calculate_q2_and_mae(X, y, n_splits=5)

        if app:
            app.show_result(f'\n\n Model Picked: {selected_model}_{formulas[selected_model]}\n')
            app.show_result(pd.DataFrame({'Q2': [Q2_3], 'MAE': [MAE_3]}).to_markdown(tablefmt="pipe", index=False))
            app.show_result(pd.DataFrame({'Q2': [Q2_5], 'MAE': [MAE_5]}).to_markdown(tablefmt="pipe", index=False))
        else:
            print("\n3-fold CV\n")
            print(pd.DataFrame({'Q2': [Q2_3], 'MAE': [MAE_3]}).to_markdown(tablefmt="pipe", index=False))
            print("\n5-fold CV\n")
            print(pd.DataFrame({'Q2': [Q2_5], 'MAE': [MAE_5]}).to_markdown(tablefmt="pipe", index=False))

        # Create a text file with the results
        with open('regression_results.txt', 'a') as f:
            f.write(f"Models list {df.to_markdown(index=False, tablefmt='pipe')} \n\n Model Coefficients\n\n{coef_df.to_markdown(tablefmt='pipe')}\n\n3-fold CV\n\n{pd.DataFrame({'Q2': [Q2_3], 'MAE': [MAE_3]}).to_markdown(tablefmt='pipe', index=False)}\n\n5-fold CV\n\n{pd.DataFrame({'Q2': [Q2_5], 'MAE': [MAE_5]}).to_markdown(tablefmt='pipe', index=False)}\n\n")
            print('Results saved to regression_results.txt in {}'.format(os.getcwd()))

        # If auto-select mode, break after processing the first model
        if auto_select_first_model:
            print("Processed the first model, exiting.")
            break

        # Generate and display the Q2 scatter plot
        _ = generate_q2_scatter_plot(y, pred, model.molecule_names, features, lwr, upr)

        # Ask the user if they want to select another model or exit
        if not app:
            cont = input("Do you want to select another model? (y/n): ").strip().lower()
            if cont != 'y':
                print("Exiting model selection.")
                break
        else:
            cont = messagebox.askyesno('Continue', 'Do you want to select another model?')
            if not cont:
                break
