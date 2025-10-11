import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load
import matplotlib.pyplot as plt

def load_data(data_path='./data/pcr_200.csv'):
    try:
        csv_path = os.path.join(data_path)
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: Data file not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None

def prepare_data(data, num_attribs):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('mm_scaler', MinMaxScaler())
    ])
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
    ])
    
    return full_pipeline.fit_transform(data), full_pipeline

def split_data(data, target_column, stratify_column=None, test_size=0.2, random_state=42):
    if stratify_column and stratify_column not in data.columns:
        print(f"Warning: Column '{stratify_column}' not found in the dataset. Proceeding without stratified sampling.")
        stratify_column = None

    if stratify_column:
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        data['stratify_column_binned'] = discretizer.fit_transform(data[[stratify_column]])
        stratify_column = 'stratify_column_binned'

    train_set, test_set = train_test_split(
        data, 
        test_size=test_size, 
        stratify=data[stratify_column] if stratify_column else None, 
        random_state=random_state
    )

    if stratify_column == 'stratify_column_binned':
        train_set = train_set.drop(columns=['stratify_column_binned'])
        test_set = test_set.drop(columns=['stratify_column_binned'])

    return train_set, test_set

def train_and_evaluate_model(X, y, model, param_grid, scoring, cv=5, refit=True, save_model_path=None):
    try:
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, return_train_score=True, refit=refit)
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        
        y_train_pred = best_model.predict(X)
        train_mse = mean_squared_error(y, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y, y_train_pred)
        train_r2 = r2_score(y, y_train_pred)
        
        print(f"训练集 - Root Mean Squared Error (RMSE): {train_rmse}")
        print(f"训练集 - Mean Absolute Error (MAE): {train_mae}")
        print(f"训练集 - R-squared (R2): {train_r2}")
        
        print(f"最佳参数组合: {grid_search.best_params_}")
        print(f"最好的评估器: {grid_search.best_estimator_}")
        
        if save_model_path:
            dump(grid_search.best_estimator_, save_model_path)
        
        return best_model, train_rmse, train_mae, train_r2
    except Exception as e:
        print(f"An error occurred during model training and evaluation: {e}")
        return None, None, None, None

def test_model(model, X_test, y_test, full_pipeline, model_name):
    try:
        X_test_prepared = full_pipeline.transform(X_test)
        final_predictions = model.predict(X_test_prepared)
        
        final_mse = mean_squared_error(y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)
        final_mae = mean_absolute_error(y_test, final_predictions)
        final_r2 = r2_score(y_test, final_predictions)
        final_re = (y_test - final_predictions) / y_test
        
        print(f"{model_name} - Root Mean Squared Error (RMSE): {final_rmse}")
        print(f"{model_name} - Mean Absolute Error (MAE): {final_mae}")
        print(f"{model_name} - R-squared (R2): {final_r2}")
        print(f"{model_name} - MAX Relative Errors (RE): {max(abs(final_re * 100)):.2f}%")
        
        results = pd.DataFrame({
            'Actual_data': y_test,
            'Predicted_data': final_predictions
        })
        results.to_csv(f'./save_model/{model_name}/{model_name}_results.csv', index=False)
        
        return final_predictions, final_re
    except Exception as e:
        print(f"An error occurred during model testing: {e}")
        return None, None

def plot_relative_errors(model_names, final_res):
    plt.figure(figsize=(14, 8))
    bar_width = 0.6
    
    for i, (model_name, final_re) in enumerate(zip(model_names, final_res)):
        ax = plt.subplot(2, 2, i+1)
        index = np.arange(len(final_re))
        ax.bar(index, final_re*100, width=bar_width, alpha=0.7, color=np.where(final_re < 0, 'r', 'b'))
        ax.set_title(f'{model_name} Relative Error')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Relative Error (%)')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylim(-45, 45)
    
    plt.tight_layout()
    plt.show()

# Add these functions to utils.py

def plot_predictions_vs_actual(y_true, y_pred, model_name):
    """Plot predicted vs actual values with a regression line."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.grid(True)
    
    # Calculate and display R-squared on the plot
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(f'./save_model/{model_name}/{model_name}_actual_vs_predicted.png')
    plt.close()

def plot_error_distribution(errors, model_name):
    """Plot the distribution of prediction errors."""
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'{model_name}: Prediction Error Distribution')
    plt.grid(True)
    plt.savefig(f'./save_model/{model_name}/{model_name}_error_distribution.png')
    plt.close()

def plot_residuals(y_true, y_pred, model_name):
    """Plot residuals vs predicted values."""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: Residual Plot')
    plt.grid(True)
    plt.savefig(f'./save_model/{model_name}/{model_name}_residual_plot.png')
    plt.close()