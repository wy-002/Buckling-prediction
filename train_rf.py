from sklearn.ensemble import RandomForestRegressor
from utils import *

def train_random_forest(data_prepared, data_labels, save_path='./save_model/RF/rf_best_model_01.joblib'):
    param_grid_rf = {
        "n_estimators": [60],
        "bootstrap": [True],
        "max_depth": [25],
        "random_state": [42]
    }
    
    print("--------------------------------------------------")
    print("RF:")
    rf_model, train_rmse_rf, train_mae_rf, train_r2_rf = train_and_evaluate_model(
        data_prepared, data_labels, 
        RandomForestRegressor(), 
        param_grid_rf, 
        'neg_mean_squared_error', 
        save_model_path=save_path
    )
    print("--------------------------------------------------")
    
    return rf_model

def visualize_results(model, X_test, y_test, full_pipeline, model_name):
    """Generate and save visualization plots for the model results."""
    X_test_prepared = full_pipeline.transform(X_test)
    y_pred = model.predict(X_test_prepared)
    
    # Create plots
    plot_predictions_vs_actual(y_test, y_pred, model_name)
    plot_error_distribution(y_test - y_pred, model_name)
    plot_residuals(y_test, y_pred, model_name)
    
    # Plot feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        features = full_pipeline.transformers_[0][2]  # Get feature names
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(f'{model_name}: Feature Importance')
        plt.bar(range(X_test.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_test.shape[1]), [features[i] for i in indices], rotation=90)
        plt.xlim([-1, X_test.shape[1]])
        plt.tight_layout()
        plt.savefig(f'./save_model/{model_name}/{model_name}_feature_importance.png')
        plt.close()

if __name__ == "__main__":
    data = load_data()
    if data is None:
        exit()
    
    target_column = "Pcr"
    stratify_column = 'Pcr'
    
    train_set, test_set = split_data(data, target_column, stratify_column)
    
    data_labels = train_set[target_column].copy()
    data_num = train_set.drop(target_column, axis=1)
    num_attribs = list(data_num.columns)
    
    data_prepared, full_pipeline = prepare_data(data_num, num_attribs)
    dump(full_pipeline, './save_model/RF/full_pipeline.joblib')
    
    rf_model = train_random_forest(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("RF Testing:")
    final_predictions_rf, final_re_rf = test_model(rf_model, X_test, y_test, full_pipeline, 'RF')
    
    # Generate visualizations
    visualize_results(rf_model, X_test, y_test, full_pipeline, 'RF')