from sklearn.neural_network import MLPRegressor
from utils import *

def train_mlp(data_prepared, data_labels, save_path='./save_model/MLP/mlp_best_model_01.joblib'):
    param_grid_mlp = {
        "hidden_layer_sizes": [(160, 180, 120)],
        'activation': ['relu'],
        'alpha': [0.0082],
        'batch_size': [16],
        'early_stopping': [True], 
        'learning_rate': ['adaptive'], 
        'learning_rate_init': [0.007],  
        'max_iter': [1000],
        'solver': ['adam'],
        'random_state': [42] 
    }
    
    print("--------------------------------------------------")
    print("MLP:")
    mlp_model, train_rmse_mlp, train_mae_mlp, train_r2_mlp = train_and_evaluate_model(
        data_prepared, data_labels, 
        MLPRegressor(), 
        param_grid_mlp, 
        'neg_mean_squared_error', 
        save_model_path=save_path
    )
    print("--------------------------------------------------")
    
    return mlp_model

def visualize_results(model, X_test, y_test, full_pipeline, model_name):
    """Generate and save visualization plots for the model results."""
    X_test_prepared = full_pipeline.transform(X_test)
    y_pred = model.predict(X_test_prepared)
    
    # Create plots
    plot_predictions_vs_actual(y_test, y_pred, model_name)
    plot_error_distribution(y_test - y_pred, model_name)
    plot_residuals(y_test, y_pred, model_name)

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
    dump(full_pipeline, './save_model/MLP/full_pipeline.joblib')
    
    mlp_model = train_mlp(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("MLP Testing:")
    final_predictions_mlp, final_re_mlp = test_model(mlp_model, X_test, y_test, full_pipeline, 'MLP')
    
    # Generate visualizations
    visualize_results(mlp_model, X_test, y_test, full_pipeline, 'MLP')