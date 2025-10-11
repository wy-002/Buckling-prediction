from sklearn.neighbors import KNeighborsRegressor
from utils import *

def train_knn(data_prepared, data_labels, save_path='./save_model/KNN/knn_best_model_01.joblib'):
    param_grid_knn = {
        "n_neighbors": [4],
        "weights": ["distance"],
        'leaf_size': [53],
        "p": [2],
        'metric': ['minkowski'],
    }
    
    print("--------------------------------------------------")
    print("KNN:")
    knn_model, train_rmse_knn, train_mae_knn, train_r2_knn = train_and_evaluate_model(
        data_prepared, data_labels, 
        KNeighborsRegressor(), 
        param_grid_knn, 
        'neg_mean_squared_error', 
        save_model_path=save_path
    )
    print("--------------------------------------------------")
    
    return knn_model

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
    dump(full_pipeline, './save_model/KNN/full_pipeline.joblib')
    
    knn_model = train_knn(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("KNN Testing:")
    final_predictions_knn, final_re_knn = test_model(knn_model, X_test, y_test, full_pipeline, 'KNN')
    
    # Generate visualizations
    visualize_results(knn_model, X_test, y_test, full_pipeline, 'KNN')