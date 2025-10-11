from sklearn import svm
from utils import *

def train_svm(data_prepared, data_labels, save_path='./save_model/SVM/svm_best_model_01.joblib'):
    param_grid_svm = {
        'kernel': ['rbf'],
        'C': [54],
        'epsilon': [0.015],
        'gamma': ['scale'],
        'tol': [0.0093]
    }
    
    print("--------------------------------------------------")
    print("SVM:")
    svm_model, train_rmse_svm, train_mae_svm, train_r2_svm = train_and_evaluate_model(
        data_prepared, data_labels, 
        svm.SVR(), 
        param_grid_svm, 
        'neg_mean_squared_error', 
        save_model_path=save_path
    )
    print("--------------------------------------------------")
    
    return svm_model

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
    dump(full_pipeline, './save_model/SVM/full_pipeline.joblib')
    
    svm_model = train_svm(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("SVM Testing:")
    final_predictions_svm, final_re_svm = test_model(svm_model, X_test, y_test, full_pipeline, 'SVM')
    
    # Generate visualizations
    visualize_results(svm_model, X_test, y_test, full_pipeline, 'SVM')