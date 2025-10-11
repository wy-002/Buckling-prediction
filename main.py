from utils import *
from train_rf import train_random_forest, visualize_results as visualize_rf
from train_mlp import train_mlp, visualize_results as visualize_mlp
from train_svm import train_svm, visualize_results as visualize_svm
from train_knn import train_knn, visualize_results as visualize_knn

def main():
    # Load and prepare data
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
    dump(full_pipeline, './save_model/main/full_pipeline.joblib')
    
    # Train all models
    rf_model = train_random_forest(data_prepared, data_labels)
    mlp_model = train_mlp(data_prepared, data_labels)
    svm_model = train_svm(data_prepared, data_labels)
    knn_model = train_knn(data_prepared, data_labels)
    
    # Test all models
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("\nTesting all models:")
    final_predictions = []
    final_res = []
    
    print("RF Testing:")
    final_predictions_rf, final_re_rf = test_model(rf_model, X_test, y_test, full_pipeline, 'RF')
    final_predictions.append(final_predictions_rf)
    final_res.append(final_re_rf)
    visualize_rf(rf_model, X_test, y_test, full_pipeline, 'RF')
    
    print("\nMLP Testing:")
    final_predictions_mlp, final_re_mlp = test_model(mlp_model, X_test, y_test, full_pipeline, 'MLP')
    final_predictions.append(final_predictions_mlp)
    final_res.append(final_re_mlp)
    visualize_mlp(mlp_model, X_test, y_test, full_pipeline, 'MLP')
    
    print("\nSVM Testing:")
    final_predictions_svm, final_re_svm = test_model(svm_model, X_test, y_test, full_pipeline, 'SVM')
    final_predictions.append(final_predictions_svm)
    final_res.append(final_re_svm)
    visualize_svm(svm_model, X_test, y_test, full_pipeline, 'SVM')
    
    print("\nKNN Testing:")
    final_predictions_knn, final_re_knn = test_model(knn_model, X_test, y_test, full_pipeline, 'KNN')
    final_predictions.append(final_predictions_knn)
    final_res.append(final_re_knn)
    visualize_knn(knn_model, X_test, y_test, full_pipeline, 'KNN')
    
    # Plot relative errors
    model_names = ['RF', 'MLP', 'SVM', 'KNN']
    plot_relative_errors(model_names, final_res)

if __name__ == "__main__":
    main()