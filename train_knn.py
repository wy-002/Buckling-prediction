from sklearn.neighbors import KNeighborsRegressor
from utils import *
import os
from pathlib import Path

def train_knn(data_prepared, data_labels, save_path='./save_model/KNN/knn_best_model_01.joblib'):
    Path('./save_model/KNN/').mkdir(parents=True, exist_ok=True)
    
    param_grid_knn = {
        "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        'metric': ['minkowski']
    }
    
    print("--------------------------------------------------")
    knn_model, train_rmse_knn, train_mae_knn, train_r2_knn = train_and_evaluate_model(
        data_prepared, data_labels, 
        KNeighborsRegressor(), 
        param_grid_knn, 
        'neg_mean_squared_error', 
        save_model_path=save_path,
        model_name='KNN'
    )
    print("--------------------------------------------------")
    
    if knn_model is not None:
        print("\n为KNN模型生成学习曲线...")
        knn_learning_curve_info = plot_learning_curve(
            knn_model, data_prepared, data_labels, 
            model_name='KNN', cv=5,
            save_path='./save_model/KNN/KNN_learning_curve.png'
        )
        
        if knn_learning_curve_info:
            print(f"KNN学习曲线分析完成:")
            print(f"  最终验证R²: {knn_learning_curve_info['final_test_score']:.4f}")
            print(f"  数据充分性: {knn_learning_curve_info['data_sufficiency']}")
    
    return knn_model

if __name__ == "__main__":
    Path('./save_model/KNN/').mkdir(parents=True, exist_ok=True)
    
    data = load_data()
    if data is None:
        exit()
    
    target_column = "Pcr"
    stratify_column="Pcr"
    diagnose_data_issues(data, target_column)

    train_set, test_set = split_data(data, target_column, stratify_column)

    data_labels = train_set[target_column].copy()
    data_num = train_set.drop(target_column, axis=1)
    num_attribs = list(data_num.columns)
    
    data_prepared, full_pipeline, valid_attribs = prepare_data(data_num, num_attribs)
    dump(full_pipeline, './save_model/KNN/full_pipeline.joblib')
    
    knn_model = train_knn(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("KNN Testing:")
    final_predictions_knn, final_re_knn = test_model(knn_model, X_test, y_test, full_pipeline, 'KNN')