from sklearn import svm
from utils import *
import os
from pathlib import Path

def train_svm(data_prepared, data_labels, save_path='./save_model/SVM/svm_best_model_01.joblib'):
    Path('./save_model/SVM/').mkdir(parents=True, exist_ok=True)
    
    param_grid_svm = {
        'kernel': ['rbf'],
        'C': [0.1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        'epsilon': [0.01, 0.05, 0.1, 0.15, 0.2],
        'gamma': ['scale']
    }
    
    print("--------------------------------------------------")
    svm_model, train_rmse_svm, train_mae_svm, train_r2_svm = train_and_evaluate_model(
        data_prepared, data_labels, 
        svm.SVR(), 
        param_grid_svm, 
        'neg_mean_squared_error', 
        save_model_path=save_path,
        model_name='SVM'
    )
    print("--------------------------------------------------")
    
    if svm_model is not None:
        print("\n为SVM模型生成学习曲线...")
        svm_learning_curve_info = plot_learning_curve(
            svm_model, data_prepared, data_labels, 
            model_name='SVM', cv=5,
            save_path='./save_model/SVM/SVM_learning_curve.png'
        )
        
        if svm_learning_curve_info:
            print(f"SVM学习曲线分析完成:")
            print(f"  最终验证R²: {svm_learning_curve_info['final_test_score']:.4f}")
            print(f"  数据充分性: {svm_learning_curve_info['data_sufficiency']}")
    
    return svm_model

if __name__ == "__main__":
    Path('./save_model/SVM/').mkdir(parents=True, exist_ok=True)
    
    data = load_data()
    if data is None:
        exit()
    
    target_column = "Pcr"
    
    train_set, test_set = split_data(data, target_column)
    
    data_labels = train_set[target_column].copy()
    data_num = train_set.drop(target_column, axis=1)
    num_attribs = list(data_num.columns)
    
    data_prepared, full_pipeline, valid_attribs = prepare_data(data_num, num_attribs)
    dump(full_pipeline, './save_model/SVM/full_pipeline.joblib')
    
    svm_model = train_svm(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("SVM Testing:")
    final_predictions_svm, final_re_svm = test_model(svm_model, X_test, y_test, full_pipeline, 'SVM')