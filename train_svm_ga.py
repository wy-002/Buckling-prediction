from sklearn import svm
from sko.GA import GA
import numpy as np
from utils import *
import os
from pathlib import Path

def ga_svm_objective_function(params, X, y, cv_folds=5):
    try:
        C = max(float(params[0]), 0.1)
        epsilon = max(float(params[1]), 0.01)
        
        model = svm.SVR(
            kernel='rbf',
            C=C,
            gamma='scale',
            epsilon=epsilon
        )
        
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        mean_score = np.mean(scores)
        
        return -mean_score
        
    except Exception as e:
        print(f"GA-SVM目标函数错误: {e}")
        return 1.0

def train_ga_svm(data_prepared, data_labels, save_path='./save_model/GA_SVM/ga_svm_best_model.joblib'):
    Path('./save_model/GA_SVM/').mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("开始训练 GA-SVM (遗传算法优化SVM)")
    print("=" * 60)
    
    param_bounds = [(0.1, 100), (0.01, 1.0)]
    
    ga = GA(
        func=lambda params: ga_svm_objective_function(params, data_prepared, data_labels),
        n_dim=2,
        size_pop=30,
        max_iter=50,
        prob_mut=0.001,
        lb=[b[0] for b in param_bounds],
        ub=[b[1] for b in param_bounds],
        precision=1e-4
    )
    
    print("开始遗传算法优化...")
    best_params, best_score = ga.run()
    
    best_C = max(float(best_params[0]), 0.1)
    best_epsilon = max(float(best_params[1]), 0.01)
    best_r2 = -float(best_score)
    
    print(f"\nGA-SVM 优化完成!")
    print(f"最佳参数: C={best_C:.4f}, epsilon={best_epsilon:.4f}")
    print(f"最佳R²分数: {best_r2:.4f}")
    
    final_svm_model = svm.SVR(
        kernel='rbf',
        C=best_C,
        gamma='scale',
        epsilon=best_epsilon
    )
    
    final_svm_model.fit(data_prepared, data_labels)
    
    y_train_pred = final_svm_model.predict(data_prepared)
    train_r2 = r2_score(data_labels, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(data_labels, y_train_pred))
    train_mae = mean_absolute_error(data_labels, y_train_pred)
    train_mape = mean_absolute_percentage_error(data_labels, y_train_pred)
    
    print(f"\nGA-SVM 训练集性能:")
    print(f"  R²: {train_r2:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  MAPE: {train_mape:.2f}%")
    
    print(f"\nGA-SVM 交叉验证:")
    cv_results = cross_validate_model(final_svm_model, data_prepared, data_labels, 
                                    cv=5, model_name="GA-SVM")
    
    if save_path:
        dump(final_svm_model, save_path)
        print(f"GA-SVM模型已保存到: {save_path}")
    
    print("\n为GA-SVM生成学习曲线...")
    ga_svm_learning_curve_info = plot_learning_curve(
        final_svm_model, data_prepared, data_labels, 
        model_name='GA_SVM', cv=5,
        save_path='./save_model/GA_SVM/GA_SVM_learning_curve.png'
    )
    
    return final_svm_model, [best_C, best_epsilon]

if __name__ == "__main__":
    Path('./save_model/GA_SVM/').mkdir(parents=True, exist_ok=True)
    
    data = load_data()
    if data is None:
        exit()
    
    target_column = "Pcr"
    
    train_set, test_set = split_data(data, target_column)
    
    data_labels = train_set[target_column].copy()
    data_num = train_set.drop(target_column, axis=1)
    num_attribs = list(data_num.columns)
    
    data_prepared, full_pipeline, valid_attribs = prepare_data(data_num, num_attribs)
    
    dump(full_pipeline, './save_model/GA_SVM/full_pipeline.joblib')
    
    ga_svm_model, best_params = train_ga_svm(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("\nGA-SVM 测试集评估:")
    final_predictions, final_re = test_model(ga_svm_model, X_test, y_test, full_pipeline, 'GA_SVM')
    
    print(f"\nGA-SVM 训练完成!")
    print(f"最佳参数: C={best_params[0]:.4f}, epsilon={best_params[1]:.4f}")