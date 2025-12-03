from sklearn import svm
from sko.PSO import PSO
import numpy as np
from utils import *
import os
from pathlib import Path

def pso_svm_objective_function(params, X, y, cv_folds=5):
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
        print(f"PSO-SVM目标函数错误: {e}")
        return 1.0

def train_pso_svm(data_prepared, data_labels, save_path='./save_model/PSO_SVM/pso_svm_best_model.joblib'):
    Path('./save_model/PSO_SVM/').mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("开始训练 PSO-SVM (粒子群算法优化SVM)")
    print("=" * 60)
    
    param_bounds = [(0.1, 100), (0.01, 1.0)]
    
    pso = PSO(
        func=lambda params: pso_svm_objective_function(params, data_prepared, data_labels),
        n_dim=2,
        pop=30,
        max_iter=50,
        lb=[b[0] for b in param_bounds],
        ub=[b[1] for b in param_bounds],
        w=0.8,
        c1=0.5,
        c2=0.5
    )
    
    print("开始粒子群优化...")
    pso.run()
    
    best_params = pso.gbest_x
    best_score = pso.gbest_y
    
    best_C = max(float(best_params[0]), 0.1)
    best_epsilon = max(float(best_params[1]), 0.01)
    best_r2 = -float(best_score)
    
    print(f"\nPSO-SVM 优化完成!")
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
    
    print(f"\nPSO-SVM 训练集性能:")
    print(f"  R²: {train_r2:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  MAPE: {train_mape:.2f}%")
    
    print(f"\nPSO-SVM 交叉验证:")
    cv_results = cross_validate_model(final_svm_model, data_prepared, data_labels, 
                                    cv=5, model_name="PSO-SVM")
    
    if save_path:
        dump(final_svm_model, save_path)
        print(f"PSO-SVM模型已保存到: {save_path}")
    
    print("\n为PSO-SVM生成学习曲线...")
    pso_svm_learning_curve_info = plot_learning_curve(
        final_svm_model, data_prepared, data_labels, 
        model_name='PSO_SVM', cv=5,
        save_path='./save_model/PSO_SVM/PSO_SVM_learning_curve.png'
    )
    
    return final_svm_model, [best_C, best_epsilon]

if __name__ == "__main__":
    Path('./save_model/PSO_SVM/').mkdir(parents=True, exist_ok=True)
    
    data = load_data()
    if data is None:
        exit()
    
    target_column = "Pcr"
    
    train_set, test_set = split_data(data, target_column)
    
    data_labels = train_set[target_column].copy()
    data_num = train_set.drop(target_column, axis=1)
    num_attribs = list(data_num.columns)
    
    data_prepared, full_pipeline, valid_attribs = prepare_data(data_num, num_attribs)
    
    dump(full_pipeline, './save_model/PSO_SVM/full_pipeline.joblib')
    
    pso_svm_model, best_params = train_pso_svm(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("\nPSO-SVM 测试集评估:")
    final_predictions, final_re = test_model(pso_svm_model, X_test, y_test, full_pipeline, 'PSO_SVM')
    
    print(f"\nPSO-SVM 训练完成!")
    print(f"最佳参数: C={best_params[0]:.4f}, epsilon={best_params[1]:.4f}")