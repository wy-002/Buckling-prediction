from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from utils import *
import os
import pandas as pd
import numpy as np

def train_gp(data_prepared, data_labels, save_path='./save_model/GP/gp_best_model_01.joblib'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    param_grid_gp = {
        "kernel": [
            ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)),
            ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)),
        ],
        "alpha": [1e-10, 1e-5, 1e-2],
        "n_restarts_optimizer": [3, 5, 7, 9, 10]
    }
    
    print("--------------------------------------------------")
    gp_model, train_rmse_gp, train_mae_gp, train_r2_gp = train_and_evaluate_model(
        data_prepared, data_labels, 
        GaussianProcessRegressor(), 
        param_grid_gp, 
        'neg_mean_squared_error', 
        save_model_path=save_path,
        model_name='Gaussian Process'
    )
    print("--------------------------------------------------")
    
    if gp_model is not None:
        print("\n为Gaussian Process模型生成学习曲线...")
        gp_learning_curve_info = plot_learning_curve(
            gp_model, data_prepared, data_labels, 
            model_name='GP', cv=5,
            save_path='./save_model/GP/GP_learning_curve.png'
        )
        
        if gp_learning_curve_info:
            print(f"GP学习曲线分析完成:")
            print(f"  最终验证R²: {gp_learning_curve_info['final_test_score']:.4f}")
            print(f"  数据充分性: {gp_learning_curve_info['data_sufficiency']}")
    
    return gp_model

if __name__ == "__main__":
    data = load_data()
    if data is None:
        exit()
    
    target_column = "Pcr"
    
    train_set, test_set = split_data(data, target_column)
    
    data_labels = train_set[target_column].copy()
    data_num = train_set.drop(target_column, axis=1)
    num_attribs = list(data_num.columns)
    
    data_prepared, full_pipeline, valid_attribs = prepare_data(data_num, num_attribs)
    
    os.makedirs('./save_model/GP', exist_ok=True)
    dump(full_pipeline, './save_model/GP/full_pipeline.joblib')
    
    gp_model = train_gp(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("GP Testing:")
    final_predictions_gp, final_re_gp = test_model(gp_model, X_test, y_test, full_pipeline, 'GP')