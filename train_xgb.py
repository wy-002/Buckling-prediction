from xgboost import XGBRegressor
from utils import *
import os

def train_xgb(data_prepared, data_labels, save_path='./save_model/XGB/xgb_best_model_01.joblib'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    param_grid_xgb = {
        "n_estimators": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "random_state": [42]
    }
    
    print("--------------------------------------------------")
    xgb_model, train_rmse_xgb, train_mae_xgb, train_r2_xgb = train_and_evaluate_model(
        data_prepared, data_labels, 
        XGBRegressor(), 
        param_grid_xgb, 
        'neg_mean_squared_error', 
        save_model_path=save_path,
        model_name='XGBoost'
    )
    print("--------------------------------------------------")
    
    if xgb_model is not None:
        print("\n为XGBoost模型生成学习曲线...")
        xgb_learning_curve_info = plot_learning_curve(
            xgb_model, data_prepared, data_labels, 
            model_name='XGB', cv=5,
            save_path='./save_model/XGB/XGB_learning_curve.png'
        )
        
        if xgb_learning_curve_info:
            print(f"XGBoost学习曲线分析完成:")
            print(f"  最终验证R²: {xgb_learning_curve_info['final_test_score']:.4f}")
            print(f"  数据充分性: {xgb_learning_curve_info['data_sufficiency']}")
    
    return xgb_model

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
    
    os.makedirs('./save_model/XGB', exist_ok=True)
    dump(full_pipeline, './save_model/XGB/full_pipeline.joblib')
    
    xgb_model = train_xgb(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("XGB Testing:")
    final_predictions_xgb, final_re_xgb = test_model(xgb_model, X_test, y_test, full_pipeline, 'XGB')