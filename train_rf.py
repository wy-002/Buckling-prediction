from sklearn.ensemble import RandomForestRegressor
from utils import *
import os

def train_rf(data_prepared, data_labels, save_path='./save_model/RF/rf_best_model_01.joblib'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    param_grid_rf = {
        "n_estimators": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,130, 140, 150, 160],
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
        "bootstrap": [True],
        "random_state": [42]
    }
    
    print("--------------------------------------------------")
    rf_model, train_rmse_rf, train_mae_rf, train_r2_rf = train_and_evaluate_model(
        data_prepared, data_labels, 
        RandomForestRegressor(), 
        param_grid_rf, 
        'neg_mean_squared_error', 
        save_model_path=save_path,
        model_name='Random Forest'
    )
    print("--------------------------------------------------")
    
    if rf_model is not None:
        print("\n为Random Forest模型生成学习曲线...")
        rf_learning_curve_info = plot_learning_curve(
            rf_model, data_prepared, data_labels, 
            model_name='RF', cv=5,
            save_path='./save_model/RF/RF_learning_curve.png'
        )
        
        if rf_learning_curve_info:
            print(f"RF学习曲线分析完成:")
            print(f"  最终验证R²: {rf_learning_curve_info['final_test_score']:.4f}")
            print(f"  数据充分性: {rf_learning_curve_info['data_sufficiency']}")
    
    return rf_model

if __name__ == "__main__":
    data = load_data()
    if data is None:
        exit()
    
    target_column = "Pcr"
    
    train_set, test_set = split_data(data, target_column)
    
    data_labels = train_set[target_column].copy()
    data_num = train_set.drop(target_column, axis=1)
    num_attribs = list(data_num.columns)
    
    data_prepared, full_pipeline, valid_attribs = prepare_data(data_num, num_attribs, is_tree_model=True)
    
    os.makedirs('./save_model/RF', exist_ok=True)
    dump(full_pipeline, './save_model/RF/full_pipeline.joblib')
    
    rf_model = train_rf(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("RF Testing:")

    final_predictions_rf, final_re_rf = test_model(rf_model, X_test, y_test, full_pipeline, 'RF')
