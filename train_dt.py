from sklearn.tree import DecisionTreeRegressor
from utils import *
import os

def train_dt(data_prepared, data_labels, save_path='./save_model/DT/dt_best_model_01.joblib'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    param_grid_dt = {
        "criterion": ["squared_error", "friedman_mse"],
        "splitter": ["best"],
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [3, 5, 7],
        "max_features": ["sqrt", "log2", None],
        "random_state": [42]
    }
    
    print("--------------------------------------------------")
    dt_model, train_rmse_dt, train_mae_dt, train_r2_dt = train_and_evaluate_model(
        data_prepared, data_labels, 
        DecisionTreeRegressor(), 
        param_grid_dt, 
        'neg_mean_squared_error', 
        save_model_path=save_path,
        model_name='Decision Tree'
    )
    print("--------------------------------------------------")
    
    if dt_model is not None:
        print("\n为Decision Tree模型生成学习曲线...")
        dt_learning_curve_info = plot_learning_curve(
            dt_model, data_prepared, data_labels, 
            model_name='DT', cv=5,
            save_path='./save_model/DT/DT_learning_curve.png'
        )
        
        if dt_learning_curve_info:
            print(f"DT学习曲线分析完成:")
            print(f"  最终验证R²: {dt_learning_curve_info['final_test_score']:.4f}")
            print(f"  数据充分性: {dt_learning_curve_info['data_sufficiency']}")
    
    return dt_model

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
    
    os.makedirs('./save_model/DT', exist_ok=True)
    dump(full_pipeline, './save_model/DT/full_pipeline.joblib')
    
    dt_model = train_dt(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("DT Testing:")

    final_predictions_dt, final_re_dt = test_model(dt_model, X_test, y_test, full_pipeline, 'DT')
