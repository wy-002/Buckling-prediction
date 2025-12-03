from sklearn.neural_network import MLPRegressor
from utils import *
import os
from pathlib import Path

def train_mlp(data_prepared, data_labels, save_path='./save_model/MLP/mlp_best_model_01.joblib'):
    Path('./save_model/MLP/').mkdir(parents=True, exist_ok=True)
    
    param_grid_mlp = {
        "hidden_layer_sizes": [
            (100, 50), (120, 80), (150, 100), (80, 50), (50, 30), (120, 50), (150, 80),
            (100, 100), (120, 120), (150, 150), (80, 80), (50, 50), (100, 80),
            (120, 100), (150, 120), (200, 150), (100, 60), (120, 60), (150, 100),
            (100, 80, 50), (120, 100, 80), (150, 120, 100),
            (100, 50, 30), (120, 80, 40), (150, 100, 50),
            (80, 60, 30), (100, 80, 40), (120, 100, 50), (150, 120, 80),
            (50, 40, 20), (80, 50, 30), (100, 60, 30), (120, 80, 30)
        ],
        'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.005, 0.01, 0.05, 0.1],
        'batch_size': [16, 32, 64],
        'early_stopping': [True],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.0005, 0.001, 0.003, 0.006, 0.01, 0.03, 0.05],
        'max_iter': [2000],
        'solver': ['adam'],
        'random_state': [42]
    }
    
    mlp_model, train_rmse_mlp, train_mae_mlp, train_r2_mlp = train_and_evaluate_model(
        data_prepared, data_labels, 
        MLPRegressor(), 
        param_grid_mlp, 
        'neg_mean_squared_error', 
        save_model_path=save_path,
        model_name='MLP'
    )
    
    if mlp_model is not None:
        print("\n为MLP模型生成学习曲线...")
        mlp_learning_curve_info = plot_learning_curve(
            mlp_model, data_prepared, data_labels, 
            model_name='MLP', cv=5,
            save_path='./save_model/MLP/MLP_learning_curve.png'
        )
        
        if mlp_learning_curve_info:
            print(f"MLP学习曲线分析完成:")
            print(f"  最终验证R²: {mlp_learning_curve_info['final_test_score']:.4f}")
            print(f"  数据充分性: {mlp_learning_curve_info['data_sufficiency']}")
    
    return mlp_model

if __name__ == "__main__":
    Path('./save_model/MLP/').mkdir(parents=True, exist_ok=True)
    
    data = load_data()
    if data is None:
        exit()
    
    target_column = "Pcr"
    
    train_set, test_set = split_data(data, target_column)
    
    data_labels = train_set[target_column].copy()
    data_num = train_set.drop(target_column, axis=1)
    num_attribs = list(data_num.columns)
    
    data_prepared, full_pipeline, valid_attribs = prepare_data(data_num, num_attribs)
    dump(full_pipeline, './save_model/MLP/full_pipeline.joblib')
    
    mlp_model = train_mlp(data_prepared, data_labels)
    
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    
    print("MLP Testing:")
    final_predictions_mlp, final_re_mlp = test_model(mlp_model, X_test, y_test, full_pipeline, 'MLP')