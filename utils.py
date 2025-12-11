import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold, cross_val_score, learning_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import friedmanchisquare, rankdata, shapiro, ttest_rel, wilcoxon
from sklearn.base import clone
import json
import itertools

def create_directory_if_not_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

def load_data(data_path='./data/pcr_400.csv'):
    try:
        csv_path = os.path.join(data_path)
        data = pd.read_csv(csv_path)
        print(f"成功加载数据，形状: {data.shape}")
        return data
    except FileNotFoundError:
        print("错误: 数据文件未找到，请检查文件路径。")
        return None
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return None

def prepare_data(data, num_attribs, is_tree_model=False):
    constant_features = []
    valid_attribs = []
    
    for attrib in num_attribs:
        if data[attrib].nunique() <= 1:
            print(f"移除常数特征: {attrib} (唯一值: {data[attrib].unique()})")
            constant_features.append(attrib)
        else:
            valid_attribs.append(attrib)
            
    if constant_features:
        print(f"共移除 {len(constant_features)} 个常数特征")
    
    if is_tree_model:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="mean")),
        ])
        print(f"  使用树模型预处理管道：仅进行缺失值填充")
    else:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="mean")),
            ('mm_scaler', MinMaxScaler())  # 保留归一化
        ])
        print(f"  使用标准预处理管道：缺失值填充 + 归一化")
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, valid_attribs),
    ])
    
    return full_pipeline.fit_transform(data), full_pipeline, valid_attribs


def split_data(data, target_column, stratify_column=None, test_size=0.2, random_state=42):
    if stratify_column and stratify_column not in data.columns:
        print(f"警告: 数据集中未找到列 '{stratify_column}'。继续不使用分层抽样。")
        stratify_column = None

    if stratify_column:
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        data['stratify_column_binned'] = discretizer.fit_transform(data[[stratify_column]])
        stratify_column = 'stratify_column_binned'

    train_set, test_set = train_test_split(
        data, 
        test_size=test_size, 
        stratify=data[stratify_column] if stratify_column else None, 
        random_state=random_state
    )

    if stratify_column == 'stratify_column_binned':
        train_set = train_set.drop(columns=['stratify_column_binned'])
        test_set = test_set.drop(columns=['stratify_column_binned'])

    print(f"训练集大小: {len(train_set)}, 测试集大小: {len(test_set)}")
    return train_set, test_set

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def diagnose_data_issues(data, target_column):
    print("\n" + "="*50)
    print("数据诊断报告")
    print("="*50)
    print(f"数据集形状: {data.shape}")
    print(f"目标变量 '{target_column}' 统计:")
    print(f"  范围: {data[target_column].min():.6f} - {data[target_column].max():.6f}")
    print(f"  均值: {data[target_column].mean():.6f}")
    print(f"  方差: {data[target_column].var():.6f}")
    print(f"  标准差: {data[target_column].std():.6f}")
    
    if data[target_column].std() < 1e-6:
        print("❌ 严重警告: 目标变量方差接近零！")
    elif data[target_column].std() < 0.01 * abs(data[target_column].mean()):
        print("⚠️  警告: 目标变量方差过小，可能导致虚假的高R²值")
    
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print(f"⚠️  警告: 发现缺失值:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"  {col}: {count} 个缺失值")
    
    print(f"\n特征与目标变量的相关性:")
    numeric_data = data.select_dtypes(include=[np.number])
    constant_cols = numeric_data.columns[numeric_data.nunique() <= 1]
    valid_data = numeric_data.drop(columns=constant_cols)
    
    if target_column in valid_data.columns:
        correlations = valid_data.corr()[target_column].abs().sort_values(ascending=False)
        for feature, corr in correlations.items():
            if feature != target_column:
                print(f"  {feature}: {corr:.4f}")
    else:
        print("  无法计算相关性：目标变量不在有效数据中")
    
    constant_features = data.columns[data.nunique() <= 1]
    if len(constant_features) > 0:
        print(f"❌ 警告: 发现常数特征: {list(constant_features)}")
    
    print("="*50)

def cross_validate_model(model, X, y, cv=5, model_name="Model"):
    print(f"\n{model_name} - {cv}折交叉验证结果:")
    print("-" * 40)
    
    scoring = {
        'rmse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error', 
        'r2': 'r2'
    }
    
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, 
        cv=kfold, 
        scoring=scoring,
        return_train_score=True,
        return_estimator=True
    )
    
    metrics = {
        'train_rmse': np.sqrt(-cv_results['train_rmse']),
        'test_rmse': np.sqrt(-cv_results['test_rmse']),
        'train_mae': -cv_results['train_mae'],
        'test_mae': -cv_results['test_mae'],
        'train_r2': cv_results['train_r2'],
        'test_r2': cv_results['test_r2']
    }
    
    for metric_name in ['rmse', 'mae', 'r2']:
        train_scores = metrics[f'train_{metric_name}']
        test_scores = metrics[f'test_{metric_name}']
        
        print(f"{metric_name.upper():>4} | 训练集: {train_scores.mean():.4f} (±{train_scores.std():.4f}) | "
              f"测试集: {test_scores.mean():.4f} (±{test_scores.std():.4f})")
    
    train_test_gap = metrics['train_r2'].mean() - metrics['test_r2'].mean()
    if train_test_gap > 0.1:
        print(f"⚠️  过拟合警告: 训练-测试集R²差距较大 ({train_test_gap:.3f})")
    
    return cv_results

def train_and_evaluate_model(X, y, model, param_grid, scoring, cv=5, refit=True, save_model_path=None, model_name="Model"):
    try:
        print(f"\n{model_name} 训练开始:")
        print("-" * 30)
        
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, 
                                 return_train_score=True, refit=refit, n_jobs=-1)
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        
        print(f"\n{model_name} 最佳参数交叉验证:")
        best_cv_results = cross_validate_model(best_model, X, y, cv=cv, model_name=f"{model_name} (最优)")
        
        y_train_pred = best_model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
        train_mae = mean_absolute_error(y, y_train_pred)
        train_r2 = r2_score(y, y_train_pred)
        train_mape = mean_absolute_percentage_error(y, y_train_pred)
        
        print(f"\n{model_name} 训练集最终结果:")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  MAPE: {train_mape:.2f}%")
        print(f"  R²: {train_r2:.4f}")
        
        print(f"\n最佳参数: {grid_search.best_params_}")
        
        if save_model_path:
            create_directory_if_not_exists(save_model_path)
            dump(best_model, save_model_path)
            print(f"模型已保存到: {save_model_path}")
        
        return best_model, train_rmse, train_mae, train_r2
        
    except Exception as e:
        print(f"模型训练和评估过程中发生错误: {e}")
        return None, None, None, None

def test_model(model, X_test, y_test, full_pipeline, model_name):
    try:
        X_test_prepared = full_pipeline.transform(X_test)
        final_predictions = model.predict(X_test_prepared)
        
        final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
        final_mae = mean_absolute_error(y_test, final_predictions)
        final_r2 = r2_score(y_test, final_predictions)
        final_mape = mean_absolute_percentage_error(y_test, final_predictions)
        final_re = (y_test - final_predictions) / y_test
        
        print(f"\n{model_name} 测试集结果:")
        print(f"  RMSE: {final_rmse:.4f}")
        print(f"  MAE: {final_mae:.4f}")
        print(f"  MAPE: {final_mape:.2f}%")
        print(f"  R²: {final_r2:.4f}")
        print(f"  最大相对误差: {max(abs(final_re * 100)):.2f}%")
        print(f"  平均相对误差: {np.mean(abs(final_re * 100)):.2f}%")
        
        results_path = f'./save_model/{model_name}/{model_name}_results.csv'
        create_directory_if_not_exists(results_path)
        
        results = pd.DataFrame({
            '实际值': y_test,
            '预测值': final_predictions,
            '绝对误差': y_test - final_predictions,
            '相对误差_百分比': final_re * 100,
            '绝对百分比误差': np.abs((y_test - final_predictions) / y_test) * 100
        })
        results.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"结果已保存到: {results_path}")
        
        return final_predictions, final_re
        
    except Exception as e:
        print(f"模型测试过程中发生错误: {e}")
        return None, None

def plot_learning_curve(model, X, y, model_name, cv=5, train_sizes=None, scoring='r2', save_path=None):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    print(f"正在为 {model_name} 计算学习曲线...")
    
    try:
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, 
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            shuffle=True
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        learning_curve_data = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_scores_mean.tolist(),
            'train_scores_std': train_scores_std.tolist(),
            'test_scores_mean': test_scores_mean.tolist(),
            'test_scores_std': test_scores_std.tolist(),
            'scoring': scoring
        }

        data_save_path = f'./save_model/{model_name}/{model_name}_learning_curve_data.json'
        create_directory_if_not_exists(data_save_path)
        with open(data_save_path, 'w', encoding='utf-8') as f:
            json.dump(learning_curve_data, f, indent=2, ensure_ascii=False)
        print(f"学习曲线数据已保存到: {data_save_path}")

        final_train_score = train_scores_mean[-1]
        final_test_score = test_scores_mean[-1]
        gap = final_train_score - final_test_score
        
        if gap > 0.1:
            convergence_status = "⚠️ 可能存在过拟合"
            status_color = "red"
        elif gap < 0.05:
            convergence_status = "✅ 拟合良好"
            status_color = "green"
        else:
            convergence_status = "🟡 拟合正常"
            status_color = "orange"
        
        if final_test_score > 0.9 and gap < 0.05:
            data_sufficiency = "✅ 数据相对充分"
        elif final_test_score > 0.8 and gap < 0.1:
            data_sufficiency = "🟡 数据基本充分"
        else:
            data_sufficiency = "❌ 数据可能不足"
        
        convergence_analysis = {
            'final_train_score': float(final_train_score),
            'final_test_score': float(final_test_score),
            'gap': float(gap),
            'convergence_status': convergence_status,
            'data_sufficiency': data_sufficiency,
            'is_converged': bool(gap < 0.1),
            'needs_more_data': bool(gap > 0.1 or final_test_score < 0.8),
            'learning_curve_data_path': data_save_path
        }
        
        return convergence_analysis
        
    except Exception as e:
        print(f"❌ {model_name} 学习曲线计算失败: {e}")
        return None

def analyze_data_sufficiency_all_models(models_dict, X, y, cv=5):
    print("\n" + "="*60)
    print("数据充分性与模型收敛性分析 - 学习曲线方法")
    print("="*60)
    
    results = {}
    
    for model_name, model in models_dict.items():
        if model is not None:
            print(f"\n分析 {model_name} 模型...")
            try:
                save_path = f'./save_model/{model_name}/{model_name}_learning_curve.png'
                convergence_info = plot_learning_curve(model, X, y, model_name, cv, save_path=save_path)
                results[model_name] = convergence_info
                
                if convergence_info:
                    print(f"  最终验证R²: {convergence_info['final_test_score']:.4f}")
                    print(f"  训练-验证差距: {convergence_info['gap']:.4f}")
                    print(f"  收敛状态: {convergence_info['convergence_status']}")
                    print(f"  数据充分性: {convergence_info['data_sufficiency']}")
                    
            except Exception as e:
                print(f"  ❌ {model_name} 学习曲线分析失败: {e}")
                results[model_name] = None
    
    converged_models = [name for name, info in results.items() 
                       if info and info['is_converged']]
    sufficient_models = [name for name, info in results.items() 
                        if info and not info['needs_more_data']]
    
    print(f"\n{'='*50}")
    print("总体数据充分性评估")
    print(f"{'='*50}")
    
    print(f"收敛模型数量: {len(converged_models)}/{len(results)}")
    print(f"数据充分模型数量: {len(sufficient_models)}/{len(results)}")
    
    if len(converged_models) >= len(results) * 0.8:
        print("✅ 总体结论: 在当前数据量下，大多数模型性能已收敛")
        print("✅ 建议: 当前数据量相对充分，可继续优化模型")
    else:
        print("⚠️  总体结论: 部分模型可能仍需更多数据以达到完全收敛")
        print("📊 建议: 考虑收集更多数据，特别是验证集性能仍在提升的模型")
    
    overall_analysis_path = './save_model/data_sufficiency_analysis.json'
    
    overall_analysis = {
        'converged_models': converged_models,
        'sufficient_models': sufficient_models,
        'total_models': len(results),
        'converged_ratio': float(len(converged_models) / len(results)),
        'sufficient_ratio': float(len(sufficient_models) / len(results)),
        'model_details': {}
    }
    
    for model_name, info in results.items():
        if info:
            overall_analysis['model_details'][model_name] = {
                'final_train_score': float(info['final_train_score']),
                'final_test_score': float(info['final_test_score']),
                'gap': float(info['gap']),
                'convergence_status': info['convergence_status'],
                'data_sufficiency': info['data_sufficiency'],
                'is_converged': bool(info['is_converged']),
                'needs_more_data': bool(info['needs_more_data']),
                'learning_curve_data_path': info['learning_curve_data_path']
            }
        else:
            overall_analysis['model_details'][model_name] = None
    
    with open(overall_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(overall_analysis, f, indent=2, ensure_ascii=False)
    print(f"\n总体分析结果已保存到: {overall_analysis_path}")
    

    return results
