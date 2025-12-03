# 机器学习回归模型训练项目（中文）

这是一个用于训练和评估多种机器学习回归模型的Python项目，包含决策树、SVM、高斯过程、梯度提升、XGBoost、MLP、随机森林、KNN以及使用智能优化算法优化的SVM模型。

## 📁 项目结构

```
├── data/
│   └── pcr_400.csv              # 数据集
├── save_model/                  # 模型保存目录（运行后生成）
│   ├── DT/                      # 决策树模型相关文件
│   ├── SVM/                     # SVM模型相关文件
│   ├── GP/                      # 高斯过程模型相关文件
│   ├── GB/                      # 梯度提升模型相关文件
│   ├── XGB/                     # XGBoost模型相关文件
│   ├── MLP/                     # MLP神经网络模型相关文件
│   ├── RF/                      # 随机森林模型相关文件
│   ├── KNN/                     # KNN模型相关文件
│   ├── GA_SVM/                  # 遗传算法优化SVM模型相关文件
│   └── PSO_SVM/                 # 粒子群算法优化SVM模型相关文件
├── train_dt.py                  # 决策树模型训练脚本
├── train_svm.py                 # SVM模型训练脚本
├── train_gp.py                  # 高斯过程模型训练脚本
├── train_gb.py                  # 梯度提升模型训练脚本
├── train_xgb.py                 # XGBoost模型训练脚本
├── train_mlp.py                 # MLP模型训练脚本
├── train_rf.py                  # 随机森林模型训练脚本
├── train_knn.py                 # KNN模型训练脚本
├── train_svm_ga.py              # 遗传算法优化SVM训练脚本
├── train_svm_pso.py             # 粒子群算法优化SVM训练脚本
├── utils.py                     # 通用工具函数库
└── README.md                    # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- 所需库请参考以下安装命令：

```bash
pip install scikit-learn pandas numpy matplotlib scipy xgboost joblib scikit-opt
```

### 数据准备

1. 将数据集文件（默认为 `pcr_400.csv`）放置在 `data/` 目录下
2. 确保数据集中包含目标变量列 "Pcr"

### 运行模型训练

每个模型都可以独立运行。例如，要训练决策树模型：

```bash
python train_dt.py
```

要训练智能优化算法优化的SVM模型：

```bash
python train_svm_ga.py    # 遗传算法优化SVM
python train_svm_pso.py   # 粒子群算法优化SVM
```

要训练所有模型，可以创建一个批处理脚本或依次运行：

```bash
python train_dt.py
python train_svm.py
python train_gp.py
python train_gb.py
python train_xgb.py
python train_mlp.py
python train_rf.py
python train_knn.py
python train_svm_ga.py
python train_svm_pso.py
```

## 🛠️ 功能特性

### 1. 数据预处理
- 自动检测并移除常数特征
- 缺失值处理（均值填充）
- 特征标准化（MinMaxScaler）
- 数据分层抽样（可选）

### 2. 模型训练与调优
- 使用GridSearchCV进行超参数优化（传统模型）
- 使用遗传算法（GA）和粒子群算法（PSO）进行智能优化
- 5折交叉验证
- 自动保存最优模型

### 3. 模型评估
- 训练集评估指标：RMSE、MAE、MAPE、R²
- 测试集评估指标：RMSE、MAE、MAPE、R²
- 相对误差分析

### 4. 学习曲线分析
- 自动生成学习曲线
- 数据充分性评估
- 模型收敛性分析

### 5. 结果保存
- 保存最佳模型（joblib格式）
- 保存预测结果（CSV格式）
- 保存学习曲线数据（JSON格式）

## 📊 支持的模型

| 模型 | 文件 | 主要特性 | 优化方法 |
|------|------|----------|----------|
| 决策树 (DT) | `train_dt.py` | 可解释性强，参数调优灵活 | GridSearchCV |
| 支持向量机 (SVM) | `train_svm.py` | 适用于高维数据，核函数选择 | GridSearchCV |
| 高斯过程 (GP) | `train_gp.py` | 提供预测不确定性估计 | GridSearchCV |
| 梯度提升 (GB) | `train_gb.py` | 集成学习，抗过拟合 | GridSearchCV |
| XGBoost (XGB) | `train_xgb.py` | 高效梯度提升实现 | GridSearchCV |
| 多层感知机 (MLP) | `train_mlp.py` | 神经网络，复杂模式学习 | GridSearchCV |
| 随机森林 (RF) | `train_rf.py` | 集成学习，特征重要性分析 | GridSearchCV |
| K近邻 (KNN) | `train_knn.py` | 简单有效，无需训练 | GridSearchCV |
| GA-SVM | `train_svm_ga.py` | 遗传算法优化SVM参数 | 遗传算法 (GA) |
| PSO-SVM | `train_svm_pso.py` | 粒子群算法优化SVM参数 | 粒子群算法 (PSO) |

## 🧬 智能优化算法介绍

### 1. 遗传算法优化SVM (GA-SVM)
**文件：** `train_svm_ga.py`

**特点：**
- 使用遗传算法（Genetic Algorithm）优化SVM的C和epsilon参数
- 模拟自然选择和遗传机制进行参数搜索
- 适用于复杂的非线性优化问题
- 具有全局搜索能力，避免陷入局部最优

**算法参数：**
- 种群大小：30
- 最大迭代次数：50
- 变异概率：0.001
- 参数范围：C∈[0.1, 100], epsilon∈[0.01, 1.0]

**优点：**
- 并行搜索多个解
- 适用于离散和连续参数空间
- 鲁棒性强

### 2. 粒子群算法优化SVM (PSO-SVM)
**文件：** `train_svm_pso.py`

**特点：**
- 使用粒子群算法（Particle Swarm Optimization）优化SVM参数
- 模拟鸟群觅食行为的群体智能算法
- 收敛速度快，计算效率高
- 适用于连续参数优化

**算法参数：**
- 粒子数量：30
- 最大迭代次数：50
- 惯性权重：0.8
- 学习因子：c1=0.5, c2=0.5
- 参数范围：C∈[0.1, 100], epsilon∈[0.01, 1.0]

**优点：**
- 收敛速度快
- 参数调节简单
- 内存需求小
- 适合实时优化

## 🔧 工具函数 (utils.py)

### 主要功能：
- `load_data()` - 加载数据集
- `prepare_data()` - 数据预处理
- `split_data()` - 数据分割
- `train_and_evaluate_model()` - 模型训练与评估
- `test_model()` - 模型测试
- `plot_learning_curve()` - 学习曲线分析
- `analyze_data_sufficiency_all_models()` - 多模型数据充分性分析
- `cross_validate_model()` - 交叉验证
- `diagnose_data_issues()` - 数据质量诊断

## 📈 输出文件

每个模型训练后会生成以下文件：

```
save_model/[MODEL_NAME]/
├── [model_name]_best_model_01.joblib      # 最优模型
├── full_pipeline.joblib                   # 数据预处理管道
├── [model_name]_results.csv               # 测试结果
├── [model_name]_learning_curve_data.json  # 学习曲线数据
└── [model_name]_learning_curve.png        # 学习曲线图（如果生成）
```

智能优化算法模型还会输出：
- 优化过程的最佳参数值
- 优化算法的收敛信息

## ⚙️ 配置说明

### 修改目标变量
默认目标变量为 "Pcr"。如需修改，请在各个训练脚本的 `__main__` 部分中更改 `target_column` 变量。

### 调整优化算法参数
对于GA-SVM和PSO-SVM模型，可以在对应脚本中调整：
- 种群/粒子数量
- 最大迭代次数
- 参数搜索范围
- 其他算法特定参数

### 更改数据路径
在 `utils.py` 的 `load_data()` 函数中修改 `data_path` 参数。

## 🎯 智能优化算法使用建议

### 何时使用智能优化算法？
1. **参数空间复杂**：当参数之间存在复杂的非线性关系时
2. **传统方法收敛慢**：当GridSearchCV需要探索大量参数组合时
3. **全局最优需求**：需要找到全局最优解而不仅仅是局部最优
4. **实时优化**：PSO算法特别适合需要快速收敛的场景

### 性能对比建议
建议同时运行以下模型进行性能对比：
- 传统GridSearchCV优化的SVM (`train_svm.py`)
- 遗传算法优化的SVM (`train_svm_ga.py`)
- 粒子群算法优化的SVM (`train_svm_pso.py`)

这样可以比较不同优化方法的效果和效率。

## 📝 注意事项

1. **数据集要求**：确保数据集为CSV格式，且包含目标变量列
2. **内存使用**：某些模型（如SVM、GP）在大数据集上可能需要较多内存
3. **运行时间**：智能优化算法可能需要较长时间，建议在性能较好的机器上运行
4. **结果重现**：智能优化算法具有一定的随机性，多次运行结果可能略有不同
5. **依赖库**：智能优化算法需要安装 `scikit-opt` 库

## 🔍 故障排除

### 常见问题：
1. **文件未找到错误**：确保数据文件路径正确
2. **内存不足**：减小数据集规模或调整算法参数
3. **导入错误**：确保所有依赖库已正确安装，特别是 `scikit-opt`
4. **优化算法不收敛**：调整算法参数（如增加迭代次数、增大种群规模）

### 调试建议：
- 启用 `diagnose_data_issues()` 函数检查数据质量
- 调整交叉验证折数减少运行时间
- 简化超参数搜索空间加速优化过程
- 对于智能优化算法，可以先使用较小种群规模和迭代次数进行测试

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 📧 联系

如有问题或建议，请通过GitHub Issues联系我们。

---

**提示**：
1. 首次运行前，请确保已安装所有依赖库，并正确配置数据集路径
2. 智能优化算法模型需要额外的 `scikit-opt` 库，请使用 `pip install scikit-opt` 安装
3. 建议先运行传统模型了解基线性能，再尝试智能优化算法模型进行对比