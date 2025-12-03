# Machine Learning Regression Model Training Project

A comprehensive Python project for training and evaluating various machine learning regression models, including Decision Tree, SVM, Gaussian Process, Gradient Boosting, XGBoost, MLP, Random Forest, KNN, and intelligently optimized SVM models.

## 📁 Project Structure

```
├── data/
│   └── pcr_400.csv              # Dataset
├── save_model/                  # Model saving directory (generated after execution)
│   ├── DT/                      # Decision Tree model files
│   ├── SVM/                     # SVM model files
│   ├── GP/                      # Gaussian Process model files
│   ├── GB/                      # Gradient Boosting model files
│   ├── XGB/                     # XGBoost model files
│   ├── MLP/                     # MLP neural network model files
│   ├── RF/                      # Random Forest model files
│   ├── KNN/                     # KNN model files
│   ├── GA_SVM/                  # Genetic Algorithm optimized SVM model files
│   └── PSO_SVM/                 # Particle Swarm Optimization optimized SVM model files
├── train_dt.py                  # Decision Tree model training script
├── train_svm.py                 # SVM model training script
├── train_gp.py                  # Gaussian Process model training script
├── train_gb.py                  # Gradient Boosting model training script
├── train_xgb.py                 # XGBoost model training script
├── train_mlp.py                 # MLP model training script
├── train_rf.py                  # Random Forest model training script
├── train_knn.py                 # KNN model training script
├── train_svm_ga.py              # Genetic Algorithm optimized SVM training script
├── train_svm_pso.py             # Particle Swarm Optimization optimized SVM training script
├── utils.py                     # Common utility functions library
└── README.md                    # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Required libraries:

```bash
pip install scikit-learn pandas numpy matplotlib scipy xgboost joblib scikit-opt
```

### Data Preparation

1. Place the dataset file (default: `pcr_400.csv`) in the `data/` directory
2. Ensure the dataset contains the target variable column "Pcr"

### Running Model Training

Each model can be run independently. For example, to train the Decision Tree model:

```bash
python train_dt.py
```

To train intelligently optimized SVM models:

```bash
python train_svm_ga.py    # Genetic Algorithm optimized SVM
python train_svm_pso.py   # Particle Swarm Optimization optimized SVM
```

To train all models, create a batch script or run sequentially:

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

## 🛠️ Features

### 1. Data Preprocessing
- Automatic detection and removal of constant features
- Missing value handling (mean imputation)
- Feature standardization (MinMaxScaler)
- Stratified data sampling (Select as needed)

### 2. Model Training and Tuning
- Hyperparameter optimization using GridSearchCV (traditional models)
- Intelligent optimization using Genetic Algorithm (GA) and Particle Swarm Optimization (PSO)
- 5-fold cross-validation
- Automatic saving of best models

### 3. Model Evaluation
- Training set metrics: RMSE, MAE, MAPE, R²
- Test set metrics: RMSE, MAE, MAPE, R²
- Relative error analysis

### 4. Learning Curve Analysis
- Automatic generation of learning curves
- Data sufficiency assessment
- Model convergence analysis

### 5. Result Saving
- Save best models (joblib format)
- Save prediction results (CSV format)
- Save learning curve data (JSON format)

## 📊 Supported Models

| Model | File | Key Features | Optimization Method |
|-------|------|--------------|---------------------|
| Decision Tree (DT) | `train_dt.py` | Highly interpretable, flexible parameter tuning | GridSearchCV |
| Support Vector Machine (SVM) | `train_svm.py` | Suitable for high-dimensional data, kernel selection | GridSearchCV |
| Gaussian Process (GP) | `train_gp.py` | Provides prediction uncertainty estimates | GridSearchCV |
| Gradient Boosting (GB) | `train_gb.py` | Ensemble learning, resistant to overfitting | GridSearchCV |
| XGBoost (XGB) | `train_xgb.py` | Efficient gradient boosting implementation | GridSearchCV |
| Multilayer Perceptron (MLP) | `train_mlp.py` | Neural network, complex pattern learning | GridSearchCV |
| Random Forest (RF) | `train_rf.py` | Ensemble learning, feature importance analysis | GridSearchCV |
| K-Nearest Neighbors (KNN) | `train_knn.py` | Simple and effective, no training required | GridSearchCV |
| GA-SVM | `train_svm_ga.py` | Genetic Algorithm optimized SVM parameters | Genetic Algorithm (GA) |
| PSO-SVM | `train_svm_pso.py` | Particle Swarm Optimization optimized SVM parameters | Particle Swarm Optimization (PSO) |

## 🧬 Intelligent Optimization Algorithms

### 1. Genetic Algorithm Optimized SVM (GA-SVM)
**File:** `train_svm_ga.py`

**Features:**
- Uses Genetic Algorithm (GA) to optimize SVM's C and epsilon parameters
- Simulates natural selection and genetic mechanisms for parameter search
- Suitable for complex nonlinear optimization problems
- Global search capability avoids local optima

**Algorithm Parameters:**
- Population size: 30
- Maximum iterations: 50
- Mutation probability: 0.001
- Parameter ranges: C∈[0.1, 100], epsilon∈[0.01, 1.0]

**Advantages:**
- Parallel search of multiple solutions
- Suitable for discrete and continuous parameter spaces
- Strong robustness

### 2. Particle Swarm Optimization Optimized SVM (PSO-SVM)
**File:** `train_svm_pso.py`

**Features:**
- Uses Particle Swarm Optimization (PSO) to optimize SVM parameters
- Simulates swarm intelligence inspired by bird flocking behavior
- Fast convergence, high computational efficiency
- Suitable for continuous parameter optimization

**Algorithm Parameters:**
- Particle count: 30
- Maximum iterations: 50
- Inertia weight: 0.8
- Learning factors: c1=0.5, c2=0.5
- Parameter ranges: C∈[0.1, 100], epsilon∈[0.01, 1.0]

**Advantages:**
- Fast convergence speed
- Simple parameter adjustment
- Low memory requirements
- Suitable for real-time optimization

## 🔧 Utility Functions (utils.py)

### Main Functions:
- `load_data()` - Load dataset
- `prepare_data()` - Data preprocessing
- `split_data()` - Data splitting
- `train_and_evaluate_model()` - Model training and evaluation
- `test_model()` - Model testing
- `plot_learning_curve()` - Learning curve analysis
- `analyze_data_sufficiency_all_models()` - Multi-model data sufficiency analysis
- `cross_validate_model()` - Cross-validation
- `diagnose_data_issues()` - Data quality diagnosis

## 📈 Output Files

Each model training generates the following files:

```
save_model/[MODEL_NAME]/
├── [model_name]_best_model_01.joblib      # Best model
├── full_pipeline.joblib                   # Data preprocessing pipeline
├── [model_name]_results.csv               # Test results
├── [model_name]_learning_curve_data.json  # Learning curve data
└── [model_name]_learning_curve.png        # Learning curve plot (if generated)
```

Intelligent optimization models additionally output:
- Best parameter values from optimization process
- Convergence information of optimization algorithms

## ⚙️ Configuration

### Modifying Target Variable
Default target variable is "Pcr". To modify, change the `target_column` variable in the `__main__` section of each training script.

### Adjusting Optimization Algorithm Parameters
For GA-SVM and PSO-SVM models, you can adjust:
- Population/particle count
- Maximum iterations
- Parameter search ranges
- Other algorithm-specific parameters

### Changing Data Path
Modify the `data_path` parameter in the `load_data()` function in `utils.py`.

## 🎯 Intelligent Optimization Usage Recommendations

### When to Use Intelligent Optimization Algorithms?
1. **Complex parameter space**: When parameters have complex nonlinear relationships
2. **Slow convergence of traditional methods**: When GridSearchCV requires exploring many parameter combinations
3. **Need for global optimum**: When finding global optimum is more important than local optimum
4. **Real-time optimization**: PSO algorithm is particularly suitable for scenarios requiring fast convergence

### Performance Comparison Recommendation
It's recommended to run the following models for performance comparison:
- Traditional GridSearchCV optimized SVM (`train_svm.py`)
- Genetic Algorithm optimized SVM (`train_svm_ga.py`)
- Particle Swarm Optimization optimized SVM (`train_svm_pso.py`)

This allows comparison of effectiveness and efficiency among different optimization methods.

## 📝 Important Notes

1. **Dataset requirements**: Ensure dataset is in CSV format and contains the target variable column
2. **Memory usage**: Some models (like SVM, GP) may require significant memory with large datasets
3. **Execution time**: Intelligent optimization algorithms may require considerable time; recommended to run on capable machines
4. **Result reproducibility**: Intelligent optimization algorithms have some randomness; results may vary slightly between runs
5. **Dependencies**: Intelligent optimization algorithms require `scikit-opt` library

## 🔍 Troubleshooting

### Common Issues:
1. **File not found error**: Ensure correct data file path
2. **Insufficient memory**: Reduce dataset size or adjust algorithm parameters
3. **Import errors**: Ensure all dependencies are correctly installed, especially `scikit-opt`
4. **Optimization algorithm not converging**: Adjust algorithm parameters (increase iterations, increase population size)

### Debugging Recommendations:
- Enable `diagnose_data_issues()` function to check data quality
- Reduce cross-validation folds to decrease execution time
- Simplify hyperparameter search space to accelerate optimization
- For intelligent optimization algorithms, start with smaller population size and iterations for testing

## 📄 License

This project is for learning and research purposes only.

## 🤝 Contribution

Issues and Pull Requests are welcome to improve this project.

## 📧 Contact

For questions or suggestions, please contact us through GitHub Issues.

---

**Tip**:
1. Before first run, ensure all dependencies are installed and data path is correctly configured
2. Intelligent optimization models require additional `scikit-opt` library, install with `pip install scikit-opt`
3. It's recommended to run traditional models first to understand baseline performance, then try intelligent optimization models for comparison