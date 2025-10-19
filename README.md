# Road Accident Risk Prediction

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S5E10-blue)](https://www.kaggle.com/competitions/playground-series-s5e10)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-lightblue)](https://www.kaggle.com/code/shreyashpatil217/road-accident-risk-prediction)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project presents an advanced machine learning solution for predicting road accident risk using ensemble regression methods. The model predicts continuous accident risk scores (0.0 to 1.0) for road conditions based on various environmental and traffic parameters.

**Competition:** Kaggle Playground Series - Season 5, Episode 10  
**Task Type:** Regression (Continuous Value Prediction)  
**Evaluation Metric:** RMSE (Root Mean Squared Error)  
**Kaggle Notebook:** [View Notebook](https://www.kaggle.com/code/shreyashpatil217/road-accident-risk-prediction)

## Results

| Metric | Score |
|--------|-------|
| **Best Individual Model** | RMSE: 0.0562 (Gradient Boosting) |
| **Ensemble Performance** | RMSE: ~0.0561 |
| **Test Predictions** | 172,585 samples |
| **Leaderboard Score** | 0.05540 |
| **Leaderboard Ranking** | Position 1751 |

## Dataset

- **Training Samples:** 172,585 road records
- **Test Samples:** 172,585 roads to predict
- **Target Variable:** Accident risk (continuous 0.0-1.0)
- **Features:** 25+ variables including road conditions, traffic, and weather
- **Source:** [Kaggle Playground Series S5E10](https://www.kaggle.com/competitions/playground-series-s5e10)

### Key Features

**Top Features by Importance:**

| Feature | Importance | Impact |
|---------|-----------|--------|
| Lighting | 53.2% | Most critical predictor |
| Speed Limit | 28.5% | Secondary predictor |
| Road Curvature | 9.2% | Road geometry factor |
| Weather | 6.7% | Environmental condition |
| Number of Reported Accidents | 2.1% | Historical data |

**Categorical Features:**
- Road Type
- Lighting Conditions
- Weather Type
- Public Road
- Holiday Status

**Numerical Features:**
- Speed Limit
- Road Curvature
- Number of Lanes
- Number of Reported Accidents

## Methodology

### 1. Data Preprocessing

```python
# Comprehensive preprocessing pipeline
- Missing value imputation (median/mode)
- Outlier detection and removal (IQR method)
- Categorical variable encoding (LabelEncoder)
- Feature scaling (RobustScaler for outlier resistance)
```

**Steps:**
- Handle missing values using statistical methods
- Detect and clip outliers using IQR (Q1 - 1.5*IQR to Q3 + 1.5*IQR)
- Encode categorical variables with LabelEncoder
- Apply RobustScaler for normalization (resistant to outliers)

### 2. Advanced Feature Engineering

**Polynomial Features:**
```python
for col in top_correlated_features:
    X[f'{col}_squared'] = X[col] ** 2
    X[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
    X[f'{col}_log'] = np.log1p(np.abs(X[col]))
```

**Interaction Features:**
```python
X[f'{feature1}_x_{feature2}'] = X[feature1] * X[feature2]
X[f'{feature1}_div_{feature2}'] = X[feature1] / (X[feature2] + 1e-8)
```

**Feature Selection:**
- SelectKBest with f_regression
- Ranked by correlation with target
- Selected top 20-30 features

### 3. Model Training

Four regression models with optimized hyperparameters:

| Model | Estimators | Max Depth | Learning Rate | RMSE |
|-------|-----------|-----------|---------------|------|
| **XGBoost** | 200 | 7 | 0.08 | 0.0562 |
| **LightGBM** | 200 | 40 | 0.08 | 0.0562 |
| **Random Forest** | 200 | 15 | - | 0.0562 |
| **Gradient Boosting** | 200 | 7 | 0.08 | **0.0562** |

**Hyperparameter Optimization:**
- n_estimators: 100-200 (increased for better accuracy)
- max_depth: 5-15 (regularization parameter)
- learning_rate: 0.08-0.1 (control learning speed)
- subsample: 0.9 (reduce overfitting)
- colsample_bytree: 0.9 (feature sampling)

### 4. Ensemble Strategy

**Weighted Regression Ensemble:**
- Weights based on inverse RMSE (better models get higher weight)
- Probability-based averaging
- Final prediction = weighted average of all 4 models

**Model Weights:**
- XGBoost: ~25%
- LightGBM: ~25%
- Random Forest: ~25%
- Gradient Boosting: ~25%

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
xgboost>=1.5.0
lightgbm>=3.2.0
catboost>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## Usage

### Quick Start

```bash
# Clone the repository
git clone https://github.com/ShreyashPatil530/road-accident-risk-prediction.git
cd road-accident-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python road_accident_prediction.py

# Or use Jupyter notebook
jupyter notebook road_accident_prediction.ipynb
```

### Step-by-Step Pipeline

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

# 1. Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Preprocess
# - Handle missing values
# - Remove outliers (IQR)
# - Encode categorical variables
# - Scale features

# 3. Feature Engineering
# - Polynomial transformations
# - Interaction features
# - Feature selection (top 20-30)

# 4. Train Models
# - XGBoost Regressor
# - LightGBM Regressor
# - Random Forest Regressor
# - Gradient Boosting Regressor

# 5. Ensemble & Predict
# - Weighted voting regressor
# - Generate predictions for test set

# 6. Create Submission
# - Format: id, accident_risk
# - Save to submission.csv
```

## Project Structure

```
road-accident-risk-prediction/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── road_accident_prediction.py  # Main Python script
├── road_accident_prediction.ipynb  # Jupyter Notebook
│
├── data/
│   ├── train.csv               # Training data (172,585 rows)
│   ├── test.csv                # Test data (172,585 rows)
│   └── submission.csv          # Final predictions
│
├── models/
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   ├── rf_model.pkl
│   └── gb_model.pkl
│
├── notebooks/
│   ├── eda.ipynb               # Exploratory Data Analysis
│   ├── feature_engineering.ipynb
│   └── model_comparison.ipynb
│
└── output/
    ├── feature_importance.csv
    ├── model_metrics.csv
    └── predictions.csv
```

## Key Insights

### Feature Importance Analysis

1. **Lighting (53.2%)** - Most critical factor
   - Dramatically impacts accident risk prediction
   - Should be prioritized in road safety measures

2. **Speed Limit (28.5%)** - Strong secondary factor
   - Controls traffic speed-related accidents
   - Important for prediction accuracy

3. **Road Curvature (9.2%)** - Geometric factor
   - Curved roads have different risk profiles
   - Important for intersection analysis

4. **Weather Conditions (6.7%)** - Environmental impact
   - Affects road traction and visibility
   - Seasonal variations important

5. **Historical Accidents (2.1%)** - Contextual information
   - High-risk areas identifiable
   - Useful for targeted interventions

### Model Performance Comparison

```
Validation RMSE Scores:
┌─────────────────────┬──────────┐
│ Model               │ RMSE     │
├─────────────────────┼──────────┤
│ XGBoost             │ 0.056217 │
│ LightGBM            │ 0.056250 │
│ Random Forest       │ 0.056225 │
│ Gradient Boosting   │ 0.056216 │
│ Ensemble (Average)  │ 0.056225 │
└─────────────────────┴──────────┘
```

All models achieved similar performance, validating the ensemble approach.

## Improvements & Optimizations

### Phase 1: Initial Baseline
- Basic preprocessing and single model
- Limited feature engineering
- RMSE: ~0.058

### Phase 2: Feature Engineering
- Added polynomial features (squared, sqrt, log)
- Created interaction features
- RMSE: ~0.057

### Phase 3: Model Optimization
- Increased estimators from 100 to 200
- Fine-tuned hyperparameters
- Added subsample and colsample parameters
- RMSE: ~0.0562

### Phase 4: Ensemble Implementation
- Weighted voting based on performance
- Combined 4 diverse models
- Final RMSE: 0.05540

## Potential Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV/Optuna
- [ ] Stacking with meta-learner (Ridge/Lasso)
- [ ] Advanced feature engineering (domain-specific)
- [ ] Cross-validation (5-fold, 10-fold)
- [ ] Feature interaction detection (SHAP)
- [ ] Neural network ensemble
- [ ] AutoML frameworks (AutoGluon, H2O)
- [ ] Outlier-specific handling
- [ ] Time-series analysis if temporal data available
- [ ] Regional/seasonal feature engineering

## Performance Metrics

### Regression Metrics Used

**RMSE (Root Mean Squared Error):**
```
RMSE = sqrt(mean((y_true - y_pred)^2))
```
- Penalizes large errors heavily
- Primary metric for this competition

**MAE (Mean Absolute Error):**
```
MAE = mean(|y_true - y_pred|)
```
- Average prediction error
- More interpretable than RMSE

**R² Score:**
```
R² = 1 - (SS_res / SS_tot)
```
- Proportion of variance explained
- Ranges from 0 to 1 (higher is better)

## Author

**Shreyash Patil**

- **Email:** [shreyashpatil530@gmail.com](mailto:shreyashpatil530@gmail.com)
- **Kaggle:** [Shreyash Patil](https://www.kaggle.com/shreyashpatil217)
- **GitHub:** [ShreyashPatil530](https://github.com/ShreyashPatil530)
- **Portfolio:** [Shreyash Patil Portfolio](https://shreyash-patil-portfolio1.netlify.app/)

## Competition Details

- **Platform:** Kaggle
- **Competition:** Playground Series - Season 5, Episode 10
- **Problem Type:** Supervised Regression
- **Evaluation Metric:** RMSE
- **Status:** Completed with Submission

## References

- [Kaggle Playground S5E10](https://www.kaggle.com/competitions/playground-series-s5e10)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Ensemble Methods](https://en.wikipedia.org/wiki/Ensemble_learning)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is for educational and research purposes. The predictions are based on training data and ensemble methods. Actual accident risk prediction should incorporate additional real-world factors and expert validation.

## Citation

If you use this project in your research or work, please cite:

```bibtex
@project{road_accident_prediction_2025,
  title={Road Accident Risk Prediction: Ensemble Machine Learning Approach},
  author={Patil, Shreyash},
  year={2025},
  url={https://github.com/ShreyashPatil530/road-accident-risk-prediction}
}
```

## Acknowledgments

- Kaggle for hosting the competition and providing datasets
- Open source community for amazing ML libraries (scikit-learn, XGBoost, LightGBM)
- Contributors and reviewers for feedback and suggestions

---

**Last Updated:** 2025  
**Status:** Active  
**Version:** 1.0.0  
**Leaderboard Score:** 0.05540

## Connect & Collaborate

- Open an issue on GitHub for bug reports or feature requests
- Connect on Kaggle for discussions
- Email for collaboration inquiries: shreyashpatil530@gmail.com

---

**Happy Learning and Coding!**
