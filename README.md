# Amazon Bestsellers Dataset Analysis & Machine Learning

This project contains a comprehensive analysis and machine learning implementation using the Amazon Bestsellers dataset. The project explores data preprocessing, feature engineering, and multiple regression techniques to predict user ratings of bestselling books.

## 📊 Dataset Overview

The dataset contains information about Amazon bestselling books from 2009-2019 with the following features:
- **Name**: Book title
- **Author**: Book author
- **User Rating**: Average user rating (target variable)
- **Reviews**: Number of user reviews
- **Price**: Book price
- **Year**: Publication year
- **Genre**: Fiction or Non-Fiction

## 🔍 Project Structure

```
├── 29-bestsellers_with_categories.csv    # Original dataset
├── filtered_dataset.csv                  # Processed dataset
├── linear_regression copy 3.ipynb        # Main analysis notebook
├── data_analysis.ipynb                   # Exploratory data analysis
├── linear_regression.ipynb               # Linear regression implementation
├── logistic_regression.ipynb             # Logistic regression analysis
├── RandomForest.ipynb                    # Random Forest model
├── SVC.ipynb                             # Support Vector Classifier
├── SVR.ipynb                             # Support Vector Regression
└── README.md                             # Project documentation
```

## 🛠️ Data Preprocessing & Feature Engineering

### Data Cleaning
1. **Missing Data Analysis**: Checked for null values across all columns
2. **Zero Value Removal**: Removed rows where numeric columns contained zero values
3. **Duplicate Handling**: Grouped books by Name and Author, aggregating:
   - User Rating (mean)
   - Reviews (maximum)
   - Price (mean)
   - Year (latest)
   - Genre (first occurrence)

### Outlier Detection & Removal
- Applied **Interquartile Range (IQR)** method for outlier detection
- Focused on 'Reviews' column for outlier removal
- Used boxplot visualizations to identify and handle outliers

### Feature Engineering
1. **Target Encoding**: Applied target encoding to 'Author' column using User Rating as target
2. **One-Hot Encoding**: Applied to categorical variables (Genre)
3. **Feature Scaling**: Implemented StandardScaler for SVR models
4. **Dummy Variable Trap**: Avoided by dropping one category from one-hot encoded variables

## 📈 Exploratory Data Analysis

### Correlation Analysis
- Generated correlation heatmap for numeric features
- Analyzed relationships between features and target variable
- Created scatter plots to visualize feature relationships

### Statistical Analysis
- Used **Ordinary Least Squares (OLS)** regression with statsmodels
- Calculated **Variance Inflation Factors (VIF)** to detect multicollinearity
- Generated comprehensive regression summary statistics

## 🤖 Machine Learning Models

### 1. Linear Regression
- **Implementation**: Scikit-learn LinearRegression
- **Performance Metrics**:
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Squared Log Error
  - Median Absolute Error
  - Explained Variance Score

### 2. Support Vector Regression (SVR)

#### Basic SVR Implementation
- Used default RBF kernel
- Compared performance with Linear Regression

#### Optimized SVR with Hyperparameter Tuning
- **GridSearchCV**: Exhaustive parameter search
  - Kernels: RBF, Polynomial, Sigmoid
  - C values: [0.1, 1, 10, 100]
  - Gamma: ['scale', 'auto', 0.001, 0.01, 0.1, 1]
  - Epsilon: [0.01, 0.1, 0.2, 0.5]

- **RandomizedSearchCV**: Efficient parameter optimization
  - 50 parameter combinations tested
  - Faster alternative to GridSearchCV

#### Custom SVR Configuration
- Optimized parameters: C=1, gamma=0.1, epsilon=0.01, kernel='rbf'

## 📊 Model Evaluation & Validation

### Cross-Validation
- **5-Fold Cross-Validation** for model reliability
- **Repeated K-Fold Cross-Validation** (5 splits, 10 repeats)
- Comparison of cross-validation scores across different models

### Performance Visualization
- **Actual vs Predicted** scatter plots
- **Residual plots** for error analysis
- **Boxplots** for outlier visualization
- **Correlation heatmaps** for feature relationships

### Model Comparison
Performance comparison between:
1. Default SVR
2. GridSearchCV optimized SVR  
3. RandomizedSearchCV optimized SVR
4. Linear Regression baseline

## 🔧 Technical Implementation

### Libraries Used
```python
# Data Manipulation & Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Feature Engineering
import category_encoders as ce

# Statistical Analysis
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

### Key Features
- **Robust Data Pipeline**: Comprehensive preprocessing and cleaning
- **Multiple Model Comparison**: Systematic evaluation of different algorithms
- **Hyperparameter Optimization**: Both grid search and random search implementations
- **Statistical Validation**: Cross-validation and statistical significance testing
- **Visualization**: Comprehensive plotting for insights and model evaluation

## 📋 Results Summary

The project demonstrates:
1. Effective data preprocessing techniques for real-world datasets
2. Comparison of linear and non-linear regression approaches
3. Importance of hyperparameter tuning for model optimization
4. Statistical validation of machine learning models
5. Comprehensive evaluation metrics for regression problems

## 🚀 Getting Started

1. **Clone the repository**
2. **Install required packages**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn category-encoders statsmodels scipy
   ```
3. **Run the notebooks** in the following order:
   - `data_analysis.ipynb` - Exploratory analysis
   - `linear_regression copy 3.ipynb` - Main analysis
   - Other model-specific notebooks as needed

## 📝 Future Enhancements

- Implementation of ensemble methods (Random Forest, Gradient Boosting)
- Deep learning approaches for comparison
- Feature importance analysis
- Time series analysis considering publication years
- Advanced text analysis of book titles and authors

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements or additional analysis techniques.

---

*This project demonstrates end-to-end machine learning workflow from data preprocessing to model evaluation, providing insights into Amazon bestseller patterns and user rating prediction.*
