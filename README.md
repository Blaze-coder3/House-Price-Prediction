# House Price Prediction

## Overview

This repository contains code for predicting house prices using machine learning techniques. The project involves data preprocessing, exploratory data analysis, model training, and evaluation. Two primary models are used: Linear Regression and Random Forest Regressor.

## Table of Contents

1. [Dependencies](#dependencies)
2. [Data Exploration](#data-exploration)
3. [Data Visualization](#data-visualization)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [File Structure](#file-structure)
8. [Contributors](#Contributors)


---

## Dependencies <a name="dependencies"></a>

- **Python Libraries:**
  - `numpy`: For numerical operations.
  - `pandas`: For data manipulation and analysis.
  - `matplotlib`: For plotting graphs and visualizations.
  - `seaborn`: For statistical data visualization.
  - `sklearn`: For machine learning modeling and evaluation.

---

## Data Exploration <a name="data-exploration"></a>

- **Function**: `explore_data(data)`
  - Displays the first few rows and summary statistics of the dataset.
  
---

## Data Visualization <a name="data-visualization"></a>

- **Function**: `visualize_correlation(data)`
  - Generates a heatmap to visualize the correlation between numeric features in the dataset.

---

## Data Preprocessing <a name="data-preprocessing"></a>

- **Function**: `preprocess_data(data)`
  - Preprocesses the dataset by encoding categorical variables and scaling numeric features.
  
---

## Model Training <a name="model-training"></a>

### Linear Regression

- **Function**: `train_linear_regression(X_train, y_train)`
  - Trains a Linear Regression model on the preprocessed dataset.
  
### Random Forest Regressor

- **Function**: `train_random_forest(X_train, y_train)`
  - Trains a Random Forest Regressor model using GridSearchCV for hyperparameter tuning.

## Best_model.pkl
- **WARNING:** `the file will be extracted automatically after model training`


---

## Model Evaluation <a name="model-evaluation"></a>

### Linear Regression

- **Function**: `train_linear_regression(X_train, y_train, X_test, y_test)`
  - Trains a Linear Regression model and evaluates its performance on both training and test datasets.
  
### Random Forest Regressor

- **Model Evaluation Metrics:**
  - R-squared (R2)
  - Mean Squared Error (MSE)

---

## File Structure <a name="file-structure"></a>

```
House-Price-Prediction/
│
├── README.md
│
├── static/
│   └── styles.css
│
├── templates/
│   └── index.html
    └── result.html
│
│── app.py
│── best_model.pkl
│── housing.csv
│── predict(Final).ipynb
└──
```

---

## Conclusion

This documentation provides a comprehensive overview of the House Price Prediction project, detailing data processing steps, model training techniques, and evaluation metrics. Users can refer to the respective sections for a deeper understanding of the project workflow and implementation details.

## Usage
You can run the Jupyter Notebook here or clone the repository and execute the code locally.

## Contributors
A Syed Khwaja
