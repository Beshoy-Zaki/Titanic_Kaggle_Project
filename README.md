# Project: Titanic - Machine Learning from Disaster

## Overview

This repository, `titanic-kaggle`, contains a complete end-to-end implementation of a machine learning pipeline for predicting survival on the Titanic using the famous Kaggle dataset. It includes data cleaning, exploratory data analysis, feature engineering, model training, evaluation, and submission generation using a **Random Forest Classifier**.

---

## Contents

- `Titanic_Kaggle.ipynb`: Main Jupyter notebook with all steps.
- `train.csv`, `test.csv`: Datasets used (from Kaggle).
- `titanic_randomforest_submission.csv`: Example output submission file.

---

## Workflow & Methodology

### 1. 📥 Load Dataset

- Read the `train.csv` and `test.csv` files using `pandas`.
- Inspect dataset with `.info()`, `.describe()`, and check for null values.

### 2. 🧹 Data Cleaning

- Fill missing values:
  - `Age` with median
  - `Embarked` and `Cabin` with mode
- Remove outliers from `Age` and `Fare` using the IQR method.
- Drop irrelevant columns: `Name`, `Ticket`, `Cabin`
- Check for and drop duplicates.

### 3. 📊 Exploratory Data Analysis (EDA)

- Use `seaborn` boxplots to visualize distributions and detect outliers.
- Understand relationships between features and the target variable `Survived`.

### 4. 🛠️ Feature Engineering

- Separate features (`X`) and target (`y`)
- Identify numerical and categorical columns
- Build preprocessing pipelines:
  - Numerical: `SimpleImputer` + `StandardScaler`
  - Categorical: `SimpleImputer` + `OneHotEncoder`
- Combine using `ColumnTransformer`

### 5. 🧠 Model Training

- Use `RandomForestClassifier` with a pipeline.
- Train the model using `train_test_split` (e.g., 80/20).
- Evaluate using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### 6. 🏁 Final Training & Kaggle Submission

- Retrain on the full dataset.
- Predict on the `test.csv` dataset.
- Generate a `titanic_randomforest_submission.csv` file.

```python
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})
submission.to_csv('titanic_randomforest_submission.csv', index=False)
```
## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/Beshoy-Zaki/Titanic_Kaggle_Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Titanic_Kaggle_Project
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Titanic_Kaggle.ipynb
   ```
4. Run the cells to execute the code and understand the implementations.


## Author

Beshoy-Zaki
