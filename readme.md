# Machine Learning Project: Regression and Classification Tasks

This project demonstrates fundamental machine learning concepts through two distinct tasks: a regression problem using Support Vector Regression (SVR) and a classification problem using K-Nearest Neighbors (KNN).

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Regression Task: Crab Age Prediction](#regression-task-crab-age-prediction)
    *   [Data](#data)
    *   [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
    *   [Model Training & Evaluation](#model-training--evaluation)
    *   [Explainability](#explainability)
    *   [Other Regression Models Discussed](#other-regression-models-discussed)
3.  [Classification Task: Crab Age Classification](#classification-task-crab-age-classification)
    *   [Data](#data-1)
    *   [Preprocessing & Feature Scaling](#preprocessing--feature-scaling)
    *   [Model Training & Evaluation](#model-training--evaluation-1)
    *   [Optimizing K for KNN](#optimizing-k-for-knn)
4.  [Blogs](#blogs)
    *   [Reinforcement Learning](#reinforcement-learning)
    *   [Oversampling and Undersampling](#oversampling-and-undersampling)


## Project Overview
This project provides a hands-on experience with supervised machine learning techniques. It covers data loading, preprocessing, exploratory data analysis (EDA), model training, prediction, evaluation, and an introduction to model explainability. Two datasets related to crab characteristics are used to predict crab age.

## Regression Task: Crab Age Prediction
**Objective:** Predict the age of a crab based on its physical measurements using a Support Vector Regressor (SVR) model.

### Data
*   **Dataset:** `Task1.csv`
*   **Target Variable:** 'Age'

### Preprocessing & Feature Engineering
1.  **Column Dropping:** The 'id' column was removed.
2.  **Handling Outliers/Invalid Data:** Rows where 'Height' was 0 were dropped.
3.  **Feature Creation:** A new feature, 'LostWeight', was engineered. It indicates whether the total weight (`Weight`) was less than the sum of internal weights (`Shucked Weight` + `Viscera Weight` + `Shell Weight`).
4.  **Categorical Encoding:** The 'Sex' column (I, M, F) was converted into numerical format using one-hot encoding.

### Model Training & Evaluation
1.  **Data Split:** The preprocessed data was split into training and testing sets (70% train, 30% test) with `random_state=42`.
2.  **Model:** Support Vector Regressor (SVR) was used for prediction.
3.  **Prediction:** Predictions were made on the test set.
4.  **Evaluation Metrics:**
    *   **R-squared (R2):** 0.550 (Indicates that 55% of the variance in crab age is explained by the model).
    *   **Mean Absolute Error (MAE):** 1.389
    *   **Mean Squared Error (MSE):** 4.425
    *   **Root Mean Squared Error (RMSE):** 2.104

### Explainability
*   `SHAP (SHapley Additive exPlanations)` was used to interpret individual SVR predictions. A `force_plot` was generated to visualize the contribution of each feature to a specific prediction, helping understand why the model made a particular output.

### Other Regression Models Discussed
*   **XGBoost Regressor:** An ensemble boosting method known for high performance and handling complex relationships, suitable for large datasets.
*   **Random Forest Regressor:** An ensemble method that builds multiple decision trees and averages their predictions, good for reducing overfitting and feature importance.
*   **Ridge Regressor:** A linear regression model with L2 regularization to prevent overfitting and handle multicollinearity by shrinking coefficients.

## Classification Task: Crab Age Classification
**Objective:** Classify the age group of a crab based on its physical measurements using the K-Nearest Neighbors (KNN) algorithm.

### Data
*   **Dataset:** `Task2.csv`
*   **Target Variable:** 'Age' (treated as a multi-class classification target).

### Preprocessing & Feature Scaling
1.  **Categorical Encoding:** The 'Sex' column was one-hot encoded.
2.  **Feature Scaling:** All numerical features (excluding the 'Age' target) were standardized using `StandardScaler` to ensure all features contribute equally to the distance calculations in KNN.

### Model Training & Evaluation
1.  **Data Split:** The scaled data was split into training and testing sets (70% train, 30% test) with `random_state=42`.
2.  **Model:** K-Nearest Neighbors (KNN) classifier was used.
3.  **Initial Performance (k=1):**
    *   An initial KNN model with `n_neighbors=1` yielded an accuracy of **0.230**.
    *   A classification report detailing precision, recall, and f1-score for each age class was generated.

### Optimizing K for KNN
1.  **K Value Iteration:** The model was trained and evaluated for `k` values ranging from 1 to 60.
2.  **Accuracy Plot:** A plot showing 'Accuracy vs. K Value' was generated to visualize the relationship and identify the optimal `k`.
3.  **Optimal K:** The optimal `k` value was determined to be **59**, which achieved the highest accuracy of **0.318**.
4.  **Final Evaluation:** The KNN model was retrained with `n_neighbors=59`, and its classification report was provided, demonstrating the improved performance with the optimal `k`.

## Blogs
### Reinforcement Learning
*(See detailed blog post in the notebook for full content)*

**Summary:** Reinforcement Learning (RL) is a paradigm where an agent learns to make optimal decisions by interacting with an environment, receiving rewards or penalties. It involves an agent, environment, states, actions, rewards, and a policy. RL is powerful for complex problems where explicit programming is difficult, such as robotics, game playing (e.g., Chess, Go), autonomous driving, and resource management.

### Oversampling and Undersampling
*(See detailed blog post in the notebook for full content)*

**Summary:** These techniques address imbalanced datasets in machine learning, where one class significantly outnumbers others. 
*   **Oversampling** increases the minority class instances (e.g., using SMOTE) to provide the model with more learning opportunities, but can lead to overfitting. 
*   **Undersampling** reduces the majority class instances to balance the dataset, which can speed up training but risks losing valuable information. The choice between them or a combination depends on the dataset size and the degree of imbalance.
