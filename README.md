# Financial Fraud Detection System

## 1. Project Overview

This project presents a comprehensive machine learning solution for detecting fraudulent financial transactions. Using a large-scale, real-world dataset, this system is designed to accurately identify and flag suspicious activities from millions of transactions, addressing the critical challenge of severe class imbalance. The final model is a tuned **XGBoost Classifier** trained with an oversampling technique to maximize fraud detection while minimizing false positives.

---

## 2. Dataset

The project utilizes a synthetic dataset generated using PaySim, which mimics real-world mobile money transactions.

* **Source:** The dataset: [Fraud Detection](here.)
* **Size:** 6,362,620 transactions
* **Features:** 10 original features including transaction type, amount, and account balances.
* **Imbalance:** The dataset is highly imbalanced, with only **8,213 (0.13%)** of transactions being fraudulent.

---

## 3. Methodology

The project followed a structured machine learning workflow:

1.  **Data Cleaning & EDA:** The data was explored to understand its structure. No missing values were found.
2.  **Feature Engineering:** New features (`errorBalanceOrig`, `errorBalanceDest`) were created to capture inconsistencies in account balances, which proved to be strong indicators of fraud.
3.  **Modeling Strategy:**
    * A **Logistic Regression** model was trained as a baseline.
    * An **XGBoost** model was selected as the primary, high-performance model.
4.  **Handling Class Imbalance:** Several techniques were systematically evaluated:
    * Using the `scale_pos_weight` parameter in XGBoost.
    * Training with oversampled data using **Random Oversampling (ROS)**, **SMOTE**, and **ADASYN**.
5.  **Hyperparameter Tuning:** `RandomizedSearchCV` was used to efficiently find the optimal hyperparameters for each model, using the **Area Under the Precision-Recall Curve (AUPRC)** as the primary evaluation metric.

---

## 4. Key Findings & Final Model Performance

The **XGBoost model trained with Random Oversampling (ROS)** was selected as the final, best-performing model due to its superior ability to correctly identify fraudulent transactions on the unseen test data.

The final model achieved the following performance:

* **AUPRC Score:** **0.9516**
* **ROC AUC Score:** **0.9945**

Using an optimal classification threshold of **0.8915**, the model produced the following business-critical results:

* ✅ **Recall (Fraud): 85%** - Successfully identified **83 out of 98** fraudulent transactions, preventing most financial losses.
* ✅ **Precision (Fraud): 94%** - When the model flagged a transaction as fraud, it was correct **94%** of the time, ensuring high trust and a low rate of false alarms for legitimate customers.

---

## 5. How to Run This Project

To replicate this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/fraud-detection-project.git](https://github.com/your-username/fraud-detection-project.git)
    cd fraud-detection-project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  Open the main project notebook (`.ipynb` file) and run the cells.

---

## 6. Tools and Libraries Used

* **Python 3.x**
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Scikit-learn:** For data preprocessing, modeling, and evaluation.
* **Imbalanced-learn (imblearn):** For handling class imbalance with ROS, SMOTE, and ADASYN.
* **XGBoost:** For the high-performance gradient boosting model.
* **Matplotlib & Seaborn:** For data visualization.
* **Jupyter Notebook:** As the main development environment.
