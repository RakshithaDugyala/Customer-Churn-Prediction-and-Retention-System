Overview

This project implements a customer churn prediction and retention system using Python and machine learning techniques. It analyzes customer data from a telecommunications dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) to predict churn probability, segment customers, and recommend retention strategies based on risk levels and customer segments.

Features





Data Preprocessing: Loads and cleans the Telco Customer Churn dataset, handling missing values and converting data types.



Encoding: Converts categorical variables to numerical format using one-hot encoding and maps the target variable (Churn) to binary values.



Model Training: Trains two machine learning models (Logistic Regression and Random Forest) to predict customer churn.



Churn Risk Scoring: Uses the Random Forest model to compute churn probabilities and categorizes customers into Low, Medium, and High risk levels.



Customer Segmentation: Applies KMeans clustering to segment customers based on tenure, monthly charges, and total charges.



Retention Strategy: Recommends tailored retention actions based on churn risk and customer segment.



Visualization: Generates a plot to visualize the distribution of churn risk levels.

Requirements

To run the notebook, install the following Python libraries:





pandas



numpy



matplotlib



seaborn



scikit-learn

You can install them using pip:

pip install pandas numpy matplotlib seaborn scikit-learn

Dataset

The dataset used is WA_Fn-UseC_-Telco-Customer-Churn.csv, which contains customer information such as:





Demographic details



Service subscriptions



Billing information



Churn status (Yes/No)

The dataset must be placed in the same directory as the notebook or the file path updated in the code.

Usage





Setup: Ensure all required libraries are installed and the dataset is available.



Run the Notebook: Open Customer_Churn_Prediction_and_Retention_System.ipynb in Jupyter Notebook or a compatible environment (e.g., Google Colab) and execute the cells sequentially.



Output: The notebook will:





Display classification reports and confusion matrices for Logistic Regression and Random Forest models.



Generate a churn risk distribution plot.



Add columns to the dataset for churn probability, risk level, customer segment, and recommended retention actions.

Code Structure

The notebook is organized into the following sections:





Imports: Imports necessary libraries for data processing, modeling, and visualization.



Data Loading and Cleaning: Loads the dataset, removes unnecessary columns (e.g., customerID), and handles missing values.



Encoding: Encodes categorical variables and the target variable (Churn).



Train-Test Split: Splits the data into training and testing sets.



Model Training: Trains and evaluates Logistic Regression and Random Forest models.



Churn Risk Scoring: Predicts churn probabilities and assigns risk levels.



Customer Segmentation: Performs KMeans clustering for customer segmentation.



Retention Strategy: Assigns retention actions based on risk and segment.



Visualization: Plots the churn risk distribution.

Results





Model Performance: Both models are evaluated using precision, recall, F1-score, and confusion matrices. Random Forest is used for final predictions due to its robustness.



Churn Risk: Customers are classified into Low (<0.4), Medium (0.4–0.7), and High (>0.7) churn risk categories.



Segmentation: Customers are grouped into 4 segments based on tenure, monthly charges, and total charges.



Retention Actions: Recommendations include loyalty benefits, discount plans, retention agent assignment, or personalized offers based on risk and segment.

Future Improvements





Experiment with additional models (e.g., XGBoost, Neural Networks) for better performance.



Incorporate feature importance analysis to identify key drivers of churn.



Enhance retention strategies with more granular recommendations.



Add more visualizations for deeper insights into segments and churn patterns.
