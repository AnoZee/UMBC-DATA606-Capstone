# Predictive Analysis of Heart Disease Using Machine Learning

## 1. Title and Author

- **Prepared for:** UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang
- **Author Name:** Suhas Parray
- **Link to the author's GitHub profile:** [[GitHub Profile Link](https://github.com/AnoZee/)]
- **Link to the author's LinkedIn profile:** [[LinkedIn Profile Link](https://linkedin.com/in/suhasparray)]

## 2. Background

Heart disease remains one of the leading causes of mortality worldwide, necessitating early detection and prevention strategies. This project aims to leverage the Heart Attack Analysis & Prediction Dataset to develop a predictive model for heart disease. By analyzing demographic and clinical parameters, such as age, sex, chest pain type, and serum cholesterol, among others, we seek to identify significant predictors of heart disease. This analysis is crucial for understanding risk factors and improving patient outcomes through early intervention.

### Research Questions:
- What are the primary factors contributing to heart disease?
- Can we predict the presence of heart disease with reasonable accuracy using machine learning models?

## 3. Data

The dataset chosen for this analysis provides a comprehensive overview of factors potentially contributing to heart diseases. It includes both categorical and numerical data, making it suitable for developing predictive models.

- **Data sources:** [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)
- **Data size:** 33 KB
- **Data shape:** 303 rows x 14 columns
- **Time period:** Not time-bound
- **Each row represents:** An individual's health profile

### Data Dictionary:
- **Age:** Age of the individual (Numerical)
- **Sex:** Sex of the individual (Categorical: Male, Female)
- **Chest Pain Type (cp):** Type of chest pain (Categorical: typical angina, atypical angina, non-anginal pain, asymptomatic)
- **Resting Blood Pressure (trtbps):** Resting blood pressure (Numerical)
- **Serum Cholesterol (chol):** Serum cholesterol in mg/dl (Numerical)
- **Fasting Blood Sugar (fbs):** Fasting blood sugar > 120 mg/dl (Categorical: True, False)
- **Resting Electrocardiographic Results (restecg):** Resting electrocardiographic results (Categorical)
- **Maximum Heart Rate Achieved (thalachh):** Maximum heart rate (Numerical)
- **Exercise Induced Angina (exang):** Exercise-induced angina (Categorical: Yes, No)
- **ST Depression Induced by Exercise Relative to Rest (oldpeak):** ST depression (Numerical)
- **The Slope of The Peak Exercise ST Segment (slp):** Slope of the peak exercise ST segment (Categorical)
- **Number of Major Vessels Colored by Flourosopy (caa):** Number of major vessels (Numerical)
- **Thallium Stress Test (thall):** Thallium stress test result (Categorical)
- **Output (Diagnosis of Heart Disease):** Diagnosis of heart disease (Categorical: 0, 1)

### Target/Label:
- **Output (Diagnosis of Heart Disease)**

### Features/Predictors:
- All other variables except the output will be considered for feature selection to develop the ML models.

## Analysis Plan

We will conduct a comprehensive exploratory data analysis (EDA) to understand the distribution and relationships among the features. This will include correlation analysis to identify potential interactions and dependencies. Cluster analysis will be employed to uncover patterns or subgroups within the data. Finally, we will develop predictive models to assess the accuracy of heart disease diagnosis based on the dataset features.

This project aims to contribute to the early detection and prevention of heart disease, leveraging data science and machine learning methodologies to improve healthcare outcomes.
