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

- **Data sources:** Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.
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

## 4. Exploratory Data Analysis (EDA)

### Missing Values
The table below shows the number of missing values in each column:
<table>
<tr>
<td>

| Column   | Missing Values |
|----------|----------------|
| age      | 0              |
| sex      | 0              |
| cp       | 0              |
| trtbps   | 0              |
| chol     | 0              |
| fbs      | 0              |
| restecg  | 0              |

</td>
<td>

| Column   | Missing Values |
|----------|----------------|
| thalachh | 0              |
| exng     | 0              |
| oldpeak  | 0              |
| slp      | 0              |
| caa      | 0              |
| thall    | 0              |
| output   | 0              |

</td>
</tr>
</table>

### Unique Values
![Unique Values](/docs/report-images/1.png)

### Data Cleaning and Transformation

- **Number of duplicate rows:** 1
- **Action Taken:** Duplicate rows have been removed. The dataset now has 302 rows and 14 columns.

**Data Cleaning:**
- The value 0 in the 'thall' column was replaced with 2. This ensures consistency in the dataset.

**Data Transformation:**
- Numerical values in various columns were mapped to categorical descriptions for better interpretability.

### Summary Statistics
- Produced summary statistics of key variables.

### Univariate Analysis
![Univariate Analysis](/docs/report-images/2.png)

### Bivariate Analysis
![Bivariate Analysis](/docs/report-images/3.png)

### Correlation Analysis
![Correlation Analysis](/docs/report-images/4.png)

**Correlation with Target (Heart Attack)**
![Correlation Analysis With Target](/docs/report-images/5.png)

### Clustering Analysis (Dimensionality Reduction - t-SNE Visualization)
![t-SNE Visualization](/docs/report-images/6.png)

## 5. Predictive Analysis

- In this section, we explore the construction of a predictive model for heart disease using various machine learning algorithms. We aim to compare the effectiveness of different models such as Logistic Regression, Support Vector Machines (SVM), Decision Trees, Random Forests, Gradient Boosting, K-Nearest Neighbors (KNN), Naive Bayes, and XGBoost.

- The dataset employed contains comprehensive patient data including demographics, blood test results, and heart examination outcomes. Our objective is to predict the presence of heart disease based on these features.

### Data Preprocessing

**Categorization of Features**
- `Categorical Columns`: Variables that represent categories (e.g., sex, chest pain type).
- `Numerical Columns`: Variables that are measured on a numeric scale (e.g., age, cholesterol levels).

### Data Splitting
- Separate the dataset into features (X) and the target variable (y), and then split these into training and testing sets to validate the models effectively.

### Model Training
- Draft

### Model Evaluation
- Draft
![t-SNE Visualization](/docs/report-images/7.png)




## 5. Model Training

### Models Used
- RandomForest (as chosen earlier)
- Other potential models: Logistic Regression, Support Vector Machines, K-Nearest Neighbors

### Training Methodology
- Train-test split: 80/20
- Python packages used: scikit-learn
- Development environment: Jupyter Notebook

### Performance Measurement
- Measured model performance using accuracy, precision, recall, F1-score, and ROC-AUC.

## 6. Application of the Trained Models

### Web App Development
- Developed a web app using **Streamlit** for user interaction with the trained models.
- Streamlit was chosen for its simplicity and ease of learning.

## 7. Conclusion

### Summary
- Successfully developed a predictive model for heart disease using machine learning.
- Identified significant predictors and achieved reasonable accuracy in predictions.

### Limitations
- Limited dataset size.
- Potential biases in the dataset.

### Lessons Learned
- Importance of data preprocessing and feature selection.
- Challenges in model selection and tuning.

### Future Research Directions
- Incorporate larger and more diverse datasets.
- Explore additional machine learning models and techniques.
- Integrate real-time data for continuous model improvement.

## 8. References

- Janosi, Andras, Steinbrunn, William, Pfisterer, Matthias, and Detrano, Robert. (1988). Heart Disease. UCI Machine Learning Repository. [https://doi.org/10.24432/C52P4X](https://doi.org/10.24432/C52P4X)
