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
![t-SNE Visualization](/docs/report-images/71.png)

### Model Comparison
- After training and evaluating each model, we plot their training and testing accuracies to compare their performance visually. This helps in identifying the models that perform well consistently and those that might be overfitting or underfitting.
![Model Comparison](/docs/report-images/8.png)

### Evaluation Results
- **Logistic Regression**: Achieved a training accuracy of 86% and a testing accuracy of 85%, indicating good generalization without overfitting.
- **Support Vector Classifier (SVC)**: Showed a high training accuracy of 90% but a lower testing accuracy of 82%, suggesting potential overfitting.
- **Decision Tree Classifier**: Exhibited perfect training accuracy of 100% but a testing accuracy of only 80%, clearly overfitting the training data.
- **Random Forest Classifier**: Also reached 100% training accuracy with a testing accuracy of 87%, demonstrating better generalization than the Decision Tree.
- **Gradient Boosting Classifier**: Similar pattern to Random Forest, with perfect training and 85% testing accuracy, indicating some overfitting.
- **K-Nearest Neighbors (KNN)**: Maintained a consistent performance with 85% training and 87% testing accuracy, showing excellent generalization.
- **Gaussian Naive Bayes**: Achieved 84% training and 85% testing accuracy, performing robustly on both datasets.
- **XGBoost Classifier**: Attained perfect training accuracy and a high testing accuracy of 87%, though the perfect training score raises concerns about overfitting.

### Top Performers
- **Random Forest**, **K-Nearest Neighbors**, and **XGBoost** displayed the highest testing accuracies at 87%, making them the most effective models for this dataset. However, Random Forest and XGBoost showed potential signs of overfitting due to their perfect training scores.

## 6. Recommendations for Future Work

### Model Optimization
- **Hyperparameter Tuning**: Further tuning of model parameters using grid search or random search could potentially improve performance and reduce overfitting.
- **Cross-Validation**: Implementing k-fold cross-validation may provide a more reliable estimate of model performance and generalizability.

### Practical Considerations
- **Model Interpretability**: In practical applications, especially in healthcare, understanding the decision-making process of the model is crucial. Models like Logistic Regression and Decision Trees, despite slightly lower performance metrics, offer greater transparency in their predictions.
- **Ensemble Techniques**: Combining the predictions of several models could enhance prediction accuracy and reliability, leveraging the strengths of individual models while mitigating their weaknesses.

While the RandomForestClassifier, KNeighborsClassifier, and XGBoostClassifier have shown promising results, it is crucial to balance model accuracy with interpretability and robustness to overfitting. Employing comprehensive validation techniques and considering the operational context will ensure that the deployed models are both effective and practical.

## 7. Summary

- This section encapsulates the major steps and findings of the project, highlighting key analytical techniques and model performance insights.

### Exploratory Data Analysis (EDA)

- **Purpose**: Understanding the basic statistical properties of the data and the distribution of various features and the target variable.
- **Findings**: The EDA stage was crucial for identifying trends, anomalies, patterns, and relationships within the data. This phase helped in preparing the data for more advanced analyses and informed the subsequent preprocessing steps.

### Correlation Analysis

- **Technique**: Employed a heatmap to visualize the correlation matrix.
- **Insights**: The correlation analysis was instrumental in identifying which features had significant positive or negative associations with each other and with the target variable 'output'. This helped in feature selection and provided a basis for the predictive modeling.

### Cluster Analysis

- **Application**: Used to group similar instances, potentially uncovering intrinsic structures within the data.
- **Utility**: Particularly valuable in exploratory stages or when labels are absent. It helped in identifying distinct groups or patterns in the data, which could be crucial for segmenting data or understanding different risk profiles.

### Machine Learning Prediction

- **Models Evaluated**: Eight different models were applied, including RandomForestClassifier, KNeighborsClassifier, and XGBoost.
- **Performance Outcome**: The RandomForestClassifier, KNeighborsClassifier, and XGBoost emerged as the top performers with the highest testing accuracy. These models demonstrated a strong capability to generalize well on unseen data.
- **Model Selection**: The choice of these models was justified by their robustness and accuracy in predicting heart disease, although considerations regarding overfitting and model complexity were also addressed.

### Conclusions

The journey through data exploration, analysis, and predictive modeling provided deep insights into the factors influencing heart disease. The selected machine learning models showed promising results, but it is essential to balance accuracy with model interpretability and robustness. Future work could explore more advanced model tuning, cross-validation techniques, and the integration of more complex models to enhance predictive performance further.

This summary provides a holistic view of the project, emphasizing the analytical process and insights gained. It serves as a comprehensive overview for stakeholders or anyone interested in understanding the critical components and outcomes of the project.

## 8. References

- Janosi, Andras, Steinbrunn, William, Pfisterer, Matthias, and Detrano, Robert. (1988). Heart Disease. UCI Machine Learning Repository. [https://doi.org/10.24432/C52P4X](https://doi.org/10.24432/C52P4X)
