import pickle
import streamlit as st
import pandas as pd
import xgboost
import numpy as np


loaded_model = xgboost.Booster()
loaded_model.load_model('xgb_model.bin')


# page title
st.title('Heart Attack Prediction using ML')

st.markdown(
    """
    <style>
    .reportview-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .main .block-container {
        flex: 1;
        max-width: 800px;
        padding-top: 5rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div style="display: flex; justify-content: center;"><img src="https://media.tenor.com/91scJf-xrKEAAAAi/emoji-coraz%C3%B3n-humano.gif" width="200"></div>', 
    unsafe_allow_html=True,
)


age = st.number_input('Enter age', min_value=18, max_value=100, step=1, value=68, help="Enter the patient's age in years. Valid range from 18 to 100.")
sex = st.radio('Sex', options=['Male', 'Female'], horizontal=True)
sex = 1 if sex == 'Male' else 0

cp = st.radio('Chest Pain Type', options=[('0: typical angina', 0),
                                          ('1: atypical angina', 1),
                                          ('2: non-anginal pain', 2),
                                          ('3: asymptomatic', 3)], horizontal=True, format_func=lambda x: x[0])[1]

trtbps = st.number_input('Enter resting blood pressure value', min_value=50, max_value=200, step=1, value=120)
chol = st.number_input('Enter cholesterol value (mg/dl)', min_value=50, max_value=500, step=1, value=200)
fbs = st.radio('Is fasting blood sugar > 120 mg/dl?', options=['Yes', 'No'], horizontal=True)
fbs = 1 if fbs == 'Yes' else 0

restecg = st.radio('Resting Electrocardiographic Results', options=[
    ('0: normal', 0),
    ('1: ST-T wave abnormality', 1),
    ('2: left ventricular hypertrophy', 2)], horizontal=True, format_func=lambda x: x[0])[1]

thalachh = st.number_input("Enter person's maximum heart rate achieved", min_value=60, max_value=220, step=1, value=150)

exng = st.radio('Exercise induced angina', options=['Yes', 'No'], horizontal=True)
exng = 1 if exng == 'Yes' else 0

oldpeak = st.number_input('Enter oldpeak (ST depression caused by activity compared to rest)', step=0.1, value=2.0)

slp = st.radio('Slope of the peak exercise ST segment', options=[
    ('0: downsloping', 0),
    ('1: flat', 1),
    ('2: upsloping', 2)], horizontal=True, format_func=lambda x: x[0])[1]

caa = st.radio('Coronary artery anomaly (number of major vessels 0-3)', options=[0, 1, 2, 3], horizontal=True)
thall = st.radio('Thalassemia', options=[0, 1, 2, 3], horizontal=True)

# features_values = {'age': age, 'trtbps': trtbps, 'chol': chol, 'thalachh': thalachh, 'oldpeak': oldpeak}


# # age = st.number_input('Enter age',step=1, value=45)
# # Age input with validation
# age = st.number_input('Enter age', min_value=18, max_value=100, step=1, value=45, help="Enter the patient's age in years. Valid range from 18 to 100.")

# # sex = st.selectbox('Enter sex', ('Male', 'Female'))
# sex = st.radio('Sex', options=['Male', 'Female'], horizontal=True)
# sex = 1 if sex == 'Male' else 0

# st.write("Chest Pain type \n\n Value 0: typical angina \n\n Value 1: atypical angina \n\n Value 2: non-anginal pain \n\n Value 3: asymptomatic trtbps : resting blood pressure (in mm Hg)")
# #cp = st.number_input('Enter Chest Pain type',step=1)
# cp = st.selectbox('Enter Chest Pain type', (0,1,2,3), index=2)
# #cp = 1 if cp == '1' else 0

# trtbps = st.number_input('Enter resting blood pressure value',step=1, value=80)

# chol = st.number_input('Enter cholestoral value(cholestoral in mg/dl fetched via BMI sensor)',step=1, value=200)

# fbs = st.selectbox('Is fasting blood sugar > 120 mg/dl', ('Yes', 'No'))
# fbs = 1 if fbs == 'Yes' else 0
# #fbs = st.number_input('Enter fbs value((fasting blood sugar > 120 mg/dl) (1 = true; 0 = false))',step=1)

# st.write("Resting Electrocardiographic Results \n\nValue 0: normal \n\n Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) \n\nValue 2: showing probable or definite left ventricular hypertrophy by Estes' criteria thalach : maximum heart rate achieved)")
# restecg = st.selectbox('Enter Resting Electrocardiographic Results value', (0,1, 2), index=2)
# #restecg = st.number_input("Enter restecg value(resting electrocardiographic results: Value 0: normal Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria thalach : maximum heart rate achieved)",step=1)

# thalachh = st.number_input("The person's maximum heart rate achieved",step=1, value=110)

# exng=st.selectbox('Enter exercise induced angina value', ('Yes', 'No'))
# exng = 1 if exng == 'Yes' else 0

# oldpeak = st.number_input('Enter oldpeak(ST depression caused by activity compared to rest) value', value=4)

# st.write("the slope of the peak exercise ST segment — \n\n 0: downsloping; \n\n 1: flat; \n\n 2: upsloping")
# slp = st.selectbox('Enter slope of the peak exercise ST segment value',(0,1,2), index=1)

# caa = st.selectbox('Enter caa(coronary artery anomaly) value(number of major vessels (0-3))',(0,1,2,3), index=2)

# thall = st.selectbox('Enter thalassemia value',(0,1,2,3), index=1)


features_values={'age':age,'trtbps':trtbps,'chol':chol,'thalachh':thalachh,'oldpeak':oldpeak}

if st.button('Predict'):
    if any(value == 0 or value == 0.00 for value in features_values.values()):
        st.warning('Please input all the details.')
    else:
        data_1 = pd.DataFrame({'thall': [thall],
                'caa': [caa],
                'cp': [cp],
                'oldpeak': [oldpeak],
                'exng': [exng],
                'chol': [chol],
                'thalachh': [thalachh]
            })
        print(data_1)
        dtest = xgboost.DMatrix(data_1)

        prediction = loaded_model.predict(dtest)
        threshold = 0.5
        prediction = np.where(prediction >= threshold, 1, 0)
        #st.write(prediction)
        if prediction == 0:
            #st.write('Patient has no risk of Heart Attack')
            st.markdown("<h2 style='text-align: center; color: green;'>No indication of a Heart Disease.</h2>", unsafe_allow_html=True)
        else:
            #st.write('Patient has risk of Heart Attack')
            st.markdown("<h2 style='text-align: center; color: red;'>RISK OF HEART DISEASE !!</h2>", unsafe_allow_html=True)
