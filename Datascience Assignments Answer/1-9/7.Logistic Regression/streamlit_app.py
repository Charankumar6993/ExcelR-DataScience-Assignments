
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model(path='titanic_logreg_pipeline.joblib'):
    return joblib.load(path)

model = load_model()
st.title('Titanic Survival Predictor (Logistic Regression)')
pclass = st.selectbox('Pclass (1 = 1st, 2 = 2nd, 3 = 3rd)', options=[1,2,3], index=1)
sex = st.selectbox('Sex', options=['male','female'])
age = st.number_input('Age (years)', min_value=0.0, max_value=120.0, value=30.0, step=0.5)
sibsp = st.number_input('Siblings / Spouse aboard (SibSp)', min_value=0, max_value=10, value=0, step=1)
parch = st.number_input('Parents / Children aboard (Parch)', min_value=0, max_value=10, value=0, step=1)
fare = st.number_input('Fare', min_value=0.0, max_value=1000.0, value=32.0, step=0.1)
embarked = st.selectbox('Port of Embarkation', options=['S','C','Q'])
if st.button('Predict Survival Probability'):
    X_new = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }])
    proba = model.predict_proba(X_new)[0,1]
    pred = model.predict(X_new)[0]
    st.write(f'Predicted probability of survival: **{proba:.3f}**')
    st.write('Predicted class:', '**Survived**' if pred == 1 else '**Did not survive**')
