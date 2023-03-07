import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0, 1000, 500)
    Radio = st.sidebar.slider('Radio', 0, 1000, 500)
    Newspaper = st.sidebar.slider('Newspaper', 0, 1000, 500)
    Sales = st.sidebar.slider('Sales', 0, 1000, 500)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,
            'Sales': Sales}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
