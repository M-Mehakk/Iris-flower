import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#title
st.title("Iris Flower prediction app")
#load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target
#train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
#model
model = RandomForestClassifier()
model.fit(X_train, Y_train)
#siderbar inputs
st.sidebar.header("Enter flower measurements")
sepal_length = st.sidebar.slider("Sepal length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()))
sepal_width = st.sidebar.slider("Sepal width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()))
petal_length = st.sidebar.slider("petal length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()))
petal_width= st.sidebar.slider("petal width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()))
#predict
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_class = iris.target_names[prediction][0]

st.write("### prediction flower type: ", predicted_class)
