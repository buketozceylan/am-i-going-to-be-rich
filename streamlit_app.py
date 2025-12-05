import streamlit as st
from model import class_model

st.title('Am I going to be rich? Or Am I already rich??')

st.write("This app is for funny purposes only. This has no meaning and won't have any meaning in the future. ")

st.dataframe(class_model.X_train)

st.metric(label="Accuracy", value=class_model.acc_score)