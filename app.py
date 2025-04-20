# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 10:44:18 2025

@author: LAB
"""

import streamlit as st
import numpy as np
import pickle

#load modal
with open('dtm_trained_model.pkl','rb') as f:
    dtm_modal = pickle.load(f)
    
st.title("Iris flower Classification")
st.write("Enter the feature of the iris flower:")

sepal_length = st.slider("Sepal Length (cm)",4.0,8.0,5.1)
sepal_width = st.slider("Sepal Width (cm)",2.0,4.5,3.5)
petal_length = st.slider("Petal Length (cm)",1.0,7.0,1.4)
petal_width = st.slider("Petal Width (cm)",0.1,2.5,0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction = dtm_modal.predict(input_data)
    speicies = ['Setosa','Versicolor','Virginica']
    st.success(f"The Predicted Species is:**{speicies[prediction[0]]}**")