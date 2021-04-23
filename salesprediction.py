# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:57:54 2021

@author: bhasker
"""
import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
import numpy as np

st.title("Predicting Sales based on youtube marketing")

st.sidebar.header("user inputs")

def user_inputs():
    youtube_budget=st.sidebar.number_input("Enter the youtubebudget")
    youtubesq=youtube_budget*youtube_budget
    inputdata={"youtube":youtube_budget,"youtube_Sq":youtubesq}
    data=pd.DataFrame(inputdata,index=[0])
    return data

inputs=user_inputs()

st.write(inputs)

#build model
train=pd.read_csv("marketing.csv")
train["youtube_Sq"]=train.youtube*train.youtube

model=smf.ols('np.log(sales)~youtube+youtube_Sq',data=train).fit()

prediction=np.exp(model.predict(inputs))

#print output

st.write("Estmated sales for the given youtube budget")

from scipy import stats

ci= stats.norm.interval(0.95,prediction)
st.write(ci[0],ci[1])