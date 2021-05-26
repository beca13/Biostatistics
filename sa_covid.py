import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb\
# Lifeline package for the Survival Analysis
from lifelines.plotting import plot_lifetimes      
plt.figure(figsize = (12,6))

##  create a dataframe
df = pd.read_csv(r"C:\Users\brank\OneDrive\BIOSTATISTIKA\datasets\COVID19_line_list_data.csv")

info=df.info()
print(info)

df.drop(['case_in_country','Unnamed: 3','summary','symptom_onset','If_onset_approximated','hosp_visit_date','exposure_start','exposure_end','symptom','source','link'], axis = 1 , inplace=True) 
df.dropna(axis=0, inplace=True)

# update death values to 1 instead of date of death (few records)
df.loc[(df['death'] != "0") & (df['death'] != "1"), 'death'] = 1
df['death'] = df['death'].astype(int)

## Have a first look at the data
head=df.head()
print(head)

## Data Types and Missing Values in Columns(if any)
info=df.info()
print(info)

##KaplanMeier curve, without breaking it in a groups of covariates.
## Import the library """
from lifelines import KaplanMeierFitter

durations = df['age'] ## Time to event data of censored and event data
event_observed = df['death']  ## It has the churned (1) and censored is (0)

## create a kmf object as km
km = KaplanMeierFitter() 

## Fit the data into the model
km.fit(durations, event_observed,label='Kaplan Meier Estimate')

## Create an estimate
km.plot()
plt.legend(loc='upper right')
plt.xlabel('Godine')
plt.ylabel('Verovatnoća preživljavanja')
plt.title("Kaplan Majerova verovatnoća preživljavanja COVID-19 u zavisnosti od godina")
plt.show()

##Lets create Kaplan Meier Curves for Cohorts¶
kmf1 = KaplanMeierFitter() 

T = df['age']     ## time to event
E = df['death']      ## event occurred or censored

## Two Cohorts are compared. 1. male, 2. female
groups = df['gender']   
i1 = (groups == 'male')      ## group i1 , having the pandas series for the 1st cohort
i2 = (groups == 'female')     ## group i2 , having the pandas series for the 2nd cohort


## fit the model for 1st cohort
kmf1.fit(T[i1], E[i1], label='male')
a1 = kmf1.plot()

## fit the model for 2nd cohort
kmf1.fit(T[i2], E[i2], label='female')
kmf1.plot(ax=a1)

plt.legend(loc='upper right')
plt.xlabel('Godine')
plt.ylabel('Verovatnoća preživljavanja')
plt.title("Kaplan Majerova verovatnoća preživljavanja COVID-19 u zavisnosti od godina i pola")
plt.show()
 
##Cox Proportional Hazard Model (Survival Regression)
from lifelines import CoxPHFitter
# Implementation of the model.Thus taking subset of the columns to train the model.
df_r= df.loc[:,['age','death','gender','visiting Wuhan','from Wuhan']] #gender
df_r.head() ## have a look at the data

## Create dummy variables
df_dummy = pd.get_dummies(df_r, drop_first=True)
df_dummy.head()

# Using Cox Proportional Hazards model
cph = CoxPHFitter()   
cph.fit(df_dummy, 'age', event_col='death')  
cph.print_summary()    ## HAve a look at the significance of the features
cph.plot()

## Check all the methods and attributes associated with the cph object.
dir(cph)

## We want to see the Survival curve at the patient level. Therefore, we have selected 4 patient (rows 46 till 49).
tr_rows = df_dummy.iloc[46:49, 0:]
tr_rows

## Lets predict the survival curve 
cph.predict_survival_function(tr_rows).plot()
plt.show()