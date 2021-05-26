import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
# Lifeline package for the Survival Analysis
from lifelines.plotting import plot_lifetimes      
plt.figure(figsize = (12,6))

##  create a dataframe
df = pd.read_csv(r"C:\Users\brank\OneDrive\BIOSTATISTIKA\datasets\COVID19_line_list_data_TimeToRecover.csv")

df.drop(['case_in_country','summary','If_onset_approximated'], axis = 1 , inplace=True) 
df['symptom_onset'] = pd.to_datetime(df['symptom_onset'])
df['recoveredDate'] = pd.to_datetime(df['recoveredDate'])
df['DaysRecovery'] =  df['recoveredDate']- df['symptom_onset']
df['DaysRecovery'] = df["DaysRecovery"].dt.days

## Have a first look at the data
head=df.head()
print(head)

## Data Types and Missing Values in Columns(if any)
info=df.info()
print(info)

## Import the library 
from lifelines import KaplanMeierFitter

durations = df['DaysRecovery'] ## Time to event data of censored and event data
event_observed = df['recovered']  ## It has the churned (1) and censored is (0)

## create a kmf object as km
km = KaplanMeierFitter() ## instantiate the class to create an object

## Fit the data into the model
km.fit(durations, event_observed,label='Kaplan Meier Estimate')

## Create an estimate
km.plot()
plt.legend(loc='upper right')
plt.xlabel('Dani')
plt.ylabel('Verovatnoća oporavka')
plt.title("Kaplan Majerova verovatnoća oporavka od COVID-19 virusa po danima")

plt.figure()
inverse = 1 - km.survival_function_
plt.step(inverse.index.to_numpy().reshape(47,1), inverse.T.to_numpy().reshape(47,1))
plt.legend(loc='upper right')
plt.xlabel('Dani')
plt.ylabel('Verovatnoća oporavka')
plt.title("Kaplan Majerova verovatnoća oporavka od COVID-19 virusa po danima")
plt.show()


kmf1 = KaplanMeierFitter() ## instantiate the class to create an object

T = df['DaysRecovery']     ## time to event
E = df['recovered']      ## event occurred or censored

## Two Cohorts are compared. 1. Streaming TV Not Subsribed by Users, 2. Streaming TV subscribed by the users.
groups = df['gender']   
i1 = (groups == 'male')      ## group i1 , having the pandas series for the 1st cohort
i2 = (groups == 'female')     ## group i2 , having the pandas series for the 2nd cohort

## fit the model for 1st cohort
kmf1.fit(T[i1], E[i1], label='male')
a1 = kmf1.plot()

## fit the model for 2nd cohort
kmf1.fit(T[i2], E[i2], label='female')
kmf1.plot(ax=a1)

plt.xlabel('Dani')
plt.ylabel('Verovatnoća oporavka')
plt.title("Kaplan Majerova verovatnoća oporavka od COVID-19 virusa u danima u zavisnosti od pola")
plt.show()
 
##Cox Proportional Hazard Model (Survival Regression)
from lifelines import CoxPHFitter
# My objective here is to introduce you to the implementation of the model.Thus taking subset of the columns to train the model.
df_r= df.loc[:,['DaysRecovery','recovered', 'gender','visiting Wuhan','from Wuhan']]
df_r.head() ## have a look at the data

## Create dummy variables
df_dummy = pd.get_dummies(df_r, drop_first=True)
df_dummy.head()

# Using Cox Proportional Hazards model
cph = CoxPHFitter()   ## Instantiate the class to create a cph object
cph.fit(df_dummy, 'DaysRecovery', event_col='recovered')   ## Fit the data to train the model
cph.print_summary()    ## HAve a look at the significance of the features

cph.plot()

## We want to see the Survival curve at the customer level. Therefore, we have selected 6 customers (rows 5 till 9).
tr_rows = df_dummy.iloc[48:50, 0:]
tr_rows

## Lets predict the survival curve for the selected customers. 
## Customers can be identified with the help of the number mentioned against each curve.
cph.predict_survival_function(tr_rows).plot()


plt.show()