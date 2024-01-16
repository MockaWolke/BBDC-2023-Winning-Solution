import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import os 

df = pd.read_csv('data/bbdc_prof/bbdc_2023_AWI_data_develop_professional.csv',delimiter=';',skiprows=[1])
df.dtypes

indices = []
for name,c in df.items():
    if c.dtype != np.float64:
        
        empty_brakcets = c.dropna().str.contains('?',regex=False)
        indices.extend(c.dropna()[empty_brakcets].index.tolist())
df.loc[indices]

df = df.drop(index = indices)
df.Uhrzeit = df.Uhrzeit.str.replace("L",':')

df.loc[3317,'Uhrzeit'] = '9:30'
df.index = pd.to_datetime(df.Datum,dayfirst=True)
df['missing_values'] = df.isna().sum(axis=1)
df['time_as_number'] = df.index.hour + df.index.minute / 60


mask = df.loc[df.index.weekday > 4].missing_values != 10
df.loc[df.index.weekday > 4].loc[mask]

eval_df = pd.read_csv('data/bbdc_prof/bbdc_2023_AWI_data_evaluate_skeleton_professional.csv',delimiter=';',skiprows=[1])
eval_df.index = pd.to_datetime(eval_df.Datum,dayfirst=True)

eval_df.index.weekday.unique()

df.isna().sum(0)

sers = (df.loc[df.Uhrzeit.isna()].missing_values < 10)
sers = sers[sers].index
df.loc[sers]

average_time_per_year = df.groupby(df.index.year).time_as_number.mean()

for d in sers:
    set_to = average_time_per_year.loc[d.year]
    
    hour = np.floor(set_to)
    minutes = np.round((set_to - hour) * 60).astype(int)
    
    uhrzeit = f'{int(hour)}:{int(minutes)}'
    
    df.loc[d,'Uhrzeit'] = uhrzeit

old_index = df.index.copy()
    
df.index = pd.to_datetime(df.index.to_series().apply(str) + " " + df.Uhrzeit)
    
df['time_as_number'] = df.index.hour + df.index.minute / 60
df.index = old_index
df.loc[df.Uhrzeit.isna()].missing_values.value_counts()

# Set missing values to 0

df = df.loc[(df.index.weekday < 5) & df.Uhrzeit.notna()]


labels = [ 'SECCI', 'Temperatur', 'Salinität', 'SiO4', 'PO4', 'NO2', 'NO3', 'NOx', 'NH4']

for d in labels:
    df[d]  = pd.to_numeric(df[d])

df[[ 'SECCI', 'Temperatur', 'Salinität', 'SiO4', 'PO4', 'NO2', 'NO3', 'NOx', 'NH4']].corr()

df = df.drop(columns=['missing_values','Datum',"Uhrzeit"])

df

df.to_csv('data/prof_trainigns_df.csv')




df = df.loc[df.index.year > 1968].copy()
df = df.interpolate(method = 'time').drop(columns = "time_as_number")


means = pd.read_json("data/bbdc_prof/means.json",typ='series')
variance = pd.read_json("data/bbdc_prof/variances.json",typ='series')

columns = means.index.tolist()
df[columns] = (df[columns]- np.array(means)) / np.sqrt(np.array(variance))


from sklearn.ensemble import GradientBoostingRegressor
val_df = df.loc[(df.index.year == 2005) | (df.index.year == 2006)][["Temperatur"	,"Salinität", "NO2","NO3","NOx"]]
train_df = df.loc[df.index.year > 2006][["Temperatur"	,"Salinität", "NO2","NO3","NOx"]]

val_x = val_df.drop(columns="NOx")
val_y = val_df.NOx
train_x =  train_df.drop(columns="NOx")
train_y = train_df.NOx


switcher = GradientBoostingRegressor(n_estimators=100,random_state=0
)

switcher.fit(train_x,train_y)

vals = switcher.predict(val_x)
np.sqrt(np.mean(np.square(vals - val_y)))

train = df.loc[df.index.year < 2004]

df.loc[df.index.year < 2004,'NOx'] = switcher.predict(train[["Temperatur"	,"Salinität", "NO2","NO3"]].fillna(method='bfill'))

df.to_csv("data/pro_train.csv")