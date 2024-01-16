
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import os 
import seaborn as sns


# Cleaning the data representations to float
df = pd.read_csv('data/bbdc_student/bbdc_2023_AWI_data_develop_student.csv',delimiter=';',skiprows=[1])

# Set to 25

df.loc[15587,'NOx'] = 25
df.NOx = df.NOx.astype(float)



def get_week_day(date):
    date = date.split('.')
    day,month,year = map(int,date)
    
    return datetime.date(year,month,day).weekday()

df['year'] = df.Datum.str.split('.').apply(lambda x: int(x[-1]))
df['day'] = df.Datum.str.split('.').apply(lambda x: int(x[0]))
df['month'] = df.Datum.str.split('.').apply(lambda x: int(x[1]))
df.loc[df.Uhrzeit.str.contains('L') == True, 'Uhrzeit'] = '9:30'
df['hour'] = df.Uhrzeit.apply(lambda x: int(x.split(':')[0]) if pd.notna(x) else pd.NA)
df['minutes'] = df.Uhrzeit.apply(lambda x: int(x.split(':')[1]) if pd.notna(x) else pd.NA)
df['weekday'] = df.Datum.apply(get_week_day)
df['time_as_number'] = df.hour + df.minutes / 60
df.time_as_number = pd.to_numeric(df.time_as_number)
df['missing_values'] = df.isna().sum(axis=1)



df = df.set_index('Datum')


sers = (df.loc[df.Uhrzeit.isna()].missing_values < 7)

df.loc[sers.loc[sers].index]
vals = df.loc[df.Uhrzeit.isna()].missing_values < 10
days = vals[vals].index

average_time_per_year = df.groupby('year').time_as_number.mean()

df.loc[days,'time_as_number'] = average_time_per_year.loc[df.loc[days].year].values

for d in days:
    set_to = average_time_per_year.loc[df.loc[d,'year']]
    
    hour = np.floor(set_to)
    minutes = np.round((set_to - hour) * 60).astype(int)
    
    uhrzeit = f'{int(hour)}:{int(minutes)}'
    
    df.loc[d,['hour','minutes','time_as_number','Uhrzeit']] = [hour,minutes, set_to,uhrzeit]
    
df.loc[df.Uhrzeit.isna()].missing_values.value_counts()
df.loc[sers.loc[sers].index]




# Drop Missing Dates entirely
trainings_df = df.loc[(df.weekday < 5) & df.Uhrzeit.notna()].copy()

trainings_df['date_time'] = pd.to_datetime(trainings_df.index + " " + trainings_df.Uhrzeit, format=r'%d.%m.%Y %H:%M')
trainings_df.to_csv('data/trainings_df.csv')



df = trainings_df.copy()



df.pop('missing_values')
df['missing_values'] = df.isna().sum(axis=1)
df.head()

df['datum'] = df.index
df.index = pd.to_datetime(df.index + " " + df.Uhrzeit,format='%d.%m.%Y %H:%M')
df['time_as_number'] = df.index.hour + df.index.minute/60



cols = df.isna().sum(0)
cols = cols.loc[cols > 0].index
cols 

# Method interpolate with time was best

df.loc[:,cols] = df[cols].interpolate(method='time')

df.index = pd.to_datetime(df.datum,format='%d.%m.%Y')

df.isna().sum()

df = df.loc[df.index.year >= 1968] 

#df.loc[df.SECCI.isna(),'SECCI'] = df.SECCI.mean() + np.random.normal(0,0.2,size=(df.SECCI.isna().sum()))
df.loc[df.NO3.isna(),'Salinität'] = df.Salinität.mean() + np.random.normal(0,0.2,size=(df.Salinität.isna().sum()))
df.loc[df.NO3.isna(),'NO2'] = df.NO2.mean() + np.random.normal(0,0.2,size=(df.NO2.isna().sum()))
df.loc[df.NO3.isna(),'NO3'] = df.NO3.mean() + np.random.normal(0,0.2,size=(df.NO3.isna().sum()))

import json
with open('data/bbdc_student/means.json',"r") as f:
    means = pd.Series(json.load(f))
    
with open('data/bbdc_student/stds.json',"r") as f:
    stds = pd.Series(json.load(f))
    stds = np.sqrt(stds)

df = df[stds.index]

df = (df- means) / stds


df_std = df.melt(var_name='Column', value_name='Normalized').dropna()
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
plt.xticks(rotation = 45)
plt.title("Value Distribution")
plt.show()


df.to_csv('data/data_for_darts_normalized_by_test.csv')

