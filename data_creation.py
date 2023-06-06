import pandas as pd
import numpy as np

def obsh(max_temp,date,min_,max_):
  l1=[max_temp[-1]]
  i=0
  while i < date:
    l = np.random.randint(0,2)
    if max_>l1[i]>min_: l =l
    elif l1[i]<min_ : l = 1
    elif l1[i]>max_ : l = 0
    if l == 0:
      l1.append(round(np.random.uniform(l1[i] - np.random.randint(1, 4), l1[i]),3))
    if l == 1:
      l1.append(round(np.random.uniform(l1[i], l1[i] + np.random.randint(1, 4)),3))
    i=i+1 
  max_temp+=  l1
  return max_temp
  
def max_p_year(max_temp):
  max_temp = obsh(max_temp,183,-30,10)
  max_temp = obsh(max_temp,182,10,50)
  return max_temp


date = np.arange(10100)
rng = np.random.default_rng()
max_temp = [round(np.random.uniform(-30, 10),3)]
for i in np.arange(int(len(date)/365)):
  max_temp_ = max_p_year(max_temp)

if (len(date)-1 - len(max_temp_))>183:
  max_temp_ = obsh(max_temp_,183,-30,10)
if (len(date)-1 - len(max_temp_))>182:
  max_temp_ = obsh(max_temp_,182,10,50)

if (len(date)-1 - len(max_temp_))>0:
  max_temp_ = obsh(max_temp_,(len(date)-1 - len(max_temp_)),-10,10)

min_temp_ = []
for i in max_temp_:
  min_temp_.append(round(np.random.uniform(i-20, i-7),3)+round(np.random.uniform(-15, 10),3))

  
for i in np.random.randint(0,len(date),int(len(date)/70)):
  min_temp_[i] =round(np.random.uniform(-40, 40),3)

i=0 
mean_temp=[]
while i < len(max_temp_):
  mean_temp.append(np.mean([max_temp_[i],min_temp_[i]]))
  i=i+1

for i in np.random.randint(0,len(date),int(len(date)/70)):
  mean_temp[i] =round(np.random.uniform(-40, 40),3)

i=0 
sunshine=[]
while i < len(mean_temp):
  m= min_temp_[i]/np.random.uniform(16, 19)+np.random.uniform(9, 15)
  sunshine.append(round(abs(m),1))
  i=i+1

global_radiation = []
i=0
while i < len(sunshine):
  global_radiation.append(round(sunshine[i]**2/2+max_temp_[i]**2/np.random.uniform(10, 12),1))
  i=i+1

pressure = []
i=0
while i < len(global_radiation):
  k=sunshine[i]*600+round(np.random.uniform(95960/1.05,109820/1.1),1)+max_temp_[i]*10

  pressure.append(round(k,1))
  i=i+1

cloud_cover = []
i=0
while i < len(global_radiation):
  cloud_cover.append(round(np.random.uniform(10, 16)-pressure[i]/11000,0))
  i=i+1

df_nev = pd.DataFrame()
df_nev["date"] = date
df_nev["max_temp"] = max_temp_
df_nev["min_temp"] = min_temp_
df_nev["mean_temp"] = mean_temp
df_nev["sunshine"] = sunshine
df_nev["global_radiation"] = global_radiation
df_nev["pressure"] = pressure
df_nev["cloud_cover"] = cloud_cover

"""date - записанная дата измерения
cloud_cover - измерение облачности в октах 
sunshine - измерение солнечного сияния в часах (часах)
global_radiation - измерение освещенности в ваттах на квадратный метр (Вт/м2)
max_temp (°C)
mean_temp (°C)
min_temp (°C) 
pressure - измерение давления в Паскалях (Па)
"""

from sklearn.model_selection import train_test_split
X = df_nev.drop(["date","sunshine"],axis=1)
y = df_nev['sunshine']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=42)
X_train['sunshine'] = y_train
X_train.to_csv('train/train.csv', index=False)
X_test.to_csv('test/test.csv', index=False)
y_test.to_csv('test/y_test.csv', index=False)

#import seaborn as sns
#sns.heatmap(df_nev.corr(), annot=True)
