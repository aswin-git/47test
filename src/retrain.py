from scipy.stats import ks_2samp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


newdata = pd.read_csv('data/data.csv')
olddata = pd.read_csv('data/train.csv')

def trainmodel(data):
    df = data

    x,y = df.drop('Churn', axis=1), df['Churn']

    x.to_csv('data/train.csv')

    model = RandomForestClassifier()

    model.fit(x,y)

for i in newdata.drop('Churn', axis=1).columns:
    stat, p = ks_2samp(newdata[i], olddata[i])

    if p < 0.03:
        trainmodel(newdata)
        print('Data drift detected')
        exit

print('Data drift not detected')
    

