from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def trainmodel(data):
    df = pd.read_csv(data)

    x,y = df.drop('Churn', axis=1), df['Churn']

    x.to_csv('data/train.csv')

    model = RandomForestClassifier()

    model.fit(x,y)


