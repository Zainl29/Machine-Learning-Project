
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("C:\\Users\\zainl\\Downloads\\bank-additional-full.csv", sep=';')

df = df.drop('duration',axis=1)


#Using MinMaxScaling on numerical variables
numerical_columns = [0,10,11,12,14,15,16,17,18]
scaler = MinMaxScaler()
df[df.columns[numerical_columns]] = scaler.fit_transform(df[df.columns[numerical_columns]])

print(df[['age','campaign','pdays','previous',
    'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']].head())


#Converting Categorical variables to numerical 
categorical_columns = ['job','marital','education','default','housing','loan',
        'contact','month','day_of_week','poutcome','y']

le = LabelEncoder()

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
    
print(df[['job','marital','education','default',
    'housing','loan','contact','month','day_of_week', 'y']].head())


df.to_csv("C:\\Users\\zainl\\Downloads\\Preprocessing-bank-additional-full.csv", index=False)





