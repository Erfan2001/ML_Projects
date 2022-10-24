import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score,mean_squared_error
import pickle
# empty strings '' or numpy.inf are not considered NA values
pd.options.mode.use_inf_as_na = True
df=pd.read_csv("Regression/housePrice.csv")
new1=df["Address"]
new2=pd.isnull(df["Address"])
df.dropna(subset=["Address"],inplace=True)
df=df[df['Area'].str.isdigit()]
df["Area"]=df['Area'].astype(int)
df["Address"]=df['Address'].astype('category').cat.codes
df.drop('Price(USD)',axis=1,inplace=True)
total=np.random.rand(len(df)) <0.2
train=df[total]
test=df[~total]
# df["Address"]=df['Address'].cat.codes
score=0
print(df[train.columns.difference(['Price'])])
for i in range(0,10):
    reg=linear_model.LinearRegression()
    reg.fit(train[train.columns.difference(['Price'])], train["Price"])
    pred_y=reg.predict(test[train.columns.difference(['Price'])])
    fig=plt.figure()
    ax=fig.add_subplot()
    if score<r2_score(test["Price"], pred_y):
        score=r2_score(test["Price"], pred_y)
pickle.dump(reg, open('Regression/test.txt', 'wb'))
# loaded_model = pickle.load(open('test.sav', 'rb'))
# result = loaded_model.score(test.columns.difference(['Price']), test["Price"])
print(score)