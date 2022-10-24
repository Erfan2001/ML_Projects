import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

np.random.seed(0)
items = pd.read_csv('RS_Project/items.csv')
totalItems=items.copy()
reviews = pd.read_csv('RS_Project/reviews.csv')
items = items[['asin','brand', 'rating', 'price']]
items.rename(columns={'rating': 'avg-rating'}, inplace=True)
items = items[items['price'] != 0]
reviews = reviews[['asin', 'name', 'rating']]
#---
xx = [
            {'asin':'B079HB518K', 'rating':4.5},
            {'asin':'B07KFNRQ5S', 'rating':5},
            {'asin':'B07L78G3D2', 'rating':2},
            {'asin':'B00K0NS0P4', 'rating':3},
            {'asin':'B07V5NSD8N', 'rating':4}
         ] 
usersInput = pd.DataFrame(xx)
usersInput=usersInput.merge(items, on='asin', how='left')
brandsOrder={}
currentValue=200
for item in usersInput['brand']:
    if item not in brandsOrder.keys():
        brandsOrder[item]=currentValue
        currentValue-=1
#---
df = pd.merge(items, reviews, on='asin')
df = df.loc[:, ['asin', 'name', 'price', 'avg-rating', 'rating']]
otherUsers = df[df['asin'].isin(np.asanyarray(usersInput['asin']))]
otherUsers = otherUsers[otherUsers['asin'].isin(np.asanyarray(usersInput['asin'].tolist()))]
groupedUsers = otherUsers.groupby(['name'])
groupedUsers = sorted(groupedUsers, key=lambda x: len(x[1]), reverse=True)[:20]
userRating = usersInput['rating'].tolist()
pearsonDic={}
for name, group in groupedUsers:
    new_df = pd.DataFrame(data={'asin': usersInput['asin'], 'rating': np.zeros(len(usersInput))})
    group = group[['asin', 'rating']]
    temp_group = group.groupby(['asin']).mean()
    mergedData = temp_group.merge(new_df, on='asin', how='right')
    mergedData['rating'] = mergedData[['rating_x', 'rating_y']].apply(lambda x: x[0],axis=1)
    mergedData.fillna(np.float32(0),inplace=True)
    mergedData.drop(['rating_x','rating_y'],axis=1,inplace=True)
    groupRating=mergedData['rating'].tolist()
    pearsonDic[name]=pearsonr(userRating,groupRating)[0]

similarity_df=pd.DataFrame.from_dict(pearsonDic, orient='index')
similarity_df.columns=['similarity']
similarity_df['name']=similarity_df.index
similarity_df.sort_values(by='similarity',ascending=False,inplace=True)
similarity_df.reset_index(drop=True,inplace=True)
similarity_df=similarity_df[similarity_df['similarity']>0]
topUsersRatings=similarity_df.merge(reviews,on='name')
topUsersRatings['weighted']=topUsersRatings['similarity']*topUsersRatings['rating']
weightedMatrix=topUsersRatings.groupby('asin').sum()[['similarity','weighted']]
weightedMatrix.columns=['sum_similarity','sum_weighted']
recommendation_df=pd.DataFrame(data={'asin':weightedMatrix.index,'average_score':weightedMatrix['sum_weighted']/weightedMatrix['sum_similarity']})
recommendation_df.reset_index(drop=True,inplace=True)
recommendation_df=recommendation_df[~recommendation_df['asin'].isin(usersInput['asin'].tolist())]
recommendation_df=pd.merge(recommendation_df,totalItems[['asin','brand','url','image','price']],on='asin')
recommendation_df = recommendation_df[recommendation_df['price'] >10]
recommendation_df=recommendation_df[recommendation_df['brand'].isin(usersInput['brand'].tolist())]

for index,row in recommendation_df.iterrows():
    print(brandsOrder[row['brand']])
    recommendation_df.at[index,'brand']=brandsOrder[row['brand']]
recommendation_df.sort_values(by=['average_score','brand'],inplace=True,ascending=False)
recommendation_df.reset_index(drop=True,inplace=True)
print(recommendation_df['brand'])
recommendation_df.to_html('RS_Project/result.html',justify='center')
usersInput.to_html('RS_Project/result2.html',justify='center')