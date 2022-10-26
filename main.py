import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score

np.random.seed(0)
df =pd.read_csv('Clustering/mall_customers.csv')
# Convert Gender to Number
df['Gender']=df['Gender'].astype('category').cat.codes
df.drop(['CustomerID'],axis=1,inplace=True)
df.rename(columns={'Annual Income (k$)':'Income','Spending Score (1-100)':'Score'},inplace=True)
features=df.columns

def multipleFeatures_Plot(dataFrame,clusters,title='My plot'):
    if -1 in set(clusters):
        n_clusters=len(set(clusters))-1
    else:
        n_clusters=len(set(clusters))
    features=dataFrame.columns
    colors=plt.cm.Spectral(np.linspace(0,1,n_clusters))
    flag=False
    if -1 in set(clusters):
        flag=True
    fig, axes=plt.subplots(len(features),len(features))
    temp_df=dataFrame.copy()
    temp_df['cluster']=clusters
    for i in range(0,len(features)):
        for j in range(0,len(features)):
            for k in range(n_clusters):
                if flag and k==n_clusters-1:
                    k=-1
                new_temp_df=temp_df[temp_df['cluster']==k]
                axes[i,j].plot(new_temp_df[features[i]],new_temp_df[features[j]],color='black' if k==-1 else colors[k],marker='o',linestyle='')
                axes[i,j].set(xlabel=features[i], ylabel=features[j])
    plt.subplots_adjust(wspace=0.5, hspace=0.6)
    fig.suptitle(title)
    plt.show()

def twoSpecificFeatures_Plot(dataFrame,clusters,row,column,axes,title='My plot'):
    n_clusters=len(set(clusters))
    flag=False
    if -1 in set(clusters):
        flag=True
    colors=plt.cm.Spectral(np.linspace(0,1,n_clusters))
    temp_df=dataFrame.copy()
    temp_df['cluster']=clusters
    for k in range(n_clusters):
        if flag and k==n_clusters-1:
            k=-1
        new_temp_df=temp_df[temp_df['cluster']==k]
        axes[row,column].plot(new_temp_df['Income'],new_temp_df['Score'],color='black' if k==-1 else colors[k],marker='o',linestyle='')
        axes[row,column].set(xlabel='Income', ylabel='Score')
    axes[row,column].set_title(title)
new_df=pd.DataFrame(StandardScaler().fit(df).transform(df),columns=df.columns)

# ------------------------   Cluster Dataset Only using all features   ----------------------------------

print("--------------------------- K-Means ------------------------------------")
kMeans_df=KMeans(init='k-means++',n_clusters=20,n_init=3).fit(new_df)
print(kMeans_df.labels_)
multipleFeatures_Plot(new_df,kMeans_df.labels_,'K-Means')
print("--------------------------- Hierarchical ------------------------------------")
agg=AgglomerativeClustering(n_clusters=3,linkage='average')
distance_mat=distance_matrix(new_df.values,new_df.values)
agg.fit(distance_mat)
print(agg.labels_)
multipleFeatures_Plot(new_df,agg.labels_,'Hierarchical')
z=hierarchy.linkage(distance_mat,'complete')
def llf(id):
    return '[%s]' % (new_df['Gender'][id] )
hierarchy.dendrogram(z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =10)
plt.show()
print("--------------------------- DBSCAN ------------------------------------")
dbscan=DBSCAN(eps=.5,min_samples=5).fit(new_df.values)
multipleFeatures_Plot(new_df,dbscan.labels_,'DBSCAN')
print(dbscan.labels_)

# ------------------------   Cluster Dataset Only with Income&Score   ----------------------------------
fig, axes=plt.subplots(2,2)
plt.subplots_adjust(wspace=0.5, hspace=0.6)
selected_df=df.drop(['Gender','Age'],axis=1)
selected_df=pd.DataFrame(StandardScaler().fit(selected_df).transform(selected_df),columns=selected_df.columns)
#--- 1
kMeans_df=KMeans(init='k-means++',n_clusters=5,n_init=12).fit(selected_df)
twoSpecificFeatures_Plot(selected_df,kMeans_df.labels_,0,0,axes,'K-Means')
#--- 2
agg=AgglomerativeClustering(n_clusters=5,linkage='average')
distance_mat=distance_matrix(selected_df.values,selected_df.values)
agg.fit(distance_mat)
twoSpecificFeatures_Plot(selected_df,agg.labels_,0,1,axes,'Hierarchical')
#--- 3
dbscan=DBSCAN(eps=.3,min_samples=4).fit(selected_df.values)
twoSpecificFeatures_Plot(selected_df,dbscan.labels_,1,0,axes,'DBSCAN-1')
#--- 4
dbscan=DBSCAN(eps=.4,min_samples=6).fit(selected_df.values)
twoSpecificFeatures_Plot(selected_df,dbscan.labels_,1,1,axes,'DBSCAN-2')

plt.show()