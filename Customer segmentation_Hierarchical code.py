#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib as mpl  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 

from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import scipy

import pandas_profiling
from pandas_profiling import ProfileReport

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"




# In[2]:


#loading datasets
points = pd.read_csv("SceneAnalytics.dbo.SP_Points_cleaned.csv")
customer_details=pd.read_csv("SceneAnalytics.dbo.SP_CustomerDetail_cleaned.csv")
fact_attribute=pd.read_csv("SceneAnalytics.dbo.SP_FactAttribute.csv")


# In[ ]:


#Explore points dataset

list(points)
points.shape
points.info()
points.describe().transpose()
points.head(n=20)
points.tail()
pd.isna(points)
points.corr()
sns.pairplot(points)


# In[3]:


# Explore customer_details dataset
list(customer_details)
customer_details.shape
customer_details.info()
customer_details.describe().transpose()
customer_details.head(n=20)
customer_details.tail()
pd.isna(customer_details)
customer_details.corr()
sns.pairplot(customer_details)


# In[4]:


# Feature Engineer Earned and Redeem  

points['Earned']=points.points[points['points']>0]

points['Earned'] = points['Earned'].fillna(0)
points['Redeem']=points.points[points['points']<0]

points['Redeem'] = points['Redeem'].fillna(0)


# In[5]:


#Aggregating Points table  by pointid, transaction amount sum, earned sum and redeem sum

points=points.groupby('Unique_member_identifier').agg(Number_Transactions=('pointid','count'),tran_amt=('TransAmount','sum'),earned=('Earned','sum'),redeem=('Redeem','sum'))

points.info()


# In[6]:


#Feature Engineer LocationType
customer_details['FSA_sliced']=customer_details['FSA'].str.slice(start=1,stop=2)    
customer_details['LocationType'] = np.where(customer_details['FSA_sliced']=="0",'Rural', 'Urban')


# In[7]:


# Merge Points and customer_details datasets
points_customer_details=pd.merge(customer_details,points, on='Unique_member_identifier',how="inner")


# In[8]:


#Check for correlation using pandas profile report
ProfileReport(points_customer_details)


# In[9]:


#Dropping some features
points_customer_details_selected=points_customer_details.drop(["City","StateProv","FSA","FSA_sliced","LanguagePreference", "gender","tran_amt","redeem"],axis=1)


# In[10]:


ProfileReport(points_customer_details_selected)


# In[11]:


#Normalizing Points Total feature

notnormalized=points_customer_details_selected['PointsTotal']
array=notnormalized.to_numpy()
array2=array.reshape(-1, 1) 
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_array = scaler.fit_transform(array2)
normalizedpoints = pd.DataFrame(data=rescaled_array, columns=["PointsTotal_Normalized"])
points_customer_details_selected=pd.concat([points_customer_details_selected, normalizedpoints], axis=1, ignore_index=False)
points_customer_details_selected=points_customer_details_selected.drop(["PointsTotal"],axis=1)


# In[12]:


#Normalizing Number_Transactions feature

notnormalized3=points_customer_details_selected['Number_Transactions']
array3=notnormalized3.to_numpy()
array4=array3.reshape(-1, 1) 
scaler = MinMaxScaler(feature_range=(0, 1)) # function was also defined earlier
rescaled_array4 = scaler.fit_transform(array4)
normalized_NumberTransactions = pd.DataFrame(data=rescaled_array4, columns=["Number_Transactions_Normalized"])
points_customer_details_selected=pd.concat([points_customer_details_selected, normalized_NumberTransactions], axis=1, ignore_index=False)
points_customer_details_selected=points_customer_details_selected.drop(["Number_Transactions"],axis=1)


# In[13]:


#Normalizing earned feature

notnormalized4=points_customer_details_selected['earned']
array4=notnormalized4.to_numpy()
array5=array4.reshape(-1, 1) 
scaler = MinMaxScaler(feature_range=(0, 1)) # function was also defined earlier
rescaled_array5 = scaler.fit_transform(array5)
normalized_earned = pd.DataFrame(data=rescaled_array5, columns=["Earned_Normalized"])
points_customer_details_selected=pd.concat([points_customer_details_selected, normalized_earned], axis=1, ignore_index=False)
points_customer_details_selected=points_customer_details_selected.drop(["earned"],axis=1)


# In[14]:


ProfileReport(points_customer_details_selected)


# In[15]:


#Selecting features from fact attribute dataset
features=['Unique_member_identifier','TuesdayAttendee_tendancy','AttendsWithChild_tendancy']
features2=fact_attribute[features]


# In[16]:


#Creating dummy variables for features in fact__attribute dataset
features2['AttendsWithChild_tendancy_Dummy'] = np.where(features2['AttendsWithChild_tendancy']==True,'1', '0')
features2['TuesdayAttendee_tendancy_Dummy'] = np.where(features2['TuesdayAttendee_tendancy']==True,'1', '0')
features3=features2.drop(['TuesdayAttendee_tendancy','AttendsWithChild_tendancy'],axis=1)


# In[17]:


#Merge features with combined customer and points dataset
X=pd.merge(points_customer_details_selected,features3, on='Unique_member_identifier',how="inner")


# In[18]:


#Fill Nas with 0
X=X.fillna(0)
X


# In[19]:


#Dropping Unique member identifier 
scaler = StandardScaler()

X1 = X.copy()
X2=X1.drop(["Unique_member_identifier"], axis=1)



# In[20]:


#Create dummy categorical features for LocationType and Age_Class
z=pd.get_dummies(X2.LocationType, prefix='LocationType',drop_first=True)
y = pd.get_dummies(X2.age_class, prefix='Age_Class',drop_first=True)


# In[22]:


#Drop LocationType &Age_Class
X2yz = pd.concat([X2, y,z], axis=1)
X3=X2yz.drop(["age_class","LocationType"], axis=1) 
X3


# In[23]:


#Create Clusters using Agglomerative Clustering Algorithm
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
agg.fit(X3)
agg.labels_


# In[24]:


#Silhouette_score
silhouette_score(X3, agg.labels_)


# In[39]:


#Generate Dendogram Chart 
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy


z=hierarchy.linkage(agg.children_,'average')

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

dn=dendrogram(z)

plt.axhline(c='blue',linestyle='--', y=6250) 

plt.show()


# In[25]:


#View members in clusters

for label in set(agg.labels_):
    print('\nCluster{}:'.format(label))
    
    print(X[agg.labels_==label])


# In[26]:


#Descriptive statistics for members in clusters

for label in set(agg.labels_):
    print('\nCluster{}:'.format(label))
    
    print(X[agg.labels_==label].describe())


# In[27]:


#View members in clusters

for label in set(agg.labels_):
    print('\nCluster{}:'.format(label))
    
    print(X[agg.labels_==label])


# In[28]:


#Defining Clusters 
cluster_0=X[agg.labels_==0]
cluster_1=X[agg.labels_==1]
cluster_2=X[agg.labels_==2]
cluster_3=X[agg.labels_==3]
cluster_4=X[agg.labels_==4]
cluster_5=X[agg.labels_==5]
cluster_6=X[agg.labels_==6]


# In[29]:


#Generate Profile Report for Cluster 0
profile0 = ProfileReport(cluster_0)
profile0


# In[30]:


#Generate Profile Report for Cluster 1
profile1 = ProfileReport(cluster_1)
profile1


# In[31]:


#Generate Profile Report for Cluster 2
profile2 = ProfileReport(cluster_2)
profile2


# In[50]:


#Generate Profile Report for Cluster 3
profile3 = ProfileReport(cluster_3)
profile3


# In[51]:


#Generate Profile Report for Cluster 4
profile4 = ProfileReport(cluster_4)
profile4


# In[52]:


#Generate Profile Report for Cluster 5
profile5 = ProfileReport(cluster_5)
profile5


# In[46]:


#Generate Profile Report for Cluster 6
profile6 = ProfileReport(cluster_6)
profile6


# In[ ]:





# In[ ]:




