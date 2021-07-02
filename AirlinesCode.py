# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:00:09 2021

@author: sairaj
"""
# importing required libraries

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# loading dataset
dataset = pd.read_excel("D:/DMhieraricalClust ml/assignment/EastWestAirlines.xlsx", sheet_name='data')

dataset.head()

# renaming column names
dataset.columns = ['ID', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles','Bonus_miles', 'Bonus_trans', 
                 'Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll', 'Award'] 

# we can use dataset.keys() or dataset.columns to get cloumn name list
dataset.keys()
dataset.columns

#checking datatypes
dataset.info()
dataset.head()

# writing loop to check datatype other than integer, if found any that will be replaced by Nan.
def column_preprocessor(df):
    count = 0
    for row in df:
        try:
            if type(row) != int:
                df.loc[count] = np.nan
        except:
            pass
        count +=1

column_preprocessor(dataset[dataset.columns])
dataset.isna().any().sum() # to know if there are any NA values
#  performing EDA,
data1 = dataset.describe().transpose()
dataset.head()

# Previously miles award status
dataset['Award'].value_counts().plot(kind='pie', autopct='%2.0f%%', fontsize='18', 
                                        colors = ['#F11A05','#43E206'], shadow =True)
plt.show()

# Since from previous award status most of the customer not awarded with any schemes.So will drop the 'Award' Column. in analysis add it on after cluster no. results.
# DATA- PREPROCESSING
# as we know ID & award will not make much contribution during clutering. we will drop both columns
dataset1 =  dataset.drop(['ID','Award'], axis=1)
dataset1.head()
 
#HIERARICAL CLUSTERING
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

df_norm = norm_func(dataset1)
df_norm.describe()

# to create a dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean") # "complete" is a linkage function

# dendrogram
plt.figure(figsize=(15, 10));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Distance');plt.ylabel('Height')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering # sklearn is ML library
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_) # to get the clusters in series or in column format

dataset1['clust'] = cluster_labels # creating a new column and assigning it to new column 

dataset01 = dataset.iloc[:, [10,0,1,2,3,4,5,6,7,8,9]]
dataset01.head()

# Aggregate mean of each cluster
dataset01.iloc[:, 2:].groupby(dataset1.clust).mean() # .groupby is used to apply the aggregation the DataFrame

# creating a csv file 
dataset01.to_csv("Airlines.csv", encoding = "utf-8")

import os
os.getcwd()

# KMEANS CLUSTERING

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y="Y", kind = "scatter") # kind = "scatter" generates the plot with the sactteer diagram of the data

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on University Data set 
airlines_data = pd.read_excel("D:\\DMhieraricalClust ml\\assignment\\airlinesdata.xlsx", sheet_name = "data")
airlines_data.head()

# renaming column names
airlines_data.columns = ['ID', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles','Bonus_miles', 'Bonus_trans', 
                 'Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll', 'Award'] 

airlines_data.describe()
airlines1 = airlines_data.drop(["ID","Award"], axis = 1)



# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airlines1.iloc[:, 0:])
df_norm # normalized data


###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_) # .inertia is a paramiter which is used to capturing the WSS(with in sum of squares)  
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
airlines1['clust'] = mb # creating a  new column and assigning it to new column 

airlines1.head()
df_norm.head() # head() used to record the first 5 data

airlines1 = airlines1.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]]
airlines1.head()

airlines1.iloc[:, 2:8].groupby(airlines1.clust).mean()

airlines1.to_csv("Kmeans_airlines_data.csv", encoding = "utf-8")

import os
os.getcwd()








































