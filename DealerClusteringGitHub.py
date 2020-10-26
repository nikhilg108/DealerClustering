#!/usr/bin/env python
# coding: utf-8

# In[128]:


### Shree
### Customer Ranking and clustering
# 1.1 Import packages

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print('Packages imported succesfully')

# 1.1 Import Data
dfDealerDataImported = pd.read_csv('DealerFilepath\DealerData.csv')
dfForecastDataImported = pd.read_csv('DealerFilepath\Forecast.csv')
dfPaymentDataImported = pd.read_csv('DealerFilepath\PaymentData.csv')
dfProductDataImported=pd.read_csv('DealerFilepath\ProductData.csv')
dfSalesDataImported=pd.read_csv('DealerFilepath\SalesData.csv')

# 1.2 Fill Missing values

dfDealerDataImported =dfDealerDataImported.fillna(method='ffill') 
dfForecastDataImported = dfForecastDataImported.fillna(method='ffill')
dfPaymentDataImported = dfPaymentDataImported.fillna(method='ffill')
dfProductDataImported=dfProductDataImported.fillna(method='ffill')
dfSalesDataImported=dfSalesDataImported.fillna(method='ffill')


print('Data Imported succesfully')
print('Dealer Data sample',dfDealerDataImported.head())
print('Forecast Data sample',dfForecastDataImported.head())
print('Payment Data sample',dfPaymentDataImported.head())
print(dfSalesDataImported.head())
print('Product Data sample',dfProductDataImported.head())


# In[ ]:





# In[129]:


# 2 Data Manipulation


# 2.1 Calculate FC at dealer level
dfForecastDataGrouped=dfForecastDataImported.copy()
dfForecastDataGrouped=dfForecastDataGrouped.groupby(['Dealer','Product']).sum()
dfForecastDataGrouped=dfForecastDataGrouped.drop(columns='Sr.no',inplace=False)

# 2.2 Calculate Sales at customer
dfSalesDataGrouped=dfSalesDataImported.copy()
dfSalesDataGrouped=dfSalesDataGrouped.groupby(['Dealer','Product']).sum()
dfSalesDataGrouped=dfSalesDataGrouped.drop(columns='Sr.no',inplace=False)
print('Group Sales Data')
print(dfSalesDataGrouped.head())

# 2.3 Calculate FC Accuracy at dealer level
dfFCAccuracy=pd.merge(dfForecastDataGrouped,dfSalesDataGrouped,on=['Dealer','Product'],how='inner')

dfFCAccuracy['FCError']=np.absolute((dfFCAccuracy['Sales']-dfFCAccuracy['FC'])/
                                     np.maximum(dfFCAccuracy['Sales'],dfFCAccuracy['FC']))
dfFCAccuracy['Accuracy']=1-dfFCAccuracy['FCError']
dfDealerFCA=dfFCAccuracy.groupby('Dealer').mean()


# 2.4 Calculate payment speed at dealer level
dfPaymentDataModified=dfPaymentDataImported.copy()
dfPaymentDataModified[['InvoiceDate','PaymentDate']]=dfPaymentDataModified[['InvoiceDate',
                                                                            'PaymentDate']].apply(pd.to_datetime)
dfPaymentDataModified['PaymentTAT']=dfPaymentDataModified['PaymentDate']-dfPaymentDataModified['InvoiceDate']
dfPaymentDataModified['PaymentTAT']=dfPaymentDataModified['PaymentTAT'].clip(lower=pd.Timedelta(0))


dfPaymentDataModified['PaymentTAT']=dfPaymentDataModified['PaymentTAT'].dt.days.astype('int16')
dfPaymentDataGrouped=dfPaymentDataModified.copy()
dfPaymentDataGrouped=dfPaymentDataGrouped.groupby('DealerName').mean()
dfPaymentDataGrouped.drop(columns='InvoiceNo',inplace=True)


# 2.5 Create dealer matrix comprising of Sales, FC accuracy and payment speed

dfDealerConsolidatedAll=dfDealerFCA.copy()
dfDealerConsolidatedAll['PaymentTAT']=dfPaymentDataGrouped['PaymentTAT']
dfDealerConsolidatedAll=pd.merge(dfDealerConsolidatedAll,dfDealerDataImported,on='Dealer',how='inner')
dfDealerConsolidated=dfDealerConsolidatedAll.copy()
dfDealerConsolidated.drop(columns=['FCError','Dealer','FC'],inplace=True)
print('Dealer consolidated info')
print(dfDealerConsolidated.head())




# In[130]:


# 3 Data Transformation
# 3.1 Transform dealer data to standard form

CScaler=StandardScaler()

vx1=CScaler.fit_transform(dfDealerConsolidated)
dfDealerTransformed=pd.DataFrame(data=vx1,columns=['Sales','Accuracy','PaymentTAT',
                                                  'Storage','Road Facing', 'StoreCount','DealerAge','DealerIncome',
                                                  'SubscriptiontoCreditServices'])

print(dfDealerTransformed.head())
print(dfDealerConsolidated.columns)


# In[131]:


# 4 K-Means without PCA
# 4.1 K-Means without PCA

cK4Means=KMeans(n_clusters=4,init='k-means++',random_state=42)
cK4Means.fit(vx1)
lIdentifiedClusterswoPCA=cK4Means.labels_
dfDealerConsolidated['Clusterno']=lIdentifiedClusterswoPCA
lWCSSwoPCA=np.round(cK4Means.inertia_,4)
print('Inertia w/o PCA')
print(lWCSSwoPCA)

fig1,ax=plt.subplots()
plt.scatter(x=dfDealerTransformed.Accuracy,y=dfDealerTransformed.PaymentTAT,c=lIdentifiedClusterswoPCA,cmap='RdBu')
plt.title('Dealer Payment TAT vs FCA')
plt.xlabel('FCA')
plt.ylabel('PaymentTAT')
plt.show()



# In[132]:


# 5 PCA for dimensionality reduction

cPCA=PCA(n_components=3)
vx1PCA=cPCA.fit(vx1)
dfPCAComponents=pd.DataFrame(data=cPCA.components_,columns=['Sales','Accuracy','PaymentTAT',
                                                  'Storage','Road Facing', 'StoreCount','DealerAge','DealerIncome',
                                                  'SubscriptiontoCreditServices'])
lPCAScores=cPCA.fit_transform(vx1)
dfPCAScores=pd.DataFrame(lPCAScores)
print('PCA Scores')
print(dfPCAScores.head())

fig2,ax=plt.subplots()
sns.heatmap(dfPCAComponents,
            vmin=-1,
           vmax=1,
           cmap='RdBu',
           annot=True)





# In[133]:


# 6 Cluster Analysis with PCA

cK4MeansPCA=KMeans(n_clusters=4,init='k-means++',random_state=42)
cK4MeansPCA.fit(lPCAScores)
lIdentifiedClustersPCA=cK4MeansPCA.labels_
dfDealerConsolidatedPCA = dfDealerConsolidated.copy()
dfDealerConsolidatedPCA['Clusterno']=lIdentifiedClustersPCA
lWCSSPCA=np.round(cK4MeansPCA.inertia_,4)
print('Inertia with PCA')
print(lWCSSPCA)

fig3,ax=plt.subplots()
plt.scatter(x=dfDealerTransformed.Accuracy,y=dfDealerTransformed.PaymentTAT,c=lIdentifiedClusters,cmap='RdBu')
plt.title('Dealer Payment TAT vs FCA with PCA')
plt.xlabel('FCA')
plt.ylabel('PaymentTAT')
plt.show()


# In[134]:


# 7 Segment analysis based on model

# 7.1 Model selection decision point

if lWCSSwoPCA < lWCSSPCA:
## *** Additional model selection criteria to be added here
    dfDealerConsolidated['ClusterNoPCA']=lIdentifiedClusterswoPCA
    print('Model selected without dimensionality Reduction')
else:
    dfDealerConsolidated['ClusterNoPCA']=lIdentifiedClustersPCA
    print('Model selected with dimensionality Reduction')

  
print('Dealer Matrix with Cluster number')
print(dfDealerConsolidated.head(10))
# 7.2 Average of Payment TAT, FC Accuracy
dfSegmentAnalysis=dfDealerConsolidated.groupby('ClusterNoPCA').mean()

# 7.3 Count of Segment

dfSegmentCount=dfDealerConsolidated.groupby('ClusterNoPCA').count()

lSegmentCount=[]
lSegmentCount=dfSegmentCount['Sales']

dfSegmentAnalysis['SegmentCount']=lSegmentCount
print('Segment Analysis')
print(dfSegmentAnalysis)


# In[135]:


# 8 Results Export for presentation in Visualization software
dfSegmentAnalysis.to_csv('DealerFilePath/SegmentAnalysis.csv')
dfDealerConsolidated.to_csv('DealerFilePath/DealerClusterDetail.csv')

