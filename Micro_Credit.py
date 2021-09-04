#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import os
import csv
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
plt.style.use('bmh')


# In[85]:


data = pd.read_csv(r"C:\Users\SAGNIK DAS\Desktop\New folder (3)\Micro_Credit.csv")


# In[86]:


# understanding the data
data.head()


# In[87]:


data.shape


# In[89]:


data.describe()


# In[90]:


data.tail()


# In[91]:


data.columns


# In[92]:


data.info()


# In[93]:


# find the null values
data.isnull().any()


# In[94]:


data.isnull().sum()


# In[95]:


data_num = data.select_dtypes(include = ['float64', 'int64', 'object'])
data_num.head()


# In[96]:


data_num.hist(figsize=(18, 22), bins=55, xlabelsize=10, ylabelsize=10); 


# In[97]:


data.drop([ 'label', 'msisdn', 'aon', 'daily_decr30', 'daily_decr90', 'rental30', 'rental90', 'last_rech_date_ma', 'last_rech_date_da', 'last_rech_amt_ma', 'cnt_ma_rech30', 'fr_ma_rech30', 'sumamnt_ma_rech30', 'medianamnt_ma_rech30', 'medianmarechprebal30', 'cnt_ma_rech90', 'fr_ma_rech90', 'sumamnt_ma_rech90', 'medianamnt_ma_rech90', 'medianmarechprebal90', 'cnt_da_rech30', 'fr_da_rech30', 'cnt_da_rech90', 'fr_da_rech90', 'medianamnt_loans30', 'medianamnt_loans90', 'pcircle', 'pdate'], axis = 1)


# In[2]:


data = pd.read_csv(r"C:\Users\SAGNIK DAS\Desktop\New folder (3)\Micro_Credit_2.csv")


# In[99]:


# understanding the new data
data.head()


# In[100]:


data.shape


# In[101]:


data.describe()


# In[102]:


data.tail()


# In[103]:


data.columns


# In[104]:


data.info()


# In[105]:


# find the null values
data.isnull().any()


# In[106]:


data.isnull().sum()


# In[107]:


corelation = data.corr()


# In[108]:


sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns
            ,annot=True)


# In[109]:


sns.boxplot


# In[110]:


sns.pairplot


# In[111]:


x=data.iloc[:,:9].values
y=data.iloc[:,-1].values
y


# In[112]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[113]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[114]:


x_train


# In[115]:


from sklearn.decomposition import PCA
pca=PCA(n_components=None)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
pca.explained_variance_ratio_


# In[116]:


y_train


# In[117]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)


# In[118]:


x_test


# In[119]:


y_test


# In[122]:


y_pred=classifier.predict(x_test)
y_pred


# In[123]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(x_test,y_pred)
cm


# In[53]:


x_set,y_set = x_train,y_train

plt.scatter(x_set[y_set==1,0],x_set[y_set==1,1],label=1)
plt.scatter(x_set[y_set==2,0],x_set[y_set==2,1],label=2)
plt.scatter(x_set[y_set==3,0],x_set[y_set==3,1],label=3)

A1=np.arange(x_set[:,0].min()-1,x_set[:,0].max()+1,0.01)
A2=np.arange(x_set[:,1].min()-1,x_set[:,1].max()+1,0.01)

X1,X2=np.meshgrid(A1,A2)

z=classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape)
plt.contourf(X1,X2,z,alpha=0.2)


plt.legend()
plt.show()


# In[124]:


X1


# In[125]:


np.array([X1.ravel(),X2.ravel()])


# In[126]:


from sklearn.linear_model import SVC
model_SVC = SVC(kernel = 'rbf', random size = 9)
model_SVC.fit(x_train, y_train)

y_pred_svm = model_SVC.decision_function(x_test)


# In[57]:


from sklearn.linear_model import LogisticRegression
model_logistic = LogisticRegression()
model_logistic.fit(x_train, y_train)

y_pred_logistic = model_logistic.decision_function(x_test)


# In[3]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[4]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive Rate')
    plt.title('Micro Credit Loan (ROC) Curve')
    plt.legend()
    plt.show()


# In[11]:


data_X, class_label=make_classification(n_samples=200000, n_classes=2, weights=[1,1], random_state=1)
x=pd.DataFrame(data_X)
y=pd.DataFrame(class_label)


# In[6]:


trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=1)


# In[7]:


model = RandomForestClassifier()
model.fit(trainX, trainy)


# In[9]:


probs = model.predict.proba(testX)
probs


# In[131]:


probs = probs[:,1]


# In[132]:


auc = roc_auc_score(testy, probs)
print('AUC: %.2f' % auc)


# In[133]:


fpr, tpr, thresholds = roc_curve(testy, probs)


# In[134]:


plot_roc_curve(fpr, tpr)


# In[42]:


from sklearn import tree
data_clf = tree.DecisionTreeClassifier(max_depth=5)
data_clf.fit(x_train, y_train)
data_data.score(x_test, y_test)

y_pred = dt_clf.predict(x_test)
data_clf.score(x_test, y_test)


# In[45]:


y_pred = data.clf.predict(x_test)
confusion_matrix(x_test, y_test)


# In[46]:


from sklearn import ensemble
rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
rf_clf.fit(x_train, y_train)
rf_clf.score(x_test, y_test)


# In[47]:


gb_clf = ensemble.GradientBoostingClassifier()
gb_clf.fit(x_train, y_train)
gb_clf.score(x_test, y_test)


# In[49]:


gb_clf = ensemble.GradientBoostingClassifier(n_estimators=40)
gb_clf.fit(x_train, y_train)
gb_clf.score(x_test, y_test)


# In[50]:


knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(x_train, y_train)
knn_clf.score(x_test, y_test)


# In[ ]:





# In[4]:


y = np.array(data['payback90'])
y.shape


# In[5]:


x = np.array(data.loc[:, 'cnt_loans30' : 'payback90'])
x.shape


# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,)


# In[7]:


x_train.shape


# In[8]:


x_test.shape


# In[9]:


y_train.shape


# In[10]:


y_test.shape


# In[17]:


from sklearn.model_selection import KFold
folds = (KFold(n_splits = 10, shuffle = True, random_state = 100))


# In[18]:


hyper_params = [{'n_features_to_select':list(range(1,8))}]


# In[33]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)


# In[34]:


from sklearn.feature_selection import RFE
rfe = RFE(lm)
from sklearn.model_selection import GridSearchCV
modelcv = GridSearchCV(estimator = rfe,
                      param_grid = hyper_params,
                      scoring = 'r2',
                      cv = folds,
                      verbose = 1,
                      return_train_score = True)
modelcv.fit(x_train, y_train)


# In[35]:


cvresults = pd.DataFrame(modelcv.cv_results_)
cvresults


# In[36]:


print(np.mean(cvresults))


# In[37]:


plt.figure(figsize = (20,8))


# In[38]:


plt.plot(cvresults['param_n_features_to_select'], cvresults['mean_test_score'])
plt.plot(cvresults['param_n_features_to_select'], cvresults['mean_train_score'])
plt.xlabel('Number of features')
plt.ylabel('Optimal number of features')


# In[39]:


n_features_optimal = 6


# In[40]:


lm = LinearRegression()
lm.fit(x_train, y_train)


# In[51]:


rfe = RFE(lm, n_features_to_select = n_features_optimal)


# In[52]:


rfe.fit(x_train, y_train)


# In[53]:


y_pred = lm.predict(x_test)
y_pred


# In[54]:


r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


# In[ ]:




