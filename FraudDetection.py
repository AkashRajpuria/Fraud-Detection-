#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import gridspec
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='once')


# # Reading data

# In[5]:


data = pd.read_csv("card_transdata.csv")


# # Exploratory Data Analysis :Understanding the data

# In[5]:


data.info()


# In[6]:


data.duplicated().value_counts() # There is no Duplicate data


# # There is no null Values and no duplicate data present in our data                  
# 
# # Visualize the distribution of variables
# 

# In[7]:


sns.distplot(data['distance_from_home'])


# In[15]:


sns.distplot(data['distance_from_last_transaction'])


# In[16]:


sns.distplot(data['ratio_to_median_purchase_price'])


# In[17]:


sns.distplot(data['repeat_retailer'])


# In[18]:


sns.distplot(data['used_chip'])#                         1000000 non-null float64


# In[20]:


sns.distplot(data['used_pin_number'])#                   1000000 non-null float64


# In[21]:


sns.distplot(data['online_order']).set_title('lalala')


# In[22]:


sns.distplot(data['fraud']).set_title('Target Variable')


# # Ratio of Target class :(Balanced or Imbalanced Data)

# In[5]:


# Determine number of fraud cases in dataset
Fraud = data[data['fraud'] == 1.0]
Valid = data[data['fraud'] == 0.0]
outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(data[data['fraud'] == 1.0])))
print('Valid Transactions: {}'.format(len(data[data['fraud'] == 0.0])))


# In[9]:


#Percentage of fraud and valid transactions
Occurance = data['fraud'].value_counts()
Percentage= Occurance/len(data.index)*100
print(Percentage)
#Distribution of Fraud and No Fraud
sns.countplot('fraud', data=data)
plt.title('Class Distributions \n (0.0: No Fraud || 1.0: Fraud)', fontsize=14)



# In[8]:


# Correlation matrix
colormap= plt.cm.Greens
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True,cmap=colormap,annot=True)
plt.show()


# # Preparing Independent and Dependent features

# In[11]:


features = ['distance_from_home', 'distance_from_last_transaction',
       'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip',
       'used_pin_number', 'online_order']
# The target variable which we would like to predict, is the 'Class' variable
target = 'fraud'

# Now create an X variable (containing the features) and an y variable (containing only the target variable)
X = data[features]
y = data[target]


# # Splitting the data in Training set and Test set

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# # Logistic Regression with imbalanced data

# In[13]:


model = LogisticRegression(max_iter=7600)

# Train the model using 'fit' method
model.fit(X_train, y_train)

# Test the model using 'predict' method
y_pred = model.predict(X_test)

score = model.score(X_test, y_test)
print("Accuracy : ",score)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))
print("confusion matrix",confusion_matrix(y_test,y_pred))
############################################################################
#LogisticRegression with balanced data
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_sample(X_train, y_train)

model = LogisticRegression(max_iter=10000)

# Train the model using 'fit' method
model.fit(X_sm, y_sm)

# Test the model using 'predict' method
y_pred = model.predict(X_test)

score = model.score(X_sm, y_sm)
print(score)

# Print the classification report 
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# # Precision and recall is not high 

# # Random Forest with balanced and Imabalanced data

# In[14]:


#RandomForestClassifier with imbalanced data
model = RandomForestClassifier(random_state=42)

model.fit(X_train,y_train)

predict = model.predict(X_test)

print(metrics.accuracy_score(y_test, predict))


average_precision = average_precision_score(y_test, predict)

#obtain precision and recall

precision, recall, _ = precision_recall_curve(y_test, predict)


print(confusion_matrix(y_test,predict))


# In[16]:


print(classification_report(y_test,predict))
print(roc_auc_score(y_test,predict))


# In[18]:


#Plotting ROC AUC Curve
from sklearn import metrics
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

y_predict_proba = model.predict_proba(X_test)

auc_score = roc_auc_score(y_test, y_predict_proba[:,1])
print('AUC: %.2f' % auc_score)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_proba[:,1])
plot_roc_curve(fpr, tpr)


# In[ ]:





# In[15]:


import matplotlib.pyplot as plt
import numpy as np
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# 0 = 'distance_from_home'                                                                           
# 1 = 'distance_from_last_transaction'                                                              
# 2 = 'ratio_to_median_purchase_price'                                                                  
# 3 = 'repeat_retailer'                                                                    
# 4 = 'used_chip'                                                                             
# 5 = 'used_pin_number'                                                                    
# 6 = 'online_order'                                                     

# # RandomForestClassifier with SMOTE
# 

# In[ ]:


smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_sample(X_train, y_train)


model = RandomForestClassifier(random_state=42)

model.fit(X_sm,y_sm)


# In[21]:


predict = model.predict(X_test)

print(metrics.accuracy_score(y_test, predict))
average_precision = average_precision_score(y_test, predict)
precision, recall, _ = precision_recall_curve(y_test, predict)
print(confusion_matrix(y_test,predict))


# In[22]:


predict_test = model.predict(X_test)
print(classification_report(y_test,predict_test))
print(roc_auc_score(y_test,predict_test))


# # Using K-fold for 10 fold to check accuracy,precision and recall of each model

# In[48]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
kfold = KFold(n_splits=10, random_state=7,shuffle=True)
model = RandomForestClassifier(random_state=42)
result = cross_validate(model, X, y, cv=kfold, scoring = ['accuracy', 'precision','f1','recall'])
print(result)


# # Saving the model for predicting fraud cases

# In[41]:


import pickle


# In[42]:


filename = 'Randomforest_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[ ]:





# In[ ]:




