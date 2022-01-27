#!/usr/bin/env python
# coding: utf-8

# <img src="https://user-images.strikinglycdn.com/res/hrscywv4p/image/upload/c_limit,fl_lossy,h_300,w_300,f_auto,q_auto/1266110/Logo_wzxi0f.png" style="float: left; margin: 20px; height: 55px">
# 
# **It's not always about convincing your parents of what you want to do, but just saying, 'This is what I'm doing; this is what I love' - [Stormzy](https://en.wikipedia.org/wiki/Stormzy)**

# # Optimisation

# ###### Set-up

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# Defining functions we will be using for our reports later

# In[2]:


from sklearn.metrics import roc_auc_score
def scores(model, X_train, X_val, y_train, y_val):
    train_prob = model.predict_proba(X_train)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    train = roc_auc_score(y_train,train_prob)
    val = roc_auc_score(y_val,val_prob)
    print('train:',round(train,2),'test:',round(val,2))


# In[3]:


def annot(fpr,tpr,thr):
    k=0
    for i, j in zip(fpr,tpr):
        if k % 50 == 0:
            plt.annotate(round(thr[k],2),
                         xy=(i,j), 
                         textcoords='data')
        k+=1


# In[4]:


from sklearn.metrics import roc_curve
def roc_plot(model,X_train,y_train,X_val,y_val):
    train_prob = model.predict_proba(X_train)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    plt.figure(figsize=(7,7))
    for data in [[y_train, train_prob],[y_val, val_prob]]: # ,[y_test, test_prob]
        fpr, tpr, threshold = roc_curve(data[0], data[1])
        plt.plot(fpr, tpr)
    annot(fpr, tpr, threshold)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.ylabel('TPR (power)')
    plt.xlabel('FPR (alpha)')
    plt.legend(['train','val'])
    plt.show()


# In[5]:


def opt_plots(opt_model):
    opt = pd.DataFrame(opt_model.cv_results_)
    cols = [col for col in opt.columns if ('mean' in col or 'std' in col) and 'time' not in col]
    params = pd.DataFrame(list(opt.params))
    opt = pd.concat([params,opt[cols]],axis=1,sort=False)
    
    plt.figure(figsize=[15,4])
    plt.subplot(121)
    sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_train_score')*100)
    plt.title('ROC_AUC - Training')
    plt.subplot(122)
    sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_test_score')*100)
    plt.title('ROC_AUC - Validation')
#     return opt


# ## Data preparation

# In[6]:


food = pd.read_csv('Resources/foodinfo.csv')
food.shape


# In[7]:


food.head()


# In[8]:


target = 'nutrition-score-uk_100g'
food['good'] = np.where(food[target]>=8,1,0)

X = food.drop([target,'good'],axis=1)
X = X.iloc[:,-2:]
y = food['good']

y.value_counts()


# In[10]:


X.head()


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=.2,random_state=42)


# ## Modelling

# ### Decision Trees

# #### Basic

# In[12]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)


# In[13]:


scores(model,X_train,X_val,y_train,y_val)
roc_plot(model,X_train,y_train,X_val,y_val)


# The ROC curve shows signs of overfitting<br>
# We want to try some hyperparameter optimisation to reduce the discrepancy between train and validation

# #### Grid Search CV with Decision Trees

# In[14]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)


# In[18]:


type(best_model)


# In[15]:


get_ipython().run_cell_magic('time', '', "param_grid = {'max_depth':range(10,50+1,10),'min_samples_leaf':[5,50,500]}\n\nopt_model = GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid,cv=skf,scoring='roc_auc',return_train_score=True,n_jobs=-1)\nopt_model.fit(X_train,y_train)\nbest_model = opt_model.best_estimator_\n\nscores(best_model,X_train,X_val,y_train,y_val)\nprint(opt_model.best_params_)\nopt_plots(opt_model)")


# In[19]:


roc_plot(best_model,X_train,y_train,X_val,y_val)


# In[20]:


get_ipython().run_cell_magic('time', '', "param_grid = {'max_depth':range(15,50+1,5),'min_samples_leaf':[5,10,15,30,40,50,75,100]}\n\nopt_model = GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid,cv=skf,scoring='roc_auc',return_train_score=True,n_jobs=-1)\nopt_model.fit(X_train,y_train)\nbest_model = opt_model.best_estimator_\n\nscores(best_model,X_train,X_val,y_train,y_val)\nprint(opt_model.best_params_)\nopt_plots(opt_model)")


# In[21]:


scores(best_model,X_train,X_val,y_train,y_val)
roc_plot(best_model,X_train,y_train,X_val,y_val)


# In[22]:


get_ipython().run_cell_magic('time', '', "param_grid = {'max_depth':range(15,25+1),'min_samples_leaf':range(5,30)}\n\nopt_model = GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid,cv=skf,scoring='roc_auc',return_train_score=True,n_jobs=-1)\nopt_model.fit(X_train,y_train)\nbest_model = opt_model.best_estimator_\n\nscores(best_model,X_train,X_val,y_train,y_val)\nprint(opt_model.best_params_)\nopt_plots(opt_model)")


# In[23]:


scores(best_model,X_train,X_val,y_train,y_val)
roc_plot(best_model,X_train,y_train,X_val,y_val)


# In[24]:


best_tree = best_model


# Feature Importance

# In[25]:


best_tree.feature_importances_


# In[26]:


from sklearn import tree


# In[27]:


plt.figure(figsize=(22.3,10))
tree.plot_tree(best_tree, filled=True)
plt.show()


# ### Random Forest

# #### Basic

# In[28]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10,n_jobs=-1,random_state=42)
model.fit(X_train,y_train)


# In[29]:


# Even off-the-shelf is better!!
scores(model,X_train,X_val,y_train,y_val)
roc_plot(model,X_train,y_train,X_val,y_val)


# The ROC curve shows signs of overfitting<br>
# We want to try some hyperparameter optimisation to reduce the discrepancy between train and validation

# #### Grid Search CV with Random Forest

# In[30]:


get_ipython().run_cell_magic('time', '', "param_grid = {'max_depth':range(1,50+1,10),'min_samples_leaf':[5,50,500]}\n\nopt_model = GridSearchCV(model,param_grid,cv=skf,scoring='roc_auc',return_train_score=True,n_jobs=-1,verbose=10)\nopt_model.fit(X_train,y_train)\nbest_model = opt_model.best_estimator_\n\nscores(best_model,X_train,X_val,y_train,y_val)\nprint(opt_model.best_params_)\nopt_plots(opt_model)")


# In[31]:


roc_plot(best_model,X_train,y_train,X_val,y_val)


# In[32]:


get_ipython().run_cell_magic('time', '', "param_grid = {'max_depth':range(15,45+1,5),'min_samples_leaf':range(5,45+1,5)}\n\nopt_model = GridSearchCV(model,param_grid,cv=skf,scoring='roc_auc',return_train_score=True,n_jobs=-1)\nopt_model.fit(X_train,y_train)\nbest_model = opt_model.best_estimator_\n\nscores(best_model,X_train,X_val,y_train,y_val)\nprint(opt_model.best_params_)\nopt_plots(opt_model)")


# In[33]:


scores(best_model,X_train,X_val,y_train,y_val)
roc_plot(best_model,X_train,y_train,X_val,y_val)


# In[34]:


get_ipython().run_cell_magic('time', '', "param_grid = {'max_depth':range(20,40+1,1),'min_samples_leaf':range(1,10+1)}\n\nopt_model = GridSearchCV(model,param_grid,cv=skf,scoring='roc_auc',return_train_score=True,n_jobs=-1)\nopt_model.fit(X_train,y_train)\nbest_model = opt_model.best_estimator_\n\nscores(best_model,X_train,X_val,y_train,y_val)\nprint(opt_model.best_params_)\nopt_plots(opt_model)\nplt.autoscale()")


# In[35]:


scores(best_model,X_train,X_val,y_train,y_val)
roc_plot(best_model,X_train,y_train,X_val,y_val)


# In[36]:


get_ipython().run_cell_magic('time', '', "param_grid = {'max_depth':range(15,30+1),'min_samples_leaf':range(2,5+1)}\n\nopt_model = GridSearchCV(model,param_grid,cv=skf,scoring='roc_auc',return_train_score=True,n_jobs=-1)\nopt_model.fit(X_train,y_train)\nbest_model = opt_model.best_estimator_\n\nscores(best_model,X_train,X_val,y_train,y_val)\nprint(opt_model.best_params_)\nopt_plots(opt_model)")


# In[37]:


scores(best_model,X_train,X_val,y_train,y_val)
roc_plot(best_model,X_train,y_train,X_val,y_val)


# In[38]:


best_forest = best_model


# Feature Importance

# In[41]:


X.head()


# In[39]:


best_forest.feature_importances_


# In[43]:


pd.DataFrame(opt_model.cv_results_)

