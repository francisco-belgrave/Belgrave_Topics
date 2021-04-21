#!/usr/bin/env python
# coding: utf-8

# <img src="https://user-images.strikinglycdn.com/res/hrscywv4p/image/upload/c_limit,fl_lossy,h_300,w_300,f_auto,q_auto/1266110/Logo_wzxi0f.png" style="float: left; margin: 20px; height: 55px">
# 
# **Celerity is never more admired than by the negligent - [Cleopatra](https://en.wikipedia.org/wiki/Cleopatra)**

# # Chapter 11 Regressions

# The linear least squares fit in the previous chapter is an example of **regression**, which is the more general problem of fitting any kind of model to any kind of data. This use of the term “regression” is a historical accident; it is
# only indirectly related to the original meaning of the word.
# 
# The goal of regression analysis is to describe the relationship between one set of variables, called the **dependent variables**, and another set of variables, called independent or **explanatory variables**.
# 
# In the previous chapter we used mother’s age as an explanatory variable to predict birth weight as a dependent variable. When there is only one dependent and one explanatory variable, that’s **simple regression**. In this chapter, we move on to multiple regression, with more than one explanatory variable. If there is more than one dependent variable, that’s multivariate regression.
# 
# If the relationship between the dependent and explanatory variable is linear, that’s linear regression. For example, if the dependent variable is y and the explanatory variables are x 1 and x 2 , we would write the following linear regression model:
# 
# ![alt text](Resources/Think_Stats/notebookpics/regression.png "Title")
# 
# where β0 is the intercept, β1 is the parameter associated with x 1 , β 2 is the parameter associated with x 2 , and ε is the residual due to random variation or other unknown factors.
# 
# Given a sequence of values for y and sequences for x 1 and x 2 , we can find the parameters, β 0 , β 1 , and β 2 , that minimize the sum of ε2. This process is called ordinary least squares. The computation is similar to `thinkstats2.LeastSquare`, but generalized to deal with more than one explanatory variable. You can find the details at https://en.wikipedia.org/wiki/Ordinary_least_squares.

# ## 1 StatsModels

# In the previous chapter I presented `thinkstats2.LeastSquares`, an implementation of simple linear regression intended to be easy to read. For multiple regression we’ll switch to StatsModels, a Python package that provides
# several forms of regression and other analyses. If you are using Anaconda, you already have StatsModels; otherwise you might have to install it.
# 
# As an example, I’ll run the model from the previous chapter with StatsModels:

# In[1]:


from Resources.Think_Stats.Thinkstats2 import first
from Resources.Think_Stats.Thinkstats2 import thinkstats2
import numpy as np
from Resources.Think_Stats.Thinkstats2 import regression
import pandas as pd
pd.options.display.max_columns = 100


# In[2]:


import statsmodels.formula.api as smf


# In[3]:


live, firsts, others = first.MakeFrames()


# In[4]:


formula = 'totalwgt_lb ~ agepreg'
model = smf.ols(formula, data=live)
results = model.fit()
results.summary()


# statsmodels provides two interfaces (APIs); the “formula” API uses strings to identify the dependent and explanatory variables. It uses a syntax called patsy; in this example, the ~ operator separates the dependent variable on the left from the explanatory variables on the right.
# 
# `smf.ols` takes the formula string and the DataFrame, live, and returns an OLS object that represents the model. The name ols stands for “ordinary least squares.”
# 
# The fit method fits the model to the data and returns a `RegressionResults` object that contains the results.
# 
# The results are also available as attributes. `params` is a Series that maps from variable names to their parameters, so we can get the intercept and slope like this:

# In[5]:


inter = results.params['Intercept']
slope = results.params['agepreg']
inter, slope


# The estimated parameters are 6.83 and 0.0175, the same as from LeastSquares.
# 
# pvalues is a Series that maps from variable names to the associated p-values, so we can check whether the estimated slope is statistically significant:

# In[6]:


slope_pvalue = results.pvalues['agepreg']
slope_pvalue


# The p-value associated with agepreg is 5.7e-11, which is less than 0.001, as expected.

# In[7]:


results.rsquared


# `results.rsquared` contains R 2 , which is 0.0047. results also provides f_pvalue, which is the p-value associated with the model as a whole, similar to testing whether R 2 is statistically significant.
# 
# And results provides resid, a sequence of residuals, and fittedvalues, a sequence of fitted values corresponding to agepreg.
# 
# The results object provides summary(), which represents the results in a readable format.

# In[8]:


results.summary()


# But it prints a lot of information that is not relevant (yet), so I use a simpler function called SummarizeResults. Here are the results of this model:

# In[9]:


regression.SummarizeResults(results)


# Std(ys) is the standard deviation of the dependent variable, which is the RMSE if you have to guess birth weights without the benefit of any explanatory variables. Std(res) is the standard deviation of the residuals, which
# is the RMSE if your guesses are informed by the mother’s age. As we have already seen, knowing the mother’s age provides no substantial improvement to the predictions.
# 
# Fit a stats model to the baby data to assess a model that predicts weight of the baby based on lenght of the pregnancy:

# In[10]:


# Code it here

formula = 'totalwgt_lb ~ prglngth'
model = smf.ols(formula, data=live)
results = model.fit()

inter = results.params['Intercept']
slope = results.params['prglngth']
print(results.summary())


# In[11]:


regression.SummarizeResults(results)


# How do you interpret the Beta and the intercept observed on the model?

# In[12]:


# Answer here

# y = B0 + B1X1
# B0 = intercept when prglngth is 0 totalwgt_lb is equal to the intercept -2.72
# B1 = "slope" How change one variable when the another one change. If prglngth grows 1 week, on average totalwgt_lb will increase by 0.259 pounds


# ## 2 Multiple Regression

# In chapter 4 we saw that first babies tend to be lighter than others, and this effect is statistically significant. But it is a strange result because there is no obvious mechanism that would cause first babies to be lighter. So we
# might wonder whether this relationship is **spurious**.
# 
# In fact, there is a possible explanation for this effect. We have seen that birth weight depends on mother’s age, and we might expect that mothers of first babies are younger than others.
# 
# With a few calculations we can check whether this explanation is plausible. Then we’ll use multiple regression to investigate more carefully. First, let’s see how big the difference in weight is:

# In[13]:


diff_weight = firsts.totalwgt_lb.mean() - others.totalwgt_lb.mean()
diff_weight


# First babies are 0.125 lbs lighter, or 2 ounces. And the difference in ages:

# In[14]:


diff_age = firsts.agepreg.mean() - others.agepreg.mean()
diff_age


# The mothers of first babies are 3.59 years younger. Running the linear model again, we get the change in birth weight as a function of age:

# In[15]:


results = smf.ols('totalwgt_lb ~ agepreg', data=live).fit()
slope = results.params['agepreg']
slope


# The slope is 0.0175 pounds per year. If we multiply the slope by the difference in ages, we get the expected difference in birth weight for first babies and others, due to mother’s age:

# In[16]:


slope * diff_age


# The result is 0.063, just about half of the observed difference. So we conclude, tentatively, that the observed difference in birth weight can be partly explained by the difference in mother’s age.
# 
# Using multiple regression, we can explore these relationships more systematically.

# In[17]:


live['isfirst'] = live.birthord == 1
formula = 'totalwgt_lb ~ isfirst'
results = smf.ols(formula, data=live).fit()
regression.SummarizeResults(results)


# How do you interpret the beta observed on the model?

# In[18]:


# Answer here

# When the babies are the first ones the totalwgt_lb trend to be lower by -0.125


# The first line creates a new column named isfirst that is True for first babies and false otherwise. Then we fit a model using isfirst as an explanatory variable.
# 
# Because isfirst is a boolean, ols treats it as a **categorical variable**, which means that the values fall into categories, like True and False, and should not be treated as numbers. The estimated parameter is the effect on birth weight when isfirst is true, so the result, -0.125 lbs, is the difference in birth weight between first babies and others.
# 
# The slope and the intercept are statistically significant, which means that they were unlikely to occur by chance, but the R 2 value for this model is small, which means that isfirst doesn’t account for a substantial part of
# the variation in birth weight.
# 
# The results are similar with agepreg:

# In[19]:


live['isfirst'] = live.birthord == 1
formula = 'totalwgt_lb ~ agepreg'
results = smf.ols(formula, data=live).fit()
regression.SummarizeResults(results)


# Again, the parameters are statistically significant, but R 2 is low.
# These models confirm results we have already seen. But now we can fit a single model that includes both variables.
# With the formula totalwgt_lb ~ isfirst + agepreg, we get:

# In[20]:


formula = 'totalwgt_lb ~ isfirst + agepreg'
results = smf.ols(formula, data=live).fit()
regression.SummarizeResults(results)


# In the combined model, the parameter for isfirst is smaller by about half, which means that part of the apparent effect of isfirst is actually accounted for by agepreg. And the p-value for isfirst is about 2.5%, which is on the
# border of statistical significance.
# 
# R2 for this model is a little higher, which indicates that the two variables together account for more variation in birth weight than either alone (but not by much).
# 
# Look at the pregnancy data and pick 5 variables that you think will be good at predicting lenght of the pregnancy. Create a linear regression and fit it, assesing the predicting power of this variables.

# In[21]:


# Code it here
formula = 'prglngth ~ isfirst + agepreg + pregordr + totalwgt_lb + cmprgbeg'
results = smf.ols(formula, data=live).fit()
regression.SummarizeResults(results)


# ## 3. Nonlinear relationships

# Remembering that the contribution of agepreg might be nonlinear, we might consider adding a variable to capture more of this relationship. One option is to create a column, agepreg2, that contains the squares of the ages:

# In[22]:


live['agepreg2'] = live.agepreg**2
formula = 'totalwgt_lb ~ isfirst + agepreg + agepreg2'


# Now by estimating parameters for agepreg and agepreg2, we are effectively fitting a parabola:

# In[23]:


results = smf.ols(formula, data=live).fit()
regression.SummarizeResults(results)


# The parameter of agepreg2 is negative, so the parabola curves downward. 
# 
# The quadratic model of agepreg accounts for more of the variability in birth weight; the parameter for isfirst is smaller in this model, and no longer statistically significant.
# 
# Using computed variables like agepreg2 is a common way to fit polynomials and other functions to data. This process is still considered linear regression, because the dependent variable is a linear function of the explanatory variables, regardless of whether some variables are nonlinear functions of others.
# 
# 
# The following table summerizes the results:
# 
# ![alt text](Resources/Think_Stats/notebookpics/modelsfit.png "Title")
# 
# Each entry is an estimated parameter and either a p-value in parentheses or an asterisk to indicate a p-value less that 0.001.
# 
# We conclude that the apparent difference in birth weight is explained, at least in part, by the difference in mother’s age. When we include mother’s age in the model, the effect of isfirst gets smaller, and the remaining effect might be due to chance.
# 
# In this example, mother’s age acts as a **control variable**; including agepreg in the model “controls for” the difference in age between first-time mothers and others, making it possible to isolate the effect (if any) of isfirst.

# ## 4. Data mining

# So far we have used regression models for explanation; for example, in the previous section we discovered that an apparent difference in birth weight is actually due to a difference in mother’s age. But the R2 values of those
# models is very low, which means that they have little predictive power. In this section we’ll try to do better.
# 
# Suppose one of your co-workers is expecting a baby and there is an office pool to guess the baby’s birth weight (if you are not familiar with betting pools, see https://en.wikipedia.org/wiki/Betting_pool).
# 
# Now suppose that you really want to win the pool. What could you do to improve your chances? Well, the NSFG dataset includes 244 variables about each pregnancy and another 3087 variables about each respondent. Maybe some of those variables have predictive power. To find out which ones are most useful, why not try them all?
# 
# Testing the variables in the pregnancy table is easy, but in order to use the variables in the respondent table, we have to match up each pregnancy with a respondent. In theory we could iterate through the rows of the pregnancy
# table, use the caseid to find the corresponding respondent, and copy the values from the correspondent table into the pregnancy table. But that would be slow.
# 
# A better option is to recognize this process as a join operation as defined in SQL and other relational database languages (see https://en.wikipedia.org/wiki/Join_(SQL)). Join is implemented as a DataFrame method, so we can perform the operation like this:

# In[24]:


from Resources.Think_Stats.Thinkstats2 import nsfg


# In[25]:


live = live[live.prglngth>30]
resp = nsfg.ReadFemResp()
resp.index = resp.caseid
join = live.join(resp, on='caseid', rsuffix='_r')


# The first line selects records for pregnancies longer than 30 weeks, assuming that the office pool is formed several weeks before the due date.
# 
# The next line reads the respondent file. The result is a DataFrame with integer indices; in order to look up respondents efficiently, I replace resp.index with resp.caseid.
# 
# The join method is invoked on live, which is considered the “left” table, and passed resp, which is the “right” table. The keyword argument on indicates the variable used to match up rows from the two tables.
# 
# In this example some column names appear in both tables, so we have to provide rsuffix, which is a string that will be appended to the names of overlapping columns from the right table. For example, both tables have a column named race that encodes the race of the respondent. The result of the join contains two columns named race and race_r.
# 
# The pandas implementation is fast. Joining the NSFG tables takes less than a second on an ordinary desktop computer. Now we can start testing variables.

# In[26]:


t = []
for name in join.columns:
    try:
        if join[name].var() < 1e-7:
            continue
            
        formula = 'totalwgt_lb ~ agepreg + ' + name
        model = smf.ols(formula, data=join)
        
        if model.nobs < len(join)/2:
            continue
            
        results = model.fit()
        
    except (ValueError, TypeError):
        continue
        
    t.append((results.rsquared, name))


# In[27]:


pd.DataFrame(t).sort_values(0, ascending=False)


# For each variable we construct a model, compute R 2 , and append the results to a list. The models all include agepreg, since we already know that it has some predictive power.
# 
# I check that each explanatory variable has some variability; otherwise the results of the regression are unreliable. I also check the number of observations for each model. Variables that contain a large number of nans are not good candidates for prediction.
# 
# For most of these variables, we haven’t done any cleaning. Some of them are encoded in ways that don’t work very well for linear regression. As a result, we might overlook some variables that would be useful if they were cleaned
# properly. But maybe we will find some good candidates.

# ## 5. Prediction

# The next step is to sort the results and select the variables that yield the highest values of R^2.

# In[28]:


t.sort(reverse=True)
for mse, name in t[:30]:
    print(name, mse)


# The first variable on the list is totalwgt_lb, followed by birthwgt_lb. Obviously, we can’t use birth weight to predict birth weight.
# 
# Similarly prglngth has useful predictive power, but for the office pool we assume pregnancy length (and the related variables) are not known yet.
# 
# The first useful predictive variable is babysex which indicates whether the baby is male or female. In the NSFG dataset, boys are about 0.3 lbs heavier. So, assuming that the sex of the baby is known, we can use it for prediction.
# 
# Next is race, which indicates whether the respondent is white, black, or other. As an explanatory variable, race can be problematic. In datasets like the NSFG, race is correlated with many other variables, including income
# and other socioeconomic factors. In a regression model, race acts as a **proxy variable**, so apparent correlations with race are often caused, at least in part, by other factors.
# 
# The next variable on the list is nbrnaliv, which indicates whether the pregnancy yielded multiple births. Twins and triplets tend to be smaller than other babies, so if we know whether our hypothetical co-worker is expecting twins, that would help.
# 
# Next on the list is paydu, which indicates whether the respondent owns her home. It is one of several income-related variables that turn out to be predictive. In datasets like the NSFG, income and wealth are correlated
# with just about everything. In this example, income is related to diet, health, health care, and other factors likely to affect birth weight.
# 
# Some of the other variables on the list are things that would not be known until later, like bfeedwks, the number of weeks the baby was breast fed. We can’t use these variables for prediction, but you might want to speculate on
# reasons bfeedwks might be correlated with birth weight.
# 
# Sometimes you start with a theory and use data to test it. Other times you start with data and go looking for possible theories. The second approach, which this section demonstrates, is called **data mining**. An advantage of
# data mining is that it can discover unexpected patterns. A hazard is that many of the patterns it discovers are either random or spurious.
# 
# Having identified potential explanatory variables, I tested a few models and settled on this one:

# In[29]:


formula = ('totalwgt_lb ~ agepreg + C(race) + babysex==1 + '
            'nbrnaliv>1 + paydu==1 + totincr')
results = smf.ols(formula, data=join).fit()


# This formula uses some syntax we have not seen yet: C(race) tells the formula parser (Patsy) to treat race as a categorical variable, even though it is encoded numerically.
# 
# The encoding for babysex is 1 for male, 2 for female; writing babysex==1 converts it to boolean, True for male and false for female.
# 
# Similarly nbrnaliv>1 is True for multiple births and paydu==1 is True for respondents who own their houses.
# 
# totincr is encoded numerically from 1-14, with each increment representing about 5000 in annual income. So we can treat these values as numerical, expressed in units of 5000. 
# 
# Here are the results of the model:

# In[30]:


regression.SummarizeResults(results)


# The estimated parameters for race are larger than I expected, especially since we control for income. The encoding is 1 for black, 2 for white, and 3 for other. Babies of black mothers are lighter than babies of other races by
# 0.27–0.36 lbs.
# 
# As we’ve already seen, boys are heavier by about 0.3 lbs; twins and other multiplets are lighter by 1.4 lbs.
# 
# People who own their homes have heavier babies by about 0.12 lbs, even when we control for income. The parameter for mother’s age is smaller than what we saw in Section 11.2, which suggests that some of the other variables
# are correlated with age, probably including paydu and totincr.
# 
# All of these variables are statistically significant, some with very low p-values, but R 2 is only 0.06, still quite small. RMSE without using the model is 1.27 lbs; with the model it drops to 1.23. So your chance of winning the pool is not substantially improved. Sorry!

# ## 6. Logistic regression

# In the previous examples, some of the explanatory variables were numerical and some categorical (including boolean). But the dependent variable was always numerical.
# 
# Linear regression can be generalized to handle other kinds of dependent variables. If the dependent variable is boolean, the generalized model is called **logistic regression**. If the dependent variable is an integer count, it’s called **Poisson regression**.
# 
# As an example of logistic regression, let’s consider a variation on the office pool scenario. Suppose a friend of yours is pregnant and you want to predict whether the baby is a boy or a girl. You could use data from the NSFG to
# find factors that affect the “sex ratio”, which is conventionally defined to be the probability of having a boy.
# 
# If you encode the dependent variable numerically, for example 0 for a girl and 1 for a boy, you could apply ordinary least squares, but there would be problems. The linear model might be something like this:
# 
# ![alt text](Resources/Think_Stats/notebookpics/logformula.png "Title")
# 
# Where y is the dependent variable, and x 1 and x 2 are explanatory variables. Then we could find the parameters that minimize the residuals.
# 
# The problem with this approach is that it produces predictions that are hard to interpret. Given estimated parameters and values for x 1 and x 2 , the model might predict y = 0.5, but the only meaningful values of y are 0 and 1.
# 
# It is tempting to interpret a result like that as a probability; for example, we might say that a respondent with particular values of x 1 and x 2 has a 50% chance of having a boy. But it is also possible for this model to predict y = 1.1 or y = −0.1, and those are not valid probabilities.
# 
# Logistic regression avoids this problem by expressing predictions in terms of odds rather than probabilities. If you are not familiar with odds, “odds in favor” of an event is the ratio of the probability it will occur to the probability that it will not.
# 
# So if I think my team has a 75% chance of winning, I would say that the odds in their favor are three to one, because the chance of winning is three times the chance of losing.
# 
# Odds and probabilities are different representations of the same information. Given a probability, you can compute the odds like this:

# In[31]:


p = .75


# In[32]:


o = p / (1-p)


# Given odds in favor, you can convert to probability like this:

# In[33]:


p = o / (o+1)


# Logistic regression is based on the following model:
# 
# ![alt text](Resources/Think_Stats/notebookpics/logformula2.png "Title")    
# 
# Where o is the odds in favor of a particular outcome; in the example, o would be the odds of having a boy.
# 
# Suppose we have estimated the parameters β 0 , β 1 , and β 2 (I’ll explain how in a minute). And suppose we are given values for x 1 and x 2 . We can compute the predicted value of log o, and then convert to a probability:

# In[34]:


o = np.exp(np.log(o))
p = o / (o+1)


# In[35]:


p


# So in the office pool scenario we could compute the predictive probability of having a boy. But how do we estimate the parameters?

# ## 7. Estimating Parameters

# Unlike linear regression, logistic regression does not have a closed form solution, so it is solved by guessing an initial solution and improving it iteratively.
# 
# The usual goal is to find the maximum-likelihood estimate (MLE), which is the set of parameters that maximizes the likelihood of the data. For example, suppose we have the following data:

# In[36]:


y = np.array([0, 1, 0, 1])
x1 = np.array([0, 0, 0, 1])
x2 = np.array([0, 1, 1, 1])


# And we start with the initial guesses β 0 = −1.5, β 1 = 2.8, and β 2 = 1.1:

# In[37]:


beta = [-1.5, 2.8, 1.1]


# Then for each row we can compute log_o:

# In[38]:


log_o = beta[0] + beta[1] * x1 + beta[2] * x2
log_o


# And convert from log odds to probabilities:

# In[39]:


o = np.exp(log_o)
o


# In[40]:


p = o / (o+1)
p


# Notice that when log_o is greater than 0, o is greater than 1 and p is greater than 0.5.
# 
# The likelihood of an outcome is p when y==1 and 1-p when y==0. For example, if we think the probability of a boy is 0.8 and the outcome is a boy, the likelihood is 0.8; if the outcome is a girl, the likelihood is 0.2. We can compute that like this:

# In[41]:


likes = y * p + (1-y) * (1-p)
likes


# The overall likelihood of the data is the product of likes:

# In[42]:


like = np.prod(likes)
like


# For these values of beta, the likelihood of the data is 0.18. The goal of logistic regression is to find parameters that maximize this like-
# lihood. To do that, most statistics packages use an iterative solver like Newton’s method (see https://en.wikipedia.org/wiki/Logistic_regression#Model_fitting).

# ## 8. Implementation
# 

# StatsModels provides an implementation of logistic regression called logit, named for the function that converts from probability to log odds. To demonstrate its use, I’ll look for variables that affect the sex ratio. Again, I load the NSFG data and select pregnancies longer than 30 weeks:

# In[43]:


live, firsts, others = first.MakeFrames()
df = live[live.prglngth>30]


# logit requires the dependent variable to be binary (rather than boolean), so I create a new column named boy, using astype(int) to convert to binary integers:

# In[44]:


df.loc[:,'boy'] = (df.babysex==1).astype(int)


# Factors that have been found to affect sex ratio include parents’ age, birth order, race, and social status. We can use logistic regression to see if these effects appear in the NSFG data. I’ll start with the mother’s age:

# In[45]:


import statsmodels.formula.api as smf

model = smf.logit('boy ~ agepreg', data=df)
results = model.fit()
regression.SummarizeResults(results)


# How do you interpret this beta1 coeficient?

# In[46]:


# Answer here:
# The beta1 coeficient is 0.001 that means that with 1 year more in agepreg the baby have e^0.001 more chances of being a boy


# logit takes the same arguments as ols, a formula in Patsy syntax and a DataFrame. The result is a Logit object that represents the model. It contains attributes called endog and exog that contain the **endogenous varible**, another name for the dependent variable, and the **exogenous variables**, another name for the explanatory variables. Since they are NumPy arrays, it is sometimes convenient to convert them to DataFrames:

# In[47]:


endog = pd.DataFrame(model.endog, columns=[model.endog_names])
exog = pd.DataFrame(model.exog, columns=model.exog_names)


# The result of model.fit is a BinaryResults object, which is similar to the RegressionResults object we got from ols. Here is a summary of the results:

# In[48]:


model.fit()


# The parameter of agepreg is positive, which suggests that older mothers are more likely to have boys, but the p-value is 0.783, which means that the apparent effect could easily be due to chance.
# 
# The coefficient of determination, R 2 , does not apply to logistic regression, but there are several alternatives that are used as “pseudo R 2 values.” These values can be useful for comparing models. For example, here’s a model that includes several factors believed to be associated with sex ratio:

# In[49]:


formula = 'boy ~ agepreg + hpagelb + birthord + C(race)'
df = df.dropna(subset=['boy'])
model = smf.logit(formula, data=df)
results = model.fit()


# Along with mother’s age, this model includes father’s age at birth (hpagelb), birth order (birthord), and race as a categorical variable. Here are the results:

# In[50]:


regression.SummarizeResults(results)


# None of the estimated parameters are statistically significant. The pseudo-R 2 value is a little higher, but that could be due to chance.
# 
# Fit now a model with the same variables that tries to predict whether a baby is black or white.

# In[51]:


# Code it here


# ## 9. Accuracy

# In the office pool scenario, we are most interested in the accuracy of the model: the number of successful predictions, compared with what we would expect by chance.
# 
# In the NSFG data, there are more boys than girls, so the baseline strategy is to guess “boy” every time. The accuracy of this strategy is just the fraction of boys:

# In[62]:


formula = 'boy ~ agepreg + hpagelb + birthord + C(race)'
df = df.dropna(subset=['boy'])
model = smf.logit(formula, data=df)
results = model.fit()


# In[63]:


endog = pd.DataFrame(model.endog, columns=[model.endog_names])
exog = pd.DataFrame(model.exog, columns=model.exog_names)


# In[64]:


actual = endog['boy'].dropna()
baseline = actual.mean()


# In[65]:


baseline


# Since actual is encoded in binary integers, the mean is the fraction of boys, which is 0.507.
# 
# Here’s how we compute the accuracy of the model:

# In[66]:


predict = (results.predict() >= 0.5)
true_pos = predict * actual
true_neg = (1 - predict) * (1 - actual)


# results.predict returns a NumPy array of probabilities, which we round off to 0 or 1. Multiplying by actual yields 1 if we predict a boy and get it right, 0 otherwise. So, true_pos indicates “true positives”.
# 
# Similarly, true_neg indicates the cases where we guess “girl” and get it right. Accuracy is the fraction of correct guesses:

# In[67]:


acc = (sum(true_pos) + sum(true_neg)) / len(actual)


# In[68]:


acc


# The result is 0.512, slightly better than the baseline, 0.507. But, you should not take this result too seriously. We used the same data to build and test the model, so the model may not have predictive power on new data.
# 
# Nevertheless, let’s use the model to make a prediction for the office pool. Suppose your friend is 35 years old and white, her husband is 39, and they are expecting their third child:

# In[58]:


columns = ['agepreg', 'hpagelb', 'birthord', 'race']
new = pd.DataFrame([[35, 39, 3, 2]], columns=columns)
y = results.predict(new)


# In[59]:


y


# To invoke results.predict for a new case, you have to construct a DataFrame with a column for each variable in the model. The result in this case is 0.52, so you should guess “boy.” But if the model improves your chances of winning, the difference is very small.
# 
# What is the accuracy of your previous model fitted to predict race?

# In[60]:


# Code it here


# Predict the race of your friend's child, who is 35 years old and white, her husband is 39, and they are expecting their third child that they know is a girl.

# In[61]:


# Code it here

