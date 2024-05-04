#!/usr/bin/env python
# coding: utf-8

# # Simple linear regression using 1 Explanatory Variable
# 
# * **Task 1:** I will choose one Variable that I believe that it mostly affects the CPI index based on extensive Expalanatory Data Analysis.
# 
# * **Task 2:** I will deploy a simple linear regression model and evaluate the results.
# 
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[2]:


data = pd.read_csv("/Users/adel/Desktop/Issachar Technologies/Cpi-Project-Updated/CPI-Project 2/Cpi-Compiled-Data-1990.csv")


# ##  Data exploration
# 
# ### Feature Explanation - Data extracted from - https://fred.stlouisfed.org/searchresults/?st=cpi&isTst=1
# - Target: CPIAUCSL - (CPIAUCSL) is a price index of a basket of goods and services paid by urban consumers.
# - Feature 1: CUSR0000SETG01 - Airline Fares in U.S. City Average
# - Feature 2: CUSR0000SAF116 - Alcoholic Beverages in U.S. City Average
# - Feature 3: CPIAPPSL - Apparel in U.S. City Average
# - Feature 4: CUSR0000SAD - Durables in U.S. City Average 
# - Feature 5: CUSR0000SEHF01 - Electricity in U.S. City Average
# - Feature 6: CPIENGSL - Energy in U.S. City Average 
# - Feature 7: CPIUFDSL - Food in U.S. City Average 
# - Feature 8: CUSR0000SEHE - Fuel Oil & Other Fuels in U.S. City Average 
# - Feature 9: CUSR0000SETB01 - Gasoline in U.S. City Average 
# - Feature 10: CPIHOSSL - Housing in U.S. City Average 
# - Feature 11: CPIMEDSL - Medical Care in U.S. City Average 
# - Feature 12: CUSR0000SAM1 - Medical Care Commodities in U.S. City Average 
# - Feature 13: CUSR0000SETA01 - New Vihicles in U.S. City Average 
# - Feature 14: CUUR0000SA0R - Purchasing Power in U.S. City Average 
# - Feature 15: CUSR0000SEHA - Rent in U.S. City Average
# - Feature 16: CUSR0000SAH1 - Shelter in U.S. City Average
# - Feature 17: CPITRNSL - Transportation in U.S. City Average
# - Feature 18: CUSR0000SETA02 - Used Cars & Trucks in U.S. City Average
#     
# Data Range: 1990-01-01 - 2022-09-01

# In[3]:


#create a data dictionary to change the names of the column

colmn_dict = {"CPIAUCSL" : "CPI",
             "CUSR0000SETG01": "Airline_Fares",
             "CUSR0000SAF116": "Alcoholic_Beverages",
             "CPIAPPSL": "Apparel",
             "CUSR0000SAD": "Durables",
             "CUSR0000SEHF01":" Electricity",
             "CPIENGSL": "Energy",
             "CPIUFDSL": "Food",
             "CUSR0000SEHE":"Fuel_Oil",
             "CUSR0000SETB01":"Gasoline",
             "CPIHOSSL":"Housing",
             "CPIMEDSL":"Medical_Care",
             "CUSR0000SAM1":"Medical_Care_Commodities",
             "CUSR0000SETA01": "New_Vehicles",
             "CUUR0000SA0R": "Purchasing_Power",
             "CUSR0000SEHA": "Rent",
             "CUSR0000SAH1": "Shelter",
             "CPITRNSL":"Transportation",
             "CUSR0000SETA02":"Used_Cars_Trucks"}


# In[4]:


#Rename the column names
data = data.rename(columns = colmn_dict)
data


# ### Explore the data size

# In[5]:


data.shape


# The dataset contains: 393 rows and 20 columns

# ### Explore the independant variables
# 

# In[6]:


columns_to_describe = data.columns[data.columns != 'CPI']  # Exclude column 'CPIAUCSL' the depandant variable
description = data[columns_to_describe].describe()
description


# ### Explore the dependant variable

# **Check for Missing values in the depandant variable**

# In[7]:


data['CPI'].isna().mean()


# **Check for Missing values in the indepandant variables**

# In[8]:


data[columns_to_describe].isna().mean()


# The data I have contains 0 missing values 

# ### Visualise the distribution of dependant variable

# In[9]:


fig = sns.histplot(data['CPI'])

fig.set_xlabel("CPI")
fig.set_title("Distribution of CPI ")

plt.savefig('Distribution of CPI.pdf')
plt.show()


# ##  Building the Model

# In[10]:


sns.pairplot(data)


# Feature 2: CUSR0000SAF116 - Alcoholic Beverages in U.S. City Average and CPI have a linear relationship
# 
# Alcoholic Beverages clearly has the strongest linear relationship with CPI. You could draw a straight line through the scatterplot of `Alcoholic Beverages` and `CPI` that confidently estimates `CPI` using `Alcoholic Beverages`.

# ### fit the model

# In[11]:


#define the ols formula
ols_formula = "CPI ~ Alcoholic_Beverages"

#fit the model
model = ols(formula = ols_formula, data = data)


# In[12]:


model = model.fit()


# In[13]:


model.summary()


# ### Check model assumptions
# 
# To justify using simple linear regression, check that the four linear regression assumptions are not violated. These assumptions are:
# 
# * Linearity
# * Independent Observations
# * Normality
# * Homoscedasticity

# ## 1. Linearity

# The linearity assumption requires a linear relationship between the independent and dependent variables. Check this assumption by creating a scatterplot comparing the independent variable with the dependent variable. 
# 
# Create a scatterplot comparing the X variable you selected with the dependent variable.

# In[14]:


sns.scatterplot(x='Alcoholic_Beverages', y='CPI', data = data)

plt.title("Scatter plot")
plt.xlabel("Alcoholic_Beverages")
plt.ylabel("CPI")

plt.savefig('Linearity assumption.pdf')
plt.show()


# **Result:** Linearity Assumption is met

# ## 2. Normality
# 
# The normality assumption states that the errors are normally distributed.
# 
# Create two plots to check this assumption:
# 
# * **Plot 1**: Histogram of the residuals
# * **Plot 2**: Q-Q plot of the residuals
# 

# In[15]:


#get the residuals

residuals = model.resid
residuals


# In[16]:


# Create a figure and subplots grid using Matplotlib
fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # 1 rows, 2 column

# Plot 1: Histogram plot using Seaborn in the first subplot (axs[0])
sns.histplot(x=residuals, data=data, ax=axs[0])
axs[0].set_title('Histogram of residuals')

# Plot 2: QQ plot using Seaborn in the second subplot (axs[1])
sm.qqplot(residuals, ax=axs[1],line = 's')
axs[1].set_title('QQ plot of residuals')

# Adjust layout and display the figure
plt.tight_layout()

plt.savefig('Normality of residuals assumption.pdf')
plt.show()


# The histogram of residuals exhibits a normal distribution. QQ plot emphasizes on that as well.

# ## 3. Homoscedasticity

# The **homoscedasticity (constant variance) assumption** is that the residuals have a constant variance for all values of `X`.
# 
# Check that this assumption is not violated by creating a scatterplot with the fitted values and residuals. Add a line at $y = 0$ to visualize the variance of residuals above and below $y = 0$.

# In[17]:


fitted_values = model.fittedvalues
fitted_values


# In[18]:


sns.scatterplot(x=fitted_values, y=residuals)

# Set the x-axis label.
plt.xlabel("fitted_values")
# Set the y-axis label.
plt.ylabel("residuals")
# Set the title.
plt.title("Homoscedasticity Assupmtion")
# Add a line at y = 0 to visualize the variance of residuals above and below 0.
plt.axhline(y=0, color='red', linestyle='--')
 
plt.savefig("Homoscedasticity assupmtion.pdf")
plt.show()


# **I noticed that this assumption does not fully hold!**

# ## Additional examination: Logarithmic Transformation of dependant variable

# In[19]:


data["CPI"].dtype


# In[20]:


import numpy as np


# In[21]:


transformed_data_CPI = np.log(data["CPI"])
transformed_data_CPI


# In[22]:


#define the ols formula
ols_formula = "transformed_data_CPI ~ Alcoholic_Beverages"

#fit the model
model = ols(formula = ols_formula, data = data)
model = model.fit()
model.summary()


# In[23]:


sns.scatterplot(x='Alcoholic_Beverages', y=transformed_data_CPI, data = data)

plt.title("Scatter plot")
plt.xlabel("Alcoholic_Beverages")
plt.ylabel("transformed_data_CPI")

plt.show()


# In[24]:


#get the residuals

residuals = model.resid
residuals


# In[25]:


# Create a figure and subplots grid using Matplotlib
fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # 1 rows, 2 column

# Plot 1: Histogram plot using Seaborn in the first subplot (axs[0])
sns.histplot(x=residuals, data=data, ax=axs[0])
axs[0].set_title('Histogram of residuals')

# Plot 2: QQ plot using Seaborn in the second subplot (axs[1])
sm.qqplot(residuals, ax=axs[1],line = 's')
axs[1].set_title('QQ plot of residuals')

# Adjust layout and display the figure
plt.tight_layout()
plt.show()


# In[26]:


fitted_values = model.fittedvalues
fitted_values


# In[27]:


sns.scatterplot(x=fitted_values, y=residuals)

# Set the x-axis label.
plt.xlabel("fitted_values")
# Set the y-axis label.
plt.ylabel("residuals")
# Set the title.
plt.title("Homoscedasticity Assupmtion")
# Add a line at y = 0 to visualize the variance of residuals above and below 0.
plt.axhline(y=0, color='red', linestyle='--')
 

plt.show()


# Even after logarithmic transformation, the assumption is still not met.
# 
# **Options to consider:**
# 
# 1-  Weighted Least Squares (WLS)
# 
# 2-  Use Robust Standard Errors
# 
# 3-  Use Generalized Least Squares (GLS)
# 
# 4-  Non-Parametric Methods

# # Multiple linear regression using at least two or more Explanatory Variables

# In[28]:


data.head(5)


# In[29]:


# Calculate the variance inflation factor (optional).

from statsmodels.stats.outliers_influence import variance_inflation_factor

X = data.iloc[:,2:]
X


#Create a DataFrame to store VIF results
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns

# Calculate VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif_data["VIF"] = round(vif_data["VIF"],2)

vif_data


# * VIF = 1: No multicollinearity. The variance of the regression coefficient of the variable is not inflated at all.
# 
# * VIF > 1: Indicates multicollinearity might be present. Typically, a VIF greater than 5 or 10 is considered high, suggesting significant multicollinearity.
# 
# * VIF > 5: Moderate multicollinearity.
# 
# * VIF > 10: High multicollinearity. The variance of the regression coefficient of the variable is significantly inflated by multicollinearity.
# 
# All features presents a large value for VIF which represents high multicollinearity
# 
# **However, I am going to proceed by selecting the two variables that has the lowest VIF and apply Multiple linear regression on these two explanatory variables**

# In[30]:


sorted_vif_data = vif_data.sort_values(by='VIF',ascending=True)
sorted_vif_data.iloc[:2,]


# In[31]:


sns.pairplot(data)


# ## Build the model

# In[32]:


# define the ols formula
ols_formula = "CPI ~ Airline_Fares + Fuel_Oil"

OLS = ols(formula = ols_formula, data=data)

model = OLS.fit()


# In[33]:


model_results = model.summary()
model_results


# ### Check for the assumptions

# #### Linearity

# In[34]:


fig, axes = plt.subplots(1,2, figsize=(20,5))

#Create scatter plot for the first feature CUSR0000SETG01

sns.scatterplot(x=data["Airline_Fares"],y=data['CPI'], ax=axes[0])
axes[0].set_title("Airline_Fares & CPI")


#Create scatter plot for the first feature CUSR0000SEHE

sns.scatterplot(x=data["Fuel_Oil"],y=data['CPI'], ax=axes[1])
axes[1].set_title("Fuel_Oil & CPI")

plt.tight_layout()
plt.show


# CUSR0000SETG01 and CPI seems to have a clear linear relationship. CUSR0000SEHE and CPI also seems to exhibit a linear relationship.
# 
# **Assumption is met**

# #### Normality of residuals

# In[35]:


residuals = model.resid
residuals


# In[36]:


fig, axes = plt.subplots(1,2, figsize=(20,5))

#Create histogram plot for the residuals

sns.histplot(residuals, ax=axes[0])
axes[0].set_title("Histogram of residuals")


#Create a qqplot for the residuals

sm.qqplot(residuals,line = 's', ax=axes[1])
axes[1].set_title("QQ plot")

plt.tight_layout()
plt.show


# The residuals follow a normal distribution.
# 
# **Assumption is met**
# 

# #### Homoscedasticity

# In[37]:


fitted_values = model.fittedvalues

fig = sns.scatterplot(fitted_values, residuals)

fig.set_title("Homoscedasticity")
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
fig.axhline(0, color = 'r')

plt.show()


# ## Results and evaluation
# 

# In[38]:


model_results


# The multiple linear regression equation is the following:
# 
# $\text{CPI} = 111.0416 + 0.0757*X_{Airlinesfares} + 0.3484*X_{fueloil}$
# 
# where: 
# * $\beta_{0} = 111.0416$
# * $\beta_{Airlinesfares} = 0.0757$
# * $\beta_{fueloil} = 0.3484$
# 
# An increase in 1 unit of Airlines fares leads to an increase of 0.0757 in CPI
# 
# An increase in 1 unit of fuel oil leads to an increase of 0.3484 in CPI

# * $\text{P-value}_{Airlines fares} = 0.042$,	Statistically significant given a 5% significance level
# * $\text{P-value}_{fuel oil} = 0.000$,	Statistically significant given a 5% significance level

# ## Considerations
# 

# Eventhough It might seem that the following multiple regression model behaves well given the statistical significance of the coefficients and the evaluation metric **Adjusted R2** which states that 79.1% of the variation in CPI is explained by Airline fares and fuel oil. 
# 
# Applying this model to this data will **NOT** resemble a good prediction modeling for the CPI because: 
# 
# * I ignored almost all features that might have an affect on CPI.
# * According to multicollinearity analysis, the features exhibits a very high VIF corresponding to high multicollinearity.
# * One of the assumptions for MLR, is invalid.
# 

# **Possible Solutions:** Several techniques can be employed to mitigate the effects of multicollinearity:
# 
# * Principal Component Analysis (PCA): Use PCA to reduce the dimensionality of the data and address multicollinearity.
# * Feature Selection: Choose a subset of relevant variables and exclude highly correlated variables.
# * Ridge Regression or Lasso Regression: These regularization techniques can help in stabilizing the coefficients and reducing the impact of multicollinearity.
# * Collecting More Data: Sometimes, collecting more data can help in reducing multicollinearity by providing a more diverse and representative dataset.

# In[ ]:




