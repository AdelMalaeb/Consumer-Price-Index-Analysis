# Simple linear regression using 1 Explanatory Variable

* **Task 1:** I will choose one Variable that I believe that it mostly affects the CPI index based on extensive Expalanatory Data Analysis.

* **Task 2:** I will deploy a simple linear regression model and evaluate the results.




```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
```


```python
data = pd.read_csv("/Users/adel/Desktop/Issachar Technologies/Cpi-Project-Updated/CPI-Project 2/Cpi-Compiled-Data-1990.csv")
```

##  Data exploration

### Feature Explanation - Data extracted from - https://fred.stlouisfed.org/searchresults/?st=cpi&isTst=1
- Target: CPIAUCSL - (CPIAUCSL) is a price index of a basket of goods and services paid by urban consumers.
- Feature 1: CUSR0000SETG01 - Airline Fares in U.S. City Average
- Feature 2: CUSR0000SAF116 - Alcoholic Beverages in U.S. City Average
- Feature 3: CPIAPPSL - Apparel in U.S. City Average
- Feature 4: CUSR0000SAD - Durables in U.S. City Average 
- Feature 5: CUSR0000SEHF01 - Electricity in U.S. City Average
- Feature 6: CPIENGSL - Energy in U.S. City Average 
- Feature 7: CPIUFDSL - Food in U.S. City Average 
- Feature 8: CUSR0000SEHE - Fuel Oil & Other Fuels in U.S. City Average 
- Feature 9: CUSR0000SETB01 - Gasoline in U.S. City Average 
- Feature 10: CPIHOSSL - Housing in U.S. City Average 
- Feature 11: CPIMEDSL - Medical Care in U.S. City Average 
- Feature 12: CUSR0000SAM1 - Medical Care Commodities in U.S. City Average 
- Feature 13: CUSR0000SETA01 - New Vihicles in U.S. City Average 
- Feature 14: CUUR0000SA0R - Purchasing Power in U.S. City Average 
- Feature 15: CUSR0000SEHA - Rent in U.S. City Average
- Feature 16: CUSR0000SAH1 - Shelter in U.S. City Average
- Feature 17: CPITRNSL - Transportation in U.S. City Average
- Feature 18: CUSR0000SETA02 - Used Cars & Trucks in U.S. City Average
    
Data Range: 1990-01-01 - 2022-09-01


```python
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


```


```python
#Rename the column names
data = data.rename(columns = colmn_dict)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>CPI</th>
      <th>Airline_Fares</th>
      <th>Alcoholic_Beverages</th>
      <th>Apparel</th>
      <th>Durables</th>
      <th>Electricity</th>
      <th>Energy</th>
      <th>Food</th>
      <th>Fuel_Oil</th>
      <th>Gasoline</th>
      <th>Housing</th>
      <th>Medical_Care</th>
      <th>Medical_Care_Commodities</th>
      <th>New_Vehicles</th>
      <th>Purchasing_Power</th>
      <th>Rent</th>
      <th>Shelter</th>
      <th>Transportation</th>
      <th>Used_Cars_Trucks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1990-01-01</td>
      <td>127.500</td>
      <td>134.800</td>
      <td>126.700</td>
      <td>119.900</td>
      <td>113.300</td>
      <td>116.000</td>
      <td>98.900</td>
      <td>129.700</td>
      <td>110.600</td>
      <td>92.900</td>
      <td>126.100</td>
      <td>156.000</td>
      <td>157.300</td>
      <td>121.300</td>
      <td>78.5</td>
      <td>135.800</td>
      <td>136.300</td>
      <td>117.000</td>
      <td>119.200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1990-02-01</td>
      <td>128.000</td>
      <td>138.200</td>
      <td>127.000</td>
      <td>122.000</td>
      <td>113.400</td>
      <td>117.200</td>
      <td>98.200</td>
      <td>130.800</td>
      <td>92.800</td>
      <td>93.000</td>
      <td>126.200</td>
      <td>157.100</td>
      <td>158.700</td>
      <td>121.200</td>
      <td>78.2</td>
      <td>136.100</td>
      <td>136.600</td>
      <td>117.200</td>
      <td>118.700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990-03-01</td>
      <td>128.600</td>
      <td>141.000</td>
      <td>127.600</td>
      <td>123.800</td>
      <td>113.300</td>
      <td>117.100</td>
      <td>97.600</td>
      <td>131.000</td>
      <td>89.800</td>
      <td>92.300</td>
      <td>126.800</td>
      <td>158.300</td>
      <td>159.700</td>
      <td>120.900</td>
      <td>77.7</td>
      <td>136.700</td>
      <td>137.600</td>
      <td>117.300</td>
      <td>118.500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1990-04-01</td>
      <td>128.900</td>
      <td>144.100</td>
      <td>127.900</td>
      <td>124.100</td>
      <td>113.200</td>
      <td>117.800</td>
      <td>97.500</td>
      <td>130.800</td>
      <td>88.400</td>
      <td>92.800</td>
      <td>127.100</td>
      <td>159.600</td>
      <td>160.900</td>
      <td>120.800</td>
      <td>77.6</td>
      <td>137.200</td>
      <td>138.200</td>
      <td>117.700</td>
      <td>118.200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1990-05-01</td>
      <td>129.100</td>
      <td>145.900</td>
      <td>128.600</td>
      <td>124.000</td>
      <td>113.200</td>
      <td>117.500</td>
      <td>96.700</td>
      <td>131.100</td>
      <td>87.500</td>
      <td>91.700</td>
      <td>127.300</td>
      <td>160.800</td>
      <td>161.800</td>
      <td>120.900</td>
      <td>77.4</td>
      <td>137.600</td>
      <td>138.600</td>
      <td>117.500</td>
      <td>117.600</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>388</th>
      <td>2022-05-01</td>
      <td>291.474</td>
      <td>336.022</td>
      <td>272.413</td>
      <td>126.927</td>
      <td>127.541</td>
      <td>248.505</td>
      <td>308.839</td>
      <td>301.879</td>
      <td>508.909</td>
      <td>372.063</td>
      <td>297.881</td>
      <td>543.220</td>
      <td>386.273</td>
      <td>171.254</td>
      <td>34.2</td>
      <td>365.365</td>
      <td>350.418</td>
      <td>270.297</td>
      <td>207.518</td>
    </tr>
    <tr>
      <th>389</th>
      <td>2022-06-01</td>
      <td>295.328</td>
      <td>329.906</td>
      <td>273.553</td>
      <td>127.929</td>
      <td>128.476</td>
      <td>252.804</td>
      <td>332.087</td>
      <td>304.867</td>
      <td>505.445</td>
      <td>413.606</td>
      <td>300.290</td>
      <td>546.861</td>
      <td>387.787</td>
      <td>172.369</td>
      <td>33.7</td>
      <td>368.203</td>
      <td>352.550</td>
      <td>280.691</td>
      <td>210.863</td>
    </tr>
    <tr>
      <th>390</th>
      <td>2022-07-01</td>
      <td>295.271</td>
      <td>304.071</td>
      <td>274.889</td>
      <td>127.818</td>
      <td>128.864</td>
      <td>256.780</td>
      <td>316.955</td>
      <td>308.220</td>
      <td>464.597</td>
      <td>381.710</td>
      <td>301.639</td>
      <td>549.282</td>
      <td>390.077</td>
      <td>173.432</td>
      <td>33.8</td>
      <td>370.789</td>
      <td>354.449</td>
      <td>274.821</td>
      <td>209.998</td>
    </tr>
    <tr>
      <th>391</th>
      <td>2022-08-01</td>
      <td>295.620</td>
      <td>290.010</td>
      <td>275.861</td>
      <td>128.091</td>
      <td>129.490</td>
      <td>260.643</td>
      <td>301.045</td>
      <td>310.664</td>
      <td>453.418</td>
      <td>341.383</td>
      <td>304.109</td>
      <td>553.006</td>
      <td>391.032</td>
      <td>174.891</td>
      <td>33.8</td>
      <td>373.525</td>
      <td>356.894</td>
      <td>268.586</td>
      <td>209.782</td>
    </tr>
    <tr>
      <th>392</th>
      <td>2022-09-01</td>
      <td>296.761</td>
      <td>292.434</td>
      <td>275.760</td>
      <td>127.722</td>
      <td>129.338</td>
      <td>261.777</td>
      <td>294.705</td>
      <td>313.101</td>
      <td>440.818</td>
      <td>324.646</td>
      <td>306.323</td>
      <td>557.426</td>
      <td>390.677</td>
      <td>176.058</td>
      <td>33.7</td>
      <td>376.679</td>
      <td>359.567</td>
      <td>267.043</td>
      <td>207.532</td>
    </tr>
  </tbody>
</table>
<p>393 rows × 20 columns</p>
</div>



### Explore the data size


```python
data.shape
```




    (393, 20)



The dataset contains: 393 rows and 20 columns

### Explore the independant variables



```python
columns_to_describe = data.columns[data.columns != 'CPI']  # Exclude column 'CPIAUCSL' the depandant variable
description = data[columns_to_describe].describe()
description
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Airline_Fares</th>
      <th>Alcoholic_Beverages</th>
      <th>Apparel</th>
      <th>Durables</th>
      <th>Electricity</th>
      <th>Energy</th>
      <th>Food</th>
      <th>Fuel_Oil</th>
      <th>Gasoline</th>
      <th>Housing</th>
      <th>Medical_Care</th>
      <th>Medical_Care_Commodities</th>
      <th>New_Vehicles</th>
      <th>Purchasing_Power</th>
      <th>Rent</th>
      <th>Shelter</th>
      <th>Transportation</th>
      <th>Used_Cars_Trucks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.00000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
      <td>393.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>238.399369</td>
      <td>201.309555</td>
      <td>125.691906</td>
      <td>116.302552</td>
      <td>167.905405</td>
      <td>171.259359</td>
      <td>200.573812</td>
      <td>203.875186</td>
      <td>184.599746</td>
      <td>200.380761</td>
      <td>341.380402</td>
      <td>283.85943</td>
      <td>141.261249</td>
      <td>52.355471</td>
      <td>230.359458</td>
      <td>231.081402</td>
      <td>175.499270</td>
      <td>145.103952</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47.119050</td>
      <td>39.927305</td>
      <td>5.041742</td>
      <td>7.775405</td>
      <td>38.492435</td>
      <td>57.552504</td>
      <td>46.108228</td>
      <td>99.715001</td>
      <td>78.483103</td>
      <td>45.682776</td>
      <td>111.694236</td>
      <td>69.13051</td>
      <td>8.338731</td>
      <td>11.524731</td>
      <td>65.253775</td>
      <td>57.972697</td>
      <td>35.534301</td>
      <td>16.627765</td>
    </tr>
    <tr>
      <th>min</th>
      <td>134.800000</td>
      <td>126.700000</td>
      <td>114.389000</td>
      <td>103.428000</td>
      <td>116.000000</td>
      <td>96.700000</td>
      <td>129.700000</td>
      <td>82.900000</td>
      <td>85.100000</td>
      <td>126.100000</td>
      <td>156.000000</td>
      <td>157.30000</td>
      <td>120.800000</td>
      <td>33.700000</td>
      <td>135.800000</td>
      <td>136.300000</td>
      <td>117.000000</td>
      <td>115.400000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>201.841000</td>
      <td>164.900000</td>
      <td>120.500000</td>
      <td>110.720000</td>
      <td>129.300000</td>
      <td>110.100000</td>
      <td>159.700000</td>
      <td>98.300000</td>
      <td>104.400000</td>
      <td>159.200000</td>
      <td>239.400000</td>
      <td>218.30000</td>
      <td>137.300000</td>
      <td>42.300000</td>
      <td>170.300000</td>
      <td>180.300000</td>
      <td>143.500000</td>
      <td>137.544000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>239.100000</td>
      <td>200.600000</td>
      <td>125.843000</td>
      <td>114.700000</td>
      <td>167.900000</td>
      <td>183.500000</td>
      <td>194.100000</td>
      <td>225.164000</td>
      <td>180.800000</td>
      <td>202.300000</td>
      <td>335.000000</td>
      <td>285.90000</td>
      <td>142.300000</td>
      <td>49.600000</td>
      <td>223.700000</td>
      <td>231.000000</td>
      <td>176.000000</td>
      <td>144.850000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>274.246000</td>
      <td>236.853000</td>
      <td>130.400000</td>
      <td>124.200000</td>
      <td>206.088000</td>
      <td>216.710000</td>
      <td>243.235000</td>
      <td>280.850000</td>
      <td>240.407000</td>
      <td>233.695000</td>
      <td>435.864000</td>
      <td>344.14200</td>
      <td>146.099000</td>
      <td>61.700000</td>
      <td>276.663000</td>
      <td>270.891000</td>
      <td>205.698000</td>
      <td>151.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>336.022000</td>
      <td>275.861000</td>
      <td>135.100000</td>
      <td>129.700000</td>
      <td>261.777000</td>
      <td>332.087000</td>
      <td>313.101000</td>
      <td>508.909000</td>
      <td>413.606000</td>
      <td>306.323000</td>
      <td>557.426000</td>
      <td>391.03200</td>
      <td>176.058000</td>
      <td>78.500000</td>
      <td>376.679000</td>
      <td>359.567000</td>
      <td>280.691000</td>
      <td>213.173000</td>
    </tr>
  </tbody>
</table>
</div>



### Explore the dependant variable

**Check for Missing values in the depandant variable**


```python
data['CPI'].isna().mean()
```




    0.0



**Check for Missing values in the indepandant variables**


```python
data[columns_to_describe].isna().mean()
```




    DATE                        0.0
    Airline_Fares               0.0
    Alcoholic_Beverages         0.0
    Apparel                     0.0
    Durables                    0.0
     Electricity                0.0
    Energy                      0.0
    Food                        0.0
    Fuel_Oil                    0.0
    Gasoline                    0.0
    Housing                     0.0
    Medical_Care                0.0
    Medical_Care_Commodities    0.0
    New_Vehicles                0.0
    Purchasing_Power            0.0
    Rent                        0.0
    Shelter                     0.0
    Transportation              0.0
    Used_Cars_Trucks            0.0
    dtype: float64



The data I have contains 0 missing values 

### Visualise the distribution of dependant variable


```python
fig = sns.histplot(data['CPI'])

fig.set_xlabel("CPI")
fig.set_title("Distribution of CPI ")

plt.savefig('Distribution of CPI.pdf')
plt.show()
```


    
![png](output_18_0.png)
    


##  Building the Model


```python
sns.pairplot(data)
```




    <seaborn.axisgrid.PairGrid at 0x7fe9dc703820>




    
![png](output_20_1.png)
    


Feature 2: CUSR0000SAF116 - Alcoholic Beverages in U.S. City Average and CPI have a linear relationship

Alcoholic Beverages clearly has the strongest linear relationship with CPI. You could draw a straight line through the scatterplot of `Alcoholic Beverages` and `CPI` that confidently estimates `CPI` using `Alcoholic Beverages`.

### fit the model


```python
#define the ols formula
ols_formula = "CPI ~ Alcoholic_Beverages"

#fit the model
model = ols(formula = ols_formula, data = data)


```


```python
    model = model.fit()
```


```python
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>CPI</td>       <th>  R-squared:         </th> <td>   0.992</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.992</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>4.916e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 04 May 2024</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>12:07:06</td>     <th>  Log-Likelihood:    </th> <td> -1078.4</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   393</td>      <th>  AIC:               </th> <td>   2161.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   391</td>      <th>  BIC:               </th> <td>   2169.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>           <td>  -12.8654</td> <td>    0.979</td> <td>  -13.137</td> <td> 0.000</td> <td>  -14.791</td> <td>  -10.940</td>
</tr>
<tr>
  <th>Alcoholic_Beverages</th> <td>    1.0580</td> <td>    0.005</td> <td>  221.720</td> <td> 0.000</td> <td>    1.049</td> <td>    1.067</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>172.395</td> <th>  Durbin-Watson:     </th> <td>   0.047</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 858.205</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 1.854</td>  <th>  Prob(JB):          </th> <td>4.40e-187</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 9.218</td>  <th>  Cond. No.          </th> <td>1.06e+03</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.06e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### Check model assumptions

To justify using simple linear regression, check that the four linear regression assumptions are not violated. These assumptions are:

* Linearity
* Independent Observations
* Normality
* Homoscedasticity

## 1. Linearity

The linearity assumption requires a linear relationship between the independent and dependent variables. Check this assumption by creating a scatterplot comparing the independent variable with the dependent variable. 

Create a scatterplot comparing the X variable you selected with the dependent variable.


```python
sns.scatterplot(x='Alcoholic_Beverages', y='CPI', data = data)

plt.title("Scatter plot")
plt.xlabel("Alcoholic_Beverages")
plt.ylabel("CPI")

plt.savefig('Linearity assumption.pdf')
plt.show()
```


    
![png](output_29_0.png)
    


**Result:** Linearity Assumption is met

## 2. Normality

The normality assumption states that the errors are normally distributed.

Create two plots to check this assumption:

* **Plot 1**: Histogram of the residuals
* **Plot 2**: Q-Q plot of the residuals



```python
#get the residuals

residuals = model.resid
residuals
```




    0       6.311780
    1       6.494368
    2       6.459544
    3       6.442132
    4       5.901504
             ...    
    388    16.115600
    389    18.763435
    390    17.292893
    391    16.613478
    392    17.861340
    Length: 393, dtype: float64




```python
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
```


    
![png](output_33_0.png)
    


The histogram of residuals exhibits a normal distribution. QQ plot emphasizes on that as well.

## 3. Homoscedasticity

The **homoscedasticity (constant variance) assumption** is that the residuals have a constant variance for all values of `X`.

Check that this assumption is not violated by creating a scatterplot with the fitted values and residuals. Add a line at $y = 0$ to visualize the variance of residuals above and below $y = 0$.


```python
fitted_values = model.fittedvalues
fitted_values
```




    0      121.188220
    1      121.505632
    2      122.140456
    3      122.457868
    4      123.198496
              ...    
    388    275.358400
    389    276.564565
    390    277.978107
    391    279.006522
    392    278.899660
    Length: 393, dtype: float64




```python
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
```


    
![png](output_38_0.png)
    


**I noticed that this assumption does not fully hold!**

## Additional examination: Logarithmic Transformation of dependant variable


```python
data["CPI"].dtype
```




    dtype('float64')




```python
import numpy as np
```


```python
transformed_data_CPI = np.log(data["CPI"])
transformed_data_CPI
```




    0      4.848116
    1      4.852030
    2      4.856707
    3      4.859037
    4      4.860587
             ...   
    388    5.674951
    389    5.688087
    390    5.687894
    391    5.689075
    392    5.692927
    Name: CPI, Length: 393, dtype: float64




```python
#define the ols formula
ols_formula = "transformed_data_CPI ~ Alcoholic_Beverages"

#fit the model
model = ols(formula = ols_formula, data = data)
model = model.fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>transformed_data_CPI</td> <th>  R-squared:         </th> <td>   0.992</td> 
</tr>
<tr>
  <th>Model:</th>                     <td>OLS</td>         <th>  Adj. R-squared:    </th> <td>   0.992</td> 
</tr>
<tr>
  <th>Method:</th>               <td>Least Squares</td>    <th>  F-statistic:       </th> <td>4.851e+04</td>
</tr>
<tr>
  <th>Date:</th>               <td>Sat, 04 May 2024</td>   <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                   <td>12:07:07</td>       <th>  Log-Likelihood:    </th> <td>  992.56</td> 
</tr>
<tr>
  <th>No. Observations:</th>        <td>   393</td>        <th>  AIC:               </th> <td>  -1981.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>            <td>   391</td>        <th>  BIC:               </th> <td>  -1973.</td> 
</tr>
<tr>
  <th>Df Model:</th>                <td>     1</td>        <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>nonrobust</td>      <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>           <td>    4.1872</td> <td>    0.005</td> <td>  830.997</td> <td> 0.000</td> <td>    4.177</td> <td>    4.197</td>
</tr>
<tr>
  <th>Alcoholic_Beverages</th> <td>    0.0054</td> <td> 2.46e-05</td> <td>  220.254</td> <td> 0.000</td> <td>    0.005</td> <td>    0.005</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 6.783</td> <th>  Durbin-Watson:     </th> <td>   0.040</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.034</td> <th>  Jarque-Bera (JB):  </th> <td>   6.969</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.317</td> <th>  Prob(JB):          </th> <td>  0.0307</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.847</td> <th>  Cond. No.          </th> <td>1.06e+03</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.06e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
sns.scatterplot(x='Alcoholic_Beverages', y=transformed_data_CPI, data = data)

plt.title("Scatter plot")
plt.xlabel("Alcoholic_Beverages")
plt.ylabel("transformed_data_CPI")

plt.show()
```


    
![png](output_45_0.png)
    



```python
#get the residuals

residuals = model.resid
residuals
```




    0     -0.024316
    1     -0.022024
    2     -0.020593
    3     -0.019885
    4     -0.022120
             ...   
    388    0.014513
    389    0.021484
    390    0.014066
    391    0.009991
    392    0.014389
    Length: 393, dtype: float64




```python
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
```


    
![png](output_47_0.png)
    



```python
fitted_values = model.fittedvalues
fitted_values
```




    0      4.872432
    1      4.874055
    2      4.877300
    3      4.878922
    4      4.882707
             ...   
    388    5.660437
    389    5.666602
    390    5.673827
    391    5.679084
    392    5.678538
    Length: 393, dtype: float64




```python
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
```


    
![png](output_49_0.png)
    


Even after logarithmic transformation, the assumption is still not met.

**Options to consider:**

1-  Weighted Least Squares (WLS)

2-  Use Robust Standard Errors

3-  Use Generalized Least Squares (GLS)

4-  Non-Parametric Methods

# Multiple linear regression using at least two or more Explanatory Variables


```python
data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>CPI</th>
      <th>Airline_Fares</th>
      <th>Alcoholic_Beverages</th>
      <th>Apparel</th>
      <th>Durables</th>
      <th>Electricity</th>
      <th>Energy</th>
      <th>Food</th>
      <th>Fuel_Oil</th>
      <th>Gasoline</th>
      <th>Housing</th>
      <th>Medical_Care</th>
      <th>Medical_Care_Commodities</th>
      <th>New_Vehicles</th>
      <th>Purchasing_Power</th>
      <th>Rent</th>
      <th>Shelter</th>
      <th>Transportation</th>
      <th>Used_Cars_Trucks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1990-01-01</td>
      <td>127.5</td>
      <td>134.8</td>
      <td>126.7</td>
      <td>119.9</td>
      <td>113.3</td>
      <td>116.0</td>
      <td>98.9</td>
      <td>129.7</td>
      <td>110.6</td>
      <td>92.9</td>
      <td>126.1</td>
      <td>156.0</td>
      <td>157.3</td>
      <td>121.3</td>
      <td>78.5</td>
      <td>135.8</td>
      <td>136.3</td>
      <td>117.0</td>
      <td>119.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1990-02-01</td>
      <td>128.0</td>
      <td>138.2</td>
      <td>127.0</td>
      <td>122.0</td>
      <td>113.4</td>
      <td>117.2</td>
      <td>98.2</td>
      <td>130.8</td>
      <td>92.8</td>
      <td>93.0</td>
      <td>126.2</td>
      <td>157.1</td>
      <td>158.7</td>
      <td>121.2</td>
      <td>78.2</td>
      <td>136.1</td>
      <td>136.6</td>
      <td>117.2</td>
      <td>118.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990-03-01</td>
      <td>128.6</td>
      <td>141.0</td>
      <td>127.6</td>
      <td>123.8</td>
      <td>113.3</td>
      <td>117.1</td>
      <td>97.6</td>
      <td>131.0</td>
      <td>89.8</td>
      <td>92.3</td>
      <td>126.8</td>
      <td>158.3</td>
      <td>159.7</td>
      <td>120.9</td>
      <td>77.7</td>
      <td>136.7</td>
      <td>137.6</td>
      <td>117.3</td>
      <td>118.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1990-04-01</td>
      <td>128.9</td>
      <td>144.1</td>
      <td>127.9</td>
      <td>124.1</td>
      <td>113.2</td>
      <td>117.8</td>
      <td>97.5</td>
      <td>130.8</td>
      <td>88.4</td>
      <td>92.8</td>
      <td>127.1</td>
      <td>159.6</td>
      <td>160.9</td>
      <td>120.8</td>
      <td>77.6</td>
      <td>137.2</td>
      <td>138.2</td>
      <td>117.7</td>
      <td>118.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1990-05-01</td>
      <td>129.1</td>
      <td>145.9</td>
      <td>128.6</td>
      <td>124.0</td>
      <td>113.2</td>
      <td>117.5</td>
      <td>96.7</td>
      <td>131.1</td>
      <td>87.5</td>
      <td>91.7</td>
      <td>127.3</td>
      <td>160.8</td>
      <td>161.8</td>
      <td>120.9</td>
      <td>77.4</td>
      <td>137.6</td>
      <td>138.6</td>
      <td>117.5</td>
      <td>117.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airline_Fares</td>
      <td>553.05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alcoholic_Beverages</td>
      <td>25054.19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apparel</td>
      <td>6777.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Durables</td>
      <td>63216.45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Electricity</td>
      <td>3837.69</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Energy</td>
      <td>95038.52</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Food</td>
      <td>14698.56</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Fuel_Oil</td>
      <td>698.83</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Gasoline</td>
      <td>22344.94</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Housing</td>
      <td>1752375.15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Medical_Care</td>
      <td>18639.05</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Medical_Care_Commodities</td>
      <td>13471.76</td>
    </tr>
    <tr>
      <th>12</th>
      <td>New_Vehicles</td>
      <td>39286.09</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Purchasing_Power</td>
      <td>1854.27</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Rent</td>
      <td>84347.73</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Shelter</td>
      <td>1019613.85</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Transportation</td>
      <td>48349.28</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Used_Cars_Trucks</td>
      <td>3674.96</td>
    </tr>
  </tbody>
</table>
</div>



* VIF = 1: No multicollinearity. The variance of the regression coefficient of the variable is not inflated at all.

* VIF > 1: Indicates multicollinearity might be present. Typically, a VIF greater than 5 or 10 is considered high, suggesting significant multicollinearity.

* VIF > 5: Moderate multicollinearity.

* VIF > 10: High multicollinearity. The variance of the regression coefficient of the variable is significantly inflated by multicollinearity.

All features presents a large value for VIF which represents high multicollinearity

**However, I am going to proceed by selecting the two variables that has the lowest VIF and apply Multiple linear regression on these two explanatory variables**


```python
sorted_vif_data = vif_data.sort_values(by='VIF',ascending=True)
sorted_vif_data.iloc[:2,]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airline_Fares</td>
      <td>553.05</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Fuel_Oil</td>
      <td>698.83</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(data)
```




    <seaborn.axisgrid.PairGrid at 0x7fe9be953d00>




    
![png](output_56_1.png)
    


## Build the model


```python
# define the ols formula
ols_formula = "CPI ~ Airline_Fares + Fuel_Oil"

OLS = ols(formula = ols_formula, data=data)

model = OLS.fit()
```


```python
model_results = model.summary()
model_results
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>CPI</td>       <th>  R-squared:         </th> <td>   0.792</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.791</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   743.9</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 04 May 2024</td> <th>  Prob (F-statistic):</th> <td>7.95e-134</td>
</tr>
<tr>
  <th>Time:</th>                 <td>12:08:12</td>     <th>  Log-Likelihood:    </th> <td> -1721.1</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   393</td>      <th>  AIC:               </th> <td>   3448.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   390</td>      <th>  BIC:               </th> <td>   3460.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>  111.0416</td> <td>    6.290</td> <td>   17.654</td> <td> 0.000</td> <td>   98.675</td> <td>  123.408</td>
</tr>
<tr>
  <th>Airline_Fares</th> <td>    0.0757</td> <td>    0.037</td> <td>    2.041</td> <td> 0.042</td> <td>    0.003</td> <td>    0.149</td>
</tr>
<tr>
  <th>Fuel_Oil</th>      <td>    0.3484</td> <td>    0.018</td> <td>   19.878</td> <td> 0.000</td> <td>    0.314</td> <td>    0.383</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>16.933</td> <th>  Durbin-Watson:     </th> <td>   0.025</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  17.909</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.509</td> <th>  Prob(JB):          </th> <td>0.000129</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.243</td> <th>  Cond. No.          </th> <td>2.11e+03</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.11e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### Check for the assumptions

#### Linearity


```python
fig, axes = plt.subplots(1,2, figsize=(20,5))

#Create scatter plot for the first feature CUSR0000SETG01

sns.scatterplot(x=data["Airline_Fares"],y=data['CPI'], ax=axes[0])
axes[0].set_title("Airline_Fares & CPI")


#Create scatter plot for the first feature CUSR0000SEHE

sns.scatterplot(x=data["Fuel_Oil"],y=data['CPI'], ax=axes[1])
axes[1].set_title("Fuel_Oil & CPI")

plt.tight_layout()
plt.show

```




    <function matplotlib.pyplot.show(close=None, block=None)>




    
![png](output_62_1.png)
    


CUSR0000SETG01 and CPI seems to have a clear linear relationship. CUSR0000SEHE and CPI also seems to exhibit a linear relationship.

**Assumption is met**

#### Normality of residuals


```python
residuals = model.resid
residuals
```




    0     -32.284299
    1     -25.839864
    2     -24.406624
    3     -23.853574
    4     -23.476296
             ...    
    388   -22.326473
    389   -16.802425
    390    -0.670842
    391     4.637888
    392     9.985439
    Length: 393, dtype: float64




```python
fig, axes = plt.subplots(1,2, figsize=(20,5))

#Create histogram plot for the residuals

sns.histplot(residuals, ax=axes[0])
axes[0].set_title("Histogram of residuals")


#Create a qqplot for the residuals

sm.qqplot(residuals,line = 's', ax=axes[1])
axes[1].set_title("QQ plot")

plt.tight_layout()
plt.show

```




    <function matplotlib.pyplot.show(close=None, block=None)>




    
![png](output_66_1.png)
    


The residuals follow a normal distribution.

**Assumption is met**


#### Homoscedasticity


```python
fitted_values = model.fittedvalues

fig = sns.scatterplot(fitted_values, residuals)

fig.set_title("Homoscedasticity")
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
fig.axhline(0, color = 'r')

plt.show()
```

    /Users/adel/opt/anaconda3/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(



    
![png](output_69_1.png)
    


## Results and evaluation



```python
model_results
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>CPI</td>       <th>  R-squared:         </th> <td>   0.792</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.791</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   743.9</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 04 May 2024</td> <th>  Prob (F-statistic):</th> <td>7.95e-134</td>
</tr>
<tr>
  <th>Time:</th>                 <td>12:08:12</td>     <th>  Log-Likelihood:    </th> <td> -1721.1</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   393</td>      <th>  AIC:               </th> <td>   3448.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   390</td>      <th>  BIC:               </th> <td>   3460.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>  111.0416</td> <td>    6.290</td> <td>   17.654</td> <td> 0.000</td> <td>   98.675</td> <td>  123.408</td>
</tr>
<tr>
  <th>Airline_Fares</th> <td>    0.0757</td> <td>    0.037</td> <td>    2.041</td> <td> 0.042</td> <td>    0.003</td> <td>    0.149</td>
</tr>
<tr>
  <th>Fuel_Oil</th>      <td>    0.3484</td> <td>    0.018</td> <td>   19.878</td> <td> 0.000</td> <td>    0.314</td> <td>    0.383</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>16.933</td> <th>  Durbin-Watson:     </th> <td>   0.025</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  17.909</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.509</td> <th>  Prob(JB):          </th> <td>0.000129</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.243</td> <th>  Cond. No.          </th> <td>2.11e+03</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.11e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



The multiple linear regression equation is the following:

$\text{CPI} = 111.0416 + 0.0757*X_{Airlinesfares} + 0.3484*X_{fueloil}$

where: 
* $\beta_{0} = 111.0416$
* $\beta_{Airlinesfares} = 0.0757$
* $\beta_{fueloil} = 0.3484$

An increase in 1 unit of Airlines fares leads to an increase of 0.0757 in CPI

An increase in 1 unit of fuel oil leads to an increase of 0.3484 in CPI

* $\text{P-value}_{Airlines fares} = 0.042$,	Statistically significant given a 5% significance level
* $\text{P-value}_{fuel oil} = 0.000$,	Statistically significant given a 5% significance level

## Considerations


Eventhough It might seem that the following multiple regression model behaves well given the statistical significance of the coefficients and the evaluation metric **Adjusted R2** which states that 79.1% of the variation in CPI is explained by Airline fares and fuel oil. 

Applying this model to this data will **NOT** resemble a good prediction modeling for the CPI because: 

* I ignored almost all features that might have an affect on CPI.
* According to multicollinearity analysis, the features exhibits a very high VIF corresponding to high multicollinearity.
* One of the assumptions for MLR, is invalid.


**Possible Solutions:** Several techniques can be employed to mitigate the effects of multicollinearity:

* Principal Component Analysis (PCA): Use PCA to reduce the dimensionality of the data and address multicollinearity.
* Feature Selection: Choose a subset of relevant variables and exclude highly correlated variables.
* Ridge Regression or Lasso Regression: These regularization techniques can help in stabilizing the coefficients and reducing the impact of multicollinearity.
* Collecting More Data: Sometimes, collecting more data can help in reducing multicollinearity by providing a more diverse and representative dataset.


```python

```
