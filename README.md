# Consumer-Price-Index-Analysis
This repo contains different techniques for analyzing CPI. This work was done in my previous work experience with a Hedge Fund - Issachar Technologies

## Privacy 
All specifications regarding the data is automated through BigQuery on google cloud for Issachar Technologies; However, for representation purposes I compiled similar data extracted from FRED ECONOMIC DATA | ST. LOUIS FED

## Dataset
Data extracted from - https://fred.stlouisfed.org/searchresults/?st=cpi&isTst=1
Data Range: 1990-01-01 - 2022-09-01

### Feature Explanation 
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
    
## Files Included:
* 1- CPI - Linear Regression Models
* 2- Pre-defined function for evaluating Linear Regression assumptions for CPI

