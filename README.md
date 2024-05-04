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

### Feature Explanation in details
* 1- CUSR0000SETG01: Airline Fares in U.S. City Average:
This feature represents the average price of airline fares in U.S. cities. It measures the cost of air travel for consumers and can be influenced by factors such as fuel prices, demand for air travel, competition among airlines, and government regulations.

* 2- CUSR0000SAF116: Alcoholic Beverages in U.S. City Average:
This feature represents the average price of alcoholic beverages in U.S. cities. It includes the prices of various types of alcoholic drinks, such as beer, wine, and spirits. Factors influencing alcoholic beverage prices may include production costs, taxes, distribution channels, and consumer preferences.

* 3- CPIAPPSL: Apparel in U.S. City Average:
This feature represents the average price of apparel (clothing and footwear) in U.S. cities. It includes the prices of clothing items, shoes, and accessories. Factors affecting apparel prices may include material costs, labor costs, fashion trends, and import/export tariffs.

* 4- CUSR0000SAD: Durables in U.S. City Average:
This feature represents the average price of durable goods in U.S. cities. Durable goods are products that have a long lifespan and are not consumed immediately. Examples include appliances, furniture, electronics, and vehicles. Prices of durables may be influenced by factors such as manufacturing costs, technological advancements, and consumer demand.

* 5- CUSR0000SEHF01: Electricity in U.S. City Average:
This feature represents the average price of electricity in U.S. cities. It measures the cost of electrical energy consumed by households and businesses. Factors influencing electricity prices may include production methods (e.g., coal, natural gas, renewables), infrastructure investments, regulatory policies, and market dynamics.

* 6- CPIENGSL: Energy in U.S. City Average:
This feature represents the average price of energy products in U.S. cities. It includes prices of various forms of energy, such as electricity, natural gas, heating oil, and gasoline. Energy prices are influenced by factors such as global supply and demand, geopolitical events, weather conditions, and government policies.

* 7- CPIUFDSL: Food in U.S. City Average:
This feature represents the average price of food items in U.S. cities. It includes prices of groceries, dining out, and other food-related expenses. Factors affecting food prices may include agricultural production, transportation costs, weather patterns, trade policies, and consumer preferences.

* 8- CUSR0000SEHE: Fuel Oil & Other Fuels in U.S. City Average:
This feature represents the average price of fuel oil and other fuels (excluding gasoline) in U.S. cities. It includes prices of heating oil, propane, and other types of fuel used for heating and energy purposes. Factors influencing fuel prices may include crude oil prices, refining costs, distribution networks, and seasonal demand variations.

* 9- CUSR0000SETB01: Gasoline in U.S. City Average:
This feature represents the average price of gasoline in U.S. cities. It measures the cost of fuel for vehicles and transportation. Gasoline prices are influenced by factors such as crude oil prices, refining costs, taxes, supply chain disruptions, and market competition.

* 10- CPIHOSSL: Housing in U.S. City Average:
This feature represents the average price of housing in U.S. cities. It includes prices of residential properties, rental rates, and housing-related expenses. Factors affecting housing prices may include location, property size, construction costs, mortgage rates, demographic trends, and government policies.

* 11- CPIMEDSL: Medical Care in U.S. City Average:
This feature represents the average price of medical care services in U.S. cities. It includes prices of healthcare services, medical procedures, prescription drugs, and health insurance premiums. Factors influencing medical care prices may include healthcare provider costs, pharmaceutical prices, insurance coverage, regulatory requirements, and technological advancements.

* 12- CUSR0000SAM1: Medical Care Commodities in U.S. City Average:
This feature represents the average price of medical care commodities (medical goods and supplies) in U.S. cities. It includes prices of medical equipment, devices, and supplies used in healthcare settings. Factors affecting medical care commodity prices may include production costs, research and development expenses, regulatory approvals, and market demand.

* 13- CUSR0000SETA01: New Vehicles in U.S. City Average:
This feature represents the average price of new vehicles (cars and trucks) in U.S. cities. It includes prices of automobiles sold by dealerships and manufacturers. Factors influencing new vehicle prices may include manufacturing costs, labor expenses, raw material prices, technological features, and consumer demand.

* 14- CUUR0000SA0R: Purchasing Power in U.S. City Average:
This feature represents the purchasing power of consumers in U.S. cities. It measures the ability of consumers to buy goods and services with their income. Purchasing power is influenced by factors such as income levels, inflation rates, employment trends, and government policies.

* 15- CUSR0000SEHA: Rent in U.S. City Average:
This feature represents the average rental prices for housing in U.S. cities. It includes rents paid by tenants for residential properties, apartments, and other rental accommodations. Rent prices may be influenced by factors such as location, property size, amenities, landlord policies, and housing market conditions.

* 16- CUSR0000SAH1: Shelter in U.S. City Average:
This feature represents the average price of shelter (housing accommodations) in U.S. cities. It includes costs associated with housing, such as rent, mortgage payments, property taxes, and utilities. Shelter prices may be affected by factors similar to those influencing housing prices, including location, property type, and market conditions.

* 17- CPITRNSL: Transportation in U.S. City Average:
This feature represents the average price of transportation services and expenses in U.S. cities. It includes costs associated with commuting, public transportation, vehicle maintenance, and other transportation-related expenditures. Transportation prices may be influenced by fuel prices, vehicle ownership costs, infrastructure investments, and commuting patterns.

* 18- CUSR0000SETA02: Used Cars & Trucks in U.S. City Average:
This feature represents the average price of used cars and trucks in U.S. cities. It includes prices of pre-owned vehicles sold by dealerships and private sellers. Factors influencing used vehicle prices may include vehicle age, mileage, condition, market demand, and depreciation rates.

These features represent various aspects of the economy and consumer spending habits, and analyzing their relationships with the target variable (CPIAUCSL) can provide insights into inflationary pressures and changes in the cost of living over time.
    
## Files Included:
* 1- CPI - Linear Regression Models
* 2- Pre-defined function for evaluating Linear Regression assumptions for CPI

