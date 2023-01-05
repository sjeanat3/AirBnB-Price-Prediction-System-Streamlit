# AirBnB-Price-Prediction-System

![AirBnB](./Images/airbnb4.jpeg)

## Overview / Business Problem
The target stakeholder is an AirBnB owner who owns property either in the cities of Asheville, Nashville, or Austin. AirBnB has provided a unique opportunity for homeowners to create a stream of income through their property. Prior experience with owning or renting real estate is not a requirement to list a home on AirBnb. As such, it is up to the discretion of the AirBnB lister to determine the daily price to charge. Listing a home for too high of a price could result in low demand and listing a price for too low could result in lost out potential income. The predictive modeling below will utilize a city’s past AirBnB listing data for the year of 2022. By utilizing this historical data, the model will predict prices for the 2023 calendar year based on attribute of the AirBnB owner’s home.

## Data Understanding
[The data](http://insideairbnb.com/get-the-data/) set comes from Inside AirBnB, a data sharing site devoted to collecting data on dozens of cities and countries around the world. There are two data sets for each city – a detailed calendar data set and a listings data set. Within the calendar data, there is 365 rows for each AirBnB listing, represent each day of the year and the price and other information. The breakdown of each city and the respective datasets is below:

-	**Asheville:**
    - Calendar Data:
        - 958,490 rows
        - 7 columns
    - Listings Data:
        - 2,626 rows
        - 74 columns
-	**Nashville:**
    - Calendar Data:
        - 2,320,689 rows
        - 7 columns
    - Listings Data:
        - 6,359 rows
        - 74 columns
-	**Austin:**
    - Calendar Data:
        - 4,369,416 rows
        - 7 columns
    - Listings Data:
        - 11,971 rows
        - 74 columns

## Exploratory Data Analysis (EDA)
**Calendar Data:**
The first data set utilized, the 'calendar data' has seven columns, outlined below. The data set contains 365 rows for each AirBnB. The earliest date is 12/15/2021 and the latest is 12/17/2022. The main steps of EDA performed on the calendar data set were:

 - 1) Converting the date column to date-time
 - 2) Converting the columns with 'object' types of 't' and 'f' to binary (0, 1) values
 - 3) Removing NaN values
 - 4) Removing 'available', 'minimum nights', and 'maximum nights' columns as these will be irrelevant from the stakeholder's perspective
 
After performing these steps the data set was left with three columns (listing_id, daily_price, date) and 958,489 rows.

**Listings Data:** The second data set utilized, the 'listings data' has 74 columns, outlined below. The data set contains one row for each AirBnB totaling 2,626 rows. The earliest date is 12/15/2022 and the latest is 12/17/2023. The main steps of EDA performed on the calendar data set were:

 - 1) Removing columns which will be irrelevant to the stakeholder, including:
    - a) All columns related to rating
    - b) Most columns related to the host including name, picture, response time, etc. with the exception of whether or not the host is a superhost
    - c) Columns related to scraping, the minimum/maximum nights, availability and price as the daily price from the calendar data set will be utilized
 - 2) Filling in any missing neighborhood data with the most common neighborhood which was typically just the city name (Asheville, Nashville, and Austin)
 - 3) Dropping neighborhoods which do not fall into the top 5 by count
 - 4) Removing any NaNs values from columns such as bedrooms, beds, and bathrooms

After performing these steps the data set was left with 28 columns and 2,269 rows.

**Combined Data:**
After performing the above EDA steps on each of the data sets, the data sets were combined based on the AirBnB listing id. Once the data was combined, outliers were dropped with the following criteria:
 - 1) Only including bathrooms which are between 1 and 5
 - 2) Only including bedrooms which are less than or equal to 6 and beds which are less than or equal to 11
 - 3) Excluding hotel rooms and shared rooms therefor only including entire home/apt and private room AirBnBs
 - 4) Limiting daily prices which are less than or equal to $1,000
 
 **Example AirBnB:**
The below figure is a plot of a single Asheville AirBnB and 365 point of daily price data. As shown below, it is clear that there is a seasonal pattern related to price in Asheville with January through April trending between 90-110 dollars per night and between May and December trending between 115 - 120 dollars per night.

![Sample AirBnB](./Images/Asheville_Sample_BnB.jpeg)

## Modeling
The data set utilized for the below models is the combined, cleaned Calendar and Listings data sets. To efficiently predict the AirBnB’s price, we will utilize Neural Network models, Random Forest Regressor modes, and an XGBoost model.

### Final Results
The scores achieved with this model were a train accuracy and MAE of ~99% and $1.80, respectively, and a test accuracy and loss of ~98% and $4.80, respectively. Although very simple, these scores with minimal MAE will best serve the stakeholder when they are seeking to predict the nightly price of their AirBnB.

## Recomendation
With the above analysis, it is recommended that the stakeholder, utilizes the final, third model, which is a simple Random Forest Regression model. As previously mentioned, AirBnB owners are not required any prior experience within real estate or renting out property. As such a number of AirBnB owners rely on best-guesses and intuition when it comes to what they should charge for a night’s stay at their property. As a result, if the owner is listing their property for too high of a price they could lose out on business, too low and there is a loss for potential income. Based the model, it appears as though if the stakeholder utilizes this model, it will correctly predict AirBnB prices with ~99% accuracy based on the home characteristics the stakeholder will input. For more detail on these inputs, please refer to the Streamlit link here () which is a user-friendly app AirBnB owners can utilize to run these predictions.

## Next Steps
Further criteria and analyses could yield additional insights to further inform the stakeholder by:

- **Consider real-world price impacts such as inflation.** The stakeholder should consider factoring in real-word price impacts. Recent changes in the world, namely rapidly increasing inflation could heavily impact the predictive model. Considering the model is running on 2022 historical data, it does not consider future impacts to price. As such, the stakeholder should consider adding an inflation factor or multiplier when predicting 2023 data.
- **Factor in other factors which impact price such as ‘experiences’.** Another factor the stakeholder should consider is including data of AirBnB ‘experiences’. AirBnB offers a service known as ‘experiences’. With these experiences, guests are able to book through their AirBnb local activities in the area such as tours, dance, classes and more. Given these experiences are booked in coordination with an AirBnB, it would be interesting for the stakeholder to consider adding data on this to evaluate if there is in fact a relationship between an AirBnB’s price and its proximity to experiences.
- **Consider additional data (older data, other cities).** Lastly, the stakeholder should consider additional data to factor into the model. Given this data set relies on just 2022 price data it would be helpful to consider adding even older data such as 2020 and 2021. Additionally, the models are only being utilized for three cities, Asheville, Nashville, and Austin. By factoring in these other attributes the model would only further train and become more accurate when reviewing unseen data.

### Repository Navigation
- Data
    - Asheville
        - [Calendar Data](./Data/AirBnB/Asheville/2021/asheville_calendar_2021.csv)
        - [Listings Data](./Data/AirBnB/Asheville/2021/asheville_listings_2021.csv)
    - Austin
        - [Calendar Data](./Data/AirBnB/Austin/2021/austin_calendar_2021.csv)
        - [Listings Data](./Data/AirBnB/Austin/2021/austin_listings_2021.csv)
    - Nashville
        - [Calendar Data](./Data/AirBnB/Nashville/2021/nashville_calendar_2021.csv)
        - [Listings Data](./Data/AirBnB/Nashville/2021/nashville_listings_2021.csv)
- Images
    - 
- Notebooks
    - [AirBnB Price Prediction Final Notebook (PDF)](./Notebooks/AirBnB Price Prediction Final Notebook - Jupyter Notebook.pdf)
    - [AirBnB Price Prediction Final Notebook](./Notebooks/AirBnB Price Prediction Final Notebook.ipynb)
    - [Asheville Price Prediction Notebook](./Notebooks/Asheville AirBnB Price Prediction Notebook.ipynb)
    - [Austin Price Prediction Notebook](./Notebooks/Austin AirBnB Price Prediction Notebook.ipynb)
    - [Nashville Price Prediction Notebook](./Notebooks/Nashville AirBnB Price Prediction Notebook.ipynb)
- Supplements