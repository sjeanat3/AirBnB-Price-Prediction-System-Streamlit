import pandas as pd
import numpy as np
import sklearn
import pickle
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from geopy.distance import geodesic

asheville_modeling_data = pd.read_pickle('../Pickles/asheville_modeling_data.pickle')
streamlit_model3_results = joblib.load('../Joblib/streamlit_model3_results.joblib')
# streamlit_model3_results = pd.read_pickle('../Supplements/Pickles/streamlit_model3_results.pickle')

#Create a model 3 X and y variable 
model3_X = asheville_modeling_data.drop(['daily_price', 'listing_id'], axis = 1)
model3_y = asheville_modeling_data['daily_price']

#Train, test, split the X and y variables
model3_X_train, model3_X_test, model3_y_train, model3_y_test = train_test_split(model3_X, 
                                                                                model3_y, 
                                                                                test_size = 0.2)

#Create a list of numeric columns
model3_numeric_cols = ['host_is_superhost', 'accommodates', 'bedrooms', 'beds', 'bathrooms', 
                       'Air conditioning', 'Wifi', 'TV', 'Kitchen', 'Washer', 'Dryer', 'Heating',
                       'distance_to_biltmore', 'distance_to_downtown']

#Create a list of nominal columns
model3_nominal_cols = ['neighborhood', 'room_type', 'day_of_week', 'month', 'week']

#Scale the numeric columns
model3_numeric_pipeline = Pipeline([('scaler', StandardScaler())])

#One hot encode the nominal columns
model3_nominal_pipeline = Pipeline([('ohe', OneHotEncoder(sparse = False))])

#Column tranform the two pipelines
ct = ColumnTransformer([('nominalpipe', model3_nominal_pipeline, model3_nominal_cols ),
                        ('numpipe', model3_numeric_pipeline, model3_numeric_cols)])

#Create a final pipeline with the column transformer and random forest regressor model
model3_final_pipe = Pipeline([('preprocess', ct),
                              ('model', RandomForestRegressor())])

st.sidebar.header('Specify AirBnB Characteristics')

yes_no_options = ['Yes', 'No']
options = list(range(len(yes_no_options)))

room_options = ['Entire home/apt', 'Private room']
room_options_len = list(range(len(room_options)))

neighborhood_options = ['Asheville', 'Candler', 'Fletcher', 'Woodfin']
neighborhood_options_len = list(range(len(neighborhood_options)))

superhost = st.sidebar.selectbox('Superhost?', options, format_func = lambda x: yes_no_options[x])
room_type = st.sidebar.selectbox('Room Type', room_options_len, format_func = lambda x: room_options[x])
accommodates = st.sidebar.slider('Accommodate', step = 1, min_value = 1, max_value = 10)
bedrooms = st.sidebar.slider('Bedrooms', step = 1, min_value = 1, max_value = 10)
beds = st.sidebar.slider('Beds', step = 1, min_value = 1, max_value = 10)
neighborhood = st.sidebar.selectbox('Neighborhood', neighborhood_options_len, format_func = lambda x: neighborhood_options[x])
bathrooms = st.sidebar.slider('Bathrooms', min_value = 1, max_value = 10)
air_conditioning = st.sidebar.selectbox('AC?', options, format_func = lambda x: yes_no_options[x])
wifi = st.sidebar.selectbox('Wifi?', options, format_func = lambda x: yes_no_options[x])
tv = st.sidebar.selectbox('TV?', options, format_func = lambda x: yes_no_options[x])
kitchen = st.sidebar.selectbox('Kitchen?', options, format_func = lambda x: yes_no_options[x])
washer = st.sidebar.selectbox('Washer?', options, format_func = lambda x: yes_no_options[x])
dryer = st.sidebar.selectbox('Dryer?', options, format_func = lambda x: yes_no_options[x])
heating = st.sidebar.selectbox('Heating?', options, format_func = lambda x: yes_no_options[x])
latitude = st.sidebar.number_input("Latitude")
longitude = st.sidebar.number_input("Longitude")
checkin_date = st.sidebar.date_input('Check-In Date')
checkout_date = st.sidebar.date_input('Check-Out Date')

def user_input_checkin_features():
    def yes_no_conversion(input):
        if input == 'Yes':
            return 1
        else:
            return 0
    
    yes_no_conversion(superhost)
    yes_no_conversion(air_conditioning)
    yes_no_conversion(wifi)
    yes_no_conversion(tv)
    yes_no_conversion(kitchen)
    yes_no_conversion(washer)
    yes_no_conversion(dryer)
    yes_no_conversion(heating)
    
    conv_long = float(longitude)
    conv_lat = float(latitude)
    test_coords = list((conv_lat, conv_long))
    
    biltmore = (35.54108101423884, -82.55210010496437) 
    downtown = (35.60405939066325, -82.54533225431588)
    
    checkin_data = {'host_is_superhost': superhost,
            'room_type': room_type,
            'accommodates': accommodates,
            'bedrooms': bedrooms,
            'beds': beds,
            'neighborhood': neighborhood,
            'bathrooms': bathrooms,
            'Air conditioning': air_conditioning,
            'Wifi': wifi,
            'TV': tv,
            'Kitchen': kitchen,
            'Washer': washer,
            'Dryer': dryer,
            'Heating': heating,
            'distance_to_biltmore': geodesic(test_coords, biltmore).miles,
            'distance_to_downtown': geodesic(test_coords, downtown).miles,
            'day_of_week': pd.to_datetime(checkin_date).dayofweek,
            'month': pd.to_datetime(checkin_date).month,
            'week': pd.to_datetime(checkin_date).week}
    
    checkin_features = pd.DataFrame(checkin_data, index=[0])
    
    return checkin_features

def user_input_checkout_features():
    def yes_no_conversion(input):
        if input == 'Yes':
            return 1
        else:
            return 0
    
    yes_no_conversion(superhost)
    yes_no_conversion(air_conditioning)
    yes_no_conversion(wifi)
    yes_no_conversion(tv)
    yes_no_conversion(kitchen)
    yes_no_conversion(washer)
    yes_no_conversion(dryer)
    yes_no_conversion(heating)
    
    conv_long = float(longitude)
    conv_lat = float(latitude)
    test_coords = list((conv_lat, conv_long))
    
    biltmore = (35.54108101423884, -82.55210010496437) 
    downtown = (35.60405939066325, -82.54533225431588)
    
    checkout_data = {'host_is_superhost': superhost,
        'room_type': room_type,
        'accommodates': accommodates,
        'bedrooms': bedrooms,
        'beds': beds,
        'neighborhood': neighborhood,
        'bathrooms': bathrooms,
        'Air conditioning': air_conditioning,
        'Wifi': wifi,
        'TV': tv,
        'Kitchen': kitchen,
        'Washer': washer,
        'Dryer': dryer,
        'Heating': heating,
        'distance_to_biltmore': geodesic(test_coords, biltmore).miles,
        'distance_to_downtown': geodesic(test_coords, downtown).miles,
        'day_of_week': pd.to_datetime(checkout_date).dayofweek,
        'month': pd.to_datetime(checkout_date).month,
        'week': pd.to_datetime(checkout_date).week}
    
    checkout_features = pd.DataFrame(checkout_data, index=[0])
    
    return checkout_features

if st.button('Submit'):
    checkin_features = user_input_checkin_features()
    checkout_features = user_input_checkout_features()
    checkin_price = float(streamlit_model3_results.predict(checkin_features))
#     checkout_price = float(streamlit_model3_results.predict(checkout_features))
#     average_price = (checkin_price + checkout_price) / 2
#     total_stay_days = (pd.to_datetime(checkout_date) - pd.to_datetime(checkin_date)).days
    st.header('Prediction of AirBnB Nightly Price')
    st.write(checkin_price)
    st.write('---')

# print(f'Checkin price ${checkin_price} per night')
# print(f'Checkout price ${checkout_price} per night')
# print(f'Total price of stay ${int((average_price * total_stay_days))}')
# print(f'Average daily price ${average_price}')
