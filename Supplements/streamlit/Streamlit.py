import pandas as pd
import numpy as np
import sklearn
import streamlit as st
# import joblib
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from geopy.distance import geodesic
from datetime import timedelta, date


# with open('asheville_modeling_data.pkl', 'rb') as f:
#     asheville_modeling_data = pickle.load(f)

# streamlit_model3_results = joblib.load('streamlit_model3_results.joblib')

st.set_page_config(layout="wide")
st.sidebar.header('Please Specify Your AirBnB Characteristics:')
st.image('airbnb5.webp', width = 600)

yes_no_options = ['Yes', 'No']
options = list(range(len(yes_no_options)))

room_options = ['Entire home/apt', 'Private room ']
room_options_len = list(range(len(room_options)))

neighborhood_options = ['Asheville', 'Candler', 'Fletcher', 'Woodfin']
neighborhood_options_len = list(range(len(neighborhood_options)))

superhost = st.sidebar.selectbox('Superhost?:', options, format_func = lambda x: yes_no_options[x])
room_type = st.sidebar.selectbox('Room Type:', room_options_len, format_func = lambda x: room_options[x])
accommodates = st.sidebar.slider('Accommodate (Number of People):', step = 1, min_value = 1, max_value = 16)
bedrooms = st.sidebar.slider('Number of Bedrooms:', step = 1, min_value = 1, max_value = 6)
beds = st.sidebar.slider('Number of Beds:', step = 1, min_value = 1, max_value = 11)
neighborhood = st.sidebar.selectbox('Neighborhood:', neighborhood_options_len, format_func = lambda x: neighborhood_options[x])
bathrooms = st.sidebar.slider('Number of Bathrooms:', step = 0.5, min_value = 1.0, max_value = 5.0)
air_conditioning = st.sidebar.selectbox('AC?:', options, format_func = lambda x: yes_no_options[x])
wifi = st.sidebar.selectbox('Wifi?:', options, format_func = lambda x: yes_no_options[x])
tv = st.sidebar.selectbox('TV?:', options, format_func = lambda x: yes_no_options[x])
kitchen = st.sidebar.selectbox('Kitchen?:', options, format_func = lambda x: yes_no_options[x])
washer = st.sidebar.selectbox('Washer?:', options, format_func = lambda x: yes_no_options[x])
dryer = st.sidebar.selectbox('Dryer?:', options, format_func = lambda x: yes_no_options[x])
heating = st.sidebar.selectbox('Heating?', options, format_func = lambda x: yes_no_options[x])
latitude = st.sidebar.number_input("Latitude:")
longitude = st.sidebar.number_input("Longitude:")
checkin_date = st.sidebar.date_input('Check-In Date:', min_value = date.today(), max_value = pd.to_datetime('12/30/2023'))
checkout_date = st.sidebar.date_input('Check-Out Date:', min_value=(date.today()), 
                                      max_value = pd.to_datetime('12/31/2023'))
st.title('Prediction of AirBnB Nightly Price')
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
    
    conv_long = longitude
    conv_lat = latitude
    test_coords = (conv_lat, conv_long)
    
    biltmore = (35.54108101423884, -82.55210010496437) 
    downtown = (35.60405939066325, -82.54533225431588)
    
    checkin_data = {'host_is_superhost': superhost,
                    'room_type': str(room_type),
                    'accommodates': int(accommodates),
                    'bedrooms': float(bedrooms),
                    'beds': float(beds),
                    'neighborhood': str(neighborhood),
                    'bathrooms': float(bathrooms),
                    'Air conditioning': air_conditioning,
                    'Wifi': wifi,
                    'TV': tv,
                    'Kitchen': kitchen,
                    'Washer': washer,
                    'Dryer': dryer,
                    'Heating': heating,
                    'distance_to_biltmore': float(geodesic(test_coords, biltmore).miles),
                    'distance_to_downtown': float(geodesic(test_coords, downtown).miles),
                    'day_of_week': pd.to_datetime(checkin_date).dayofweek,
                    'month': pd.to_datetime(checkin_date).month,
                    'week': pd.to_datetime(checkin_date).week}
    
    checkin_features = pd.DataFrame(checkin_data, index = [0])
    
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
    
    conv_long = longitude
    conv_lat = latitude
    test_coords = (conv_lat, conv_long)
    
    biltmore = (35.54108101423884, -82.55210010496437) 
    downtown = (35.60405939066325, -82.54533225431588)
    
    checkout_data = {'host_is_superhost': superhost,
                    'room_type': str(room_type),
                    'accommodates': int(accommodates),
                    'bedrooms': float(bedrooms),
                    'beds': float(beds),
                    'neighborhood': str(neighborhood),
                    'bathrooms': float(bathrooms),
                    'Air conditioning': air_conditioning,
                    'Wifi': wifi,
                    'TV': tv,
                    'Kitchen': kitchen,
                    'Washer': washer,
                    'Dryer': dryer,
                    'Heating': heating,
                    'distance_to_biltmore': float(geodesic(test_coords, biltmore).miles),
                    'distance_to_downtown': float(geodesic(test_coords, downtown).miles),
                     'day_of_week': pd.to_datetime(checkout_date).dayofweek,
                     'month': pd.to_datetime(checkout_date).month,
                     'week': pd.to_datetime(checkout_date).week}
    
    checkout_features = pd.DataFrame(checkout_data, index=[0])
    
    return checkout_features

total_stay_days = (pd.to_datetime(checkout_date) - pd.to_datetime(checkin_date)).days

if st.button('Submit'):
    
    with open('streamlit_model3_results.pkl', 'rb') as f:
        streamlit_model3_results = pickle.load(f)
    
    checkin_features = user_input_checkin_features()
    checkout_features = user_input_checkout_features()

    checkin_price = streamlit_model3_results.predict(checkin_features)
    checkin_price = streamlit_model3_results.predict(checkin_features) 

    checkin_price = float(checkin_price)
    checkout_price = float(streamlit_model3_results.predict(checkout_features))
    average_price = (checkin_price + checkout_price) / 2
    st.header('Estimated AirBnB Check-In Price')
    st.subheader(f'${checkin_price}')
    st.write('---')
    st.header('Estimated AirBnB Check-Out Price')
    st.subheader(f'${checkout_price}')
    st.write('---')
    st.header('Estimated AirBnB Average Daily Price')
    st.subheader(f'${(average_price)}')
    st.write('---')
    st.header('Estimated Total Price of Stay')
    st.write('Note: If Check-In Date is the same as Check-Out Date, this will return $0')
    st.subheader(f'${(average_price * total_stay_days)}')
