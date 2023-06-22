
import streamlit as st
import pandas as pd
import joblib
import sklearn
import xgboost
import category_encoders

st.header("AirLine Flights Web Site")


Inputs = joblib.load("Inputs.pkl")
Model = joblib.load("Model.pkl")

def prediction (Airline, Source, Destination, StopCount, Journey_Year, Journey_Month, Journey_day, Season):
    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0,'Airline'] = Airline
    test_df.at[0, 'Source'] = Source
    test_df.at[0, 'Destination'] = Destination
    test_df.at[0, 'StopCount'] = StopCount
    test_df.at[0, 'Journey_Year'] = Journey_Year
    test_df.at[0, 'Journey_Month'] = Journey_Month
    test_df.at[0, 'Journey_day'] = Journey_day
    test_df.at[0, 'Season'] = Season
    result = Model.predict(test_df)
    return result[0]

def main():
    st.text('Airline Flights Price Prediction')
    st.image("Air_Line.jpg")
    Airline = st.selectbox('Airline_Company', ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
       'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',
       'Vistara Premium economy', 'Jet Airways Business',
       'Multiple carriers Premium economy', 'Trujet'])
    Source = st.selectbox('Source', ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
    Destination = st.selectbox('Destination', ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])
    StopCount = st.selectbox('Transit',[0,1,2,3,4])
    Journey_Year = st.selectbox('Journey_Year', [2019])
    Journey_Month = st.slider('Month',min_value=1 , max_value= 12, value=1 , step= 1)
    Journey_day = st.slider('Day',min_value=3 , max_value=27 , value=1 , step=1 )
    Season = st.selectbox('Season', ['spring', 'winter', 'fall', 'summer'])
    
    if st.button("predict"):
        result = prediction (Airline, Source, Destination, StopCount, Journey_Year, Journey_Month, Journey_day, Season)
        st.text(f'This Flight Trip Will Cost ${result} USD')

if __name__ == '__main__':
    main()
