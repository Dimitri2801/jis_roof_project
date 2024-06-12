import streamlit as st
import numpy as np
from catboost import CatBoostClassifier

# Load model
best_catboost_model = CatBoostClassifier()
best_catboost_model.load_model('best_catboost_model.bin')

# Mapping for wind direction
ddd_car_mapping = {'W': 1, 'SW': 2, 'E': 3, 'S': 4, 'NE': 5, 'SE': 6, 'C': 7, 'N': 8, 'NW': 9}

# Function to predict rain
def predict_rain(Tn, Tx, ff_avg, ff_x, Tavg, ddd_car, ss, ddd_x, RH):
    # Convert wind direction to numeric value
    ddd_car_numeric = ddd_car_mapping.get(ddd_car, 0)
    
    # Prepare input features
    input_features = np.array([[Tn, Tx, ff_avg, ff_x, Tavg, ss, ddd_x, ddd_car_numeric, RH]])
    
    # Predict using model
    prediction = best_catboost_model.predict(input_features)
    
    return prediction


# Streamlit app
# st.title("Jakarta International Stadium Rain Detector")
st.header("Jakarta International Stadium Rain Detector", anchor=None, help=None, divider=False)
# st.image('https://asset-2.tstatic.net/tribunnewswiki/foto/bank/images/Jakarta-International-Stadium-1122333.jpg', caption=None, width=680, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
# Input form
st.sidebar.title("Input Parameters")
Tn = st.sidebar.number_input("Masukkan nilai min temperature (°C) Tn: ", min_value=22.0, max_value=50.0, value=22.0)
Tx = st.sidebar.number_input("Masukkan nilai max temperature (°C) Tx: ", min_value=22.0, max_value=50.0, value=23.0)
ff_avg = st.sidebar.number_input("Masukkan nilai average wind speed (m/s) ff_avg: ", min_value=0.0, max_value=100.0, value=30.0)
ff_x = st.sidebar.number_input("Masukkan nilai max wind speed (m/s) ff_x: ", min_value=0.0, max_value=100.0, value=15.0)
Tavg = st.sidebar.number_input("Masukkan nilai avg temperature (°C) Tavg: ", min_value=22.0, max_value=50.0, value=22.0)
ddd_car = st.sidebar.selectbox("Masukkan wind direction (°) ddd_car: ", list(ddd_car_mapping.keys()))
ss = st.sidebar.number_input("Masukkan nilai duration of sunshine (hour) ss: ", min_value=0.0, max_value=24.0, value=2.0)
ddd_x = st.sidebar.number_input("Masukkan nilai wind direction at maximum speed (°) ddd_x: ", min_value=0, max_value=360, value=180)
RH = st.sidebar.number_input("Masukkan nilai avg humidity(%) RH: ", min_value=0, max_value=100, value=55)

# Predict
if st.sidebar.button("Predict"):
    prediction = predict_rain(Tn, Tx, ff_avg, ff_x, Tavg, ddd_car, ss, ddd_x, RH)
    if prediction[0] == 1:
        st.video("close_jis.mp4", format="video/mp4", start_time=0, autoplay=True)
        st.write("Berdasarkan input yang diberikan, model memprediksi bahwa kemungkinan akan ada hujan.\n Atap ditutup.")
    else:
        st.video("open_jis.mp4", format="video/mp4", start_time=0, autoplay=True)
        st.write("Berdasarkan input yang diberikan, model memprediksi bahwa kemungkinan tidak akan ada hujan.\n Atap dibuka.")
        

