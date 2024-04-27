import streamlit as st
from PIL import Image

image = Image.open('data/logo.png')
st.image(image)

st.header("Greetings! With Predict-IV you can input your desired features and one of our models (of your choosing) will predict the monthly rent for your desired place of residence")
streets = ['Abrego', 'Camino Corto', 'Cordoba', 'Del Playa', 'El Embarcadero', 'El Nido', 'Embarcadero del Norte', 'Pasado', 'Picasso', 'Sabado Tarde', 'Seville', 'Sueno', 'Trigo', 'Cervantes', 'El Greco', 'Embarcadero del Mar', 'Madrid', 'Pardall', 'Segovia', 'Camino del Sur', 'El Colegio']

leasors = ['KAMAP', 'Bartlein & Company, Inc.', 'Ben Roberts Properties', 'Harwin Management', 'Embarcadero Company', 'Wolfe and Associates', 'SFMVDM', 'Playa Life', 'Meridian', 'Ventura Investment Co', 'ICON', 'CBC Sweeps Essex', 'Gallagher Property', 'Micahel Gilson']

model = st.radio("ML Model", ["Linear Regression", "Random Forest (Recommended)", "Decision Tree"])
year = st.slider("Year", 1970,2070)
bedroom = st.number_input("Bedrooms", min_value = 0, step = 1)
bathroom = st.number_input("Bathrooms", min_value = 0, step = 1)
street = st.selectbox("Street Name", street)
leasor = st.selectbox("Leasor Name", leasors)
pred = st.button("Predict Price")

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

model_df = pd.read_csv("FUPD.csv")
X = model_df[['YEAR', 'STREET NAME', 'BEDROOM', 'BATHROOM', 'LEASOR']]
y = model_df[['MONTHLY RENT ($)']]
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 427)

if (model == 'Linear Regression'):
  lr_model = LinearRegression()
  lr_model.fit(X_train, y_train)

elif (model == 'Random Forest (Recommended)'):
  rf_model = RandomForestRegressor(max_depth = 30, min_samples_split = 2, n_estimators = 10, random_state = 427)
  rf_model.fit(X_train, y_train)

elif (model == 'Decision Tree'):
  dt_model = DecisionTreeRegressor(max_depth = 20, max_leaf_nodes = 80, min_samples_split = 2, random_state = 427)
  dt_model.fit(X_train, y_train)

predictx = pd.DataFrame({
  'YEAR': [year],
  'STREET NAME': [streets.index(street)],
  'BEDROOM': [bedroom],
  'BATHROOM': [bathroom],
  'LEASOR': [leasors.index(leasor)]})

if (pred):
  if (model == 'Linear Regression'):
    price = lr_model.predict(predictx)
    if (price >= 0):
      st.subheader("Your predicted monthly rent is: $" + str('{0:.2f}'.format(price[0][0])))
    else:
      st.subheader("Your predicted monthly rent is: -$" + str('{0:.2f}'.format(abs(price[0][0]))))
  elif (model == 'Random Forest (Recommended'):
    if (price >= 0):
      st.subheader("Your predicted monthly rent is: $" + str('{0:.2f}'.format(price[0])))
    else:
      st.subheader("Your predicted monthly rent is: -$" + str('{0:.2f}'.format(abs(price[0]))))
  elif (model == "Decision Tree"):
    if (price >= 0):
      st.subheader("Your predicted monthly rent is: $" + str('{0:.2f}'.format(price[0])))
    else:
      st.subheader("Your predicted monthly rent is: -$" + str('{0:.2f}'.format(abs(price[0]))))