import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta
from prophet import Prophet

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Tourist dashboard',
    page_icon=':flag-lk:', # This is an emoji shortcode. Could be a URL too.
)

# Reading file
df = pd.read_csv("tourist_data_weekly.csv")

# Normalize the date column to ensure consistent formatting
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
# Sort the DataFrame by the date column
df = df.sort_values(by='date')


'''
# :flag-lk: Sri Lankan Tourist Dashboard

This Dashboard compares the prediction of tourists arriving to Sri Lanka for the next 3 months using 2 machine learning models.

**Project flow:** 
1. Get data as a csv from [Sri Lankan Tourist Development Authority](https://www.sltda.gov.lk/en/statistics) website for tourist arrivals 
2. Read the data using python
3. Clean the data 
4. Train and use Linear Regression model to predict tourists for the next 3 months
5. Train and use FBProphet Timeseries model to forecast tourists for the next 3 months 
6. Plot the results in charts using streamlit to visualize it
7. Deploy project using streamlit 
'''

# Add some spacing
''
''

''
# st.dataframe(df)

# Linear Regression to predict future tourists
# Prepare the data
df['days'] = (df['date'] - df['date'].min()).dt.days
X = df[['days']]
y = df['Tourists Total']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict future values for the next 3 months
future_dates = [df['date'].max() + timedelta(days=i*30) for i in range(1, 4)]
future_days = np.array([(date - df['date'].min()).days for date in future_dates]).reshape(-1, 1)
future_predictions = model.predict(future_days)

# Create a DataFrame for future predictions
future_df = pd.DataFrame({'date': future_dates, 'Tourists': future_predictions})

# Combine historical and future data
combined_df = pd.concat([df[['date', 'Tourists Total']], future_df])


fig = px.line(df, x='date', y=list(df.columns), labels={'date': 'Date', 'Tourists Total': 'Number of Tourists'})
st.plotly_chart(fig)

# Plot total tourists over time with future predictions (Linear Regression)
st.subheader('Prediction for Total Tourists')

# st.line_chart(
#     df,
#     x='date',
#     y='Tourists Total'
# )

fig = px.line(df, x='date', y='Tourists Total', title='Using Linear Regression Model', labels={'date': 'Date', 'Tourists Total': 'Number of Tourists'})
fig.add_scatter(x=future_df['date'], y=future_df['Tourists'], mode='lines', name='Predicted Tourists', line=dict(dash='dot'))
st.plotly_chart(fig)

# Facebook Prophet to predict future tourists
# Prepare data for Prophet
prophet_df = df[['date', 'Tourists Total']].rename(columns={'date': 'ds', 'Tourists Total': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_df)

# Create future dataframe for 3 months
future_prophet = prophet_model.make_future_dataframe(periods=90)
forecast = prophet_model.predict(future_prophet)

# Plot total tourists over time with future predictions (Facebook Prophet)
# st.subheader('Facebook Prophet Predictions')
fig = px.line(prophet_df, x='ds', y='y', title='Using Facebook Prophet Predictions', labels={'ds': 'Date', 'y': 'Number of Tourists'})
fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Tourists', line=dict(dash='dot'))
st.plotly_chart(fig)

# Select a country to visualize
st.subheader('Select a Country to Visualize with Linear Regression Predictions')
country = st.selectbox('Select a country for Linear Regression', df.columns[2:])

# Linear Regression for selected country
X_country = df[['days']]
y_country = df[country]

country_model = LinearRegression()
country_model.fit(X_country, y_country)

# Predict future values for the next 3 months
future_country_predictions = country_model.predict(future_days)

# Create a DataFrame for future predictions
future_country_df = pd.DataFrame({'date': future_dates, country: future_country_predictions})

# Combine historical and future data
combined_country_df = pd.concat([df[['date', country]], future_country_df])

# Plot selected country with future predictions (Linear Regression)
fig = px.line(df, x='date', y=country, title=f'Tourists from {country} Over Time with Predictions', labels={'date': 'Date', country: 'Number of Tourists'})
fig.add_scatter(x=future_country_df['date'], y=future_country_df[country], mode='lines', name='Predicted Tourists', line=dict(dash='dot'))
st.plotly_chart(fig)

# Select a country to visualize with Facebook Prophet
st.subheader('Select a Country to Visualize with Facebook Prophet Predictions')
country_fb = st.selectbox('Select a country for Facebook Prophet', df.columns[2:], key='fb')

# Prepare data for Prophet for the selected country
prophet_country_df = df[['date', country_fb]].rename(columns={'date': 'ds', country_fb: 'y'})
prophet_country_model = Prophet()
prophet_country_model.fit(prophet_country_df)

# Create future dataframe for 3 months
future_prophet_country = prophet_country_model.make_future_dataframe(periods=90)
forecast_country = prophet_country_model.predict(future_prophet_country)

# Plot selected country with future predictions (Facebook Prophet)
fig = px.line(prophet_country_df, x='ds', y='y', title=f'Tourists from {country_fb} Over Time with Predictions', labels={'ds': 'Date', 'y': 'Number of Tourists'})
fig.add_scatter(x=forecast_country['ds'], y=forecast_country['yhat'], mode='lines', name='Predicted Tourists', line=dict(dash='dot'))
st.plotly_chart(fig)
