# streamlit app for air passengers dataset

import base64

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.signal import periodogram

#Page configuration
st.set_page_config(page_title="Air Passenger Forecast",
                   page_icon="logo3.png",
                   layout="wide")
st.title("✈️ Air Passenger Data Analysis & Forecasting")

#Background styling
def set_background(image):
    with open(image, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        /*background */
        .stApp {{
            background: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Content inside */
        .st-emotion-cache-1avcm0n, .st-emotion-cache-1y4p8pa, 
        .stMarkdown, .stDataFrame, .stPlotlyChart {{
            background-color: rgba(255, 255, 255, 0.88) !important;
            border-radius: 8px !important;
            padding: 12px !important;
            margin: 5px 0px !important;
            color: black !important;
        }}

        /* NAVIGATION BAR FEATURES */
        .stSelectbox > div > div {{
            background-color: black !important;   
            color: white !important;             
            border: 2px solid white !important;  
            border-radius: 6px !important;
            font-weight: bold !important;
        }}
        .stSelectbox [role="listbox"] {{
            background-color: black !important;
            color: white !important;
        }}
        .stSelectbox [role="option"]:hover {{
            background-color: #222222 !important; 
            color: #00ccff !important;             
        }}
        .stSelectbox label {{
            color: white !important;
            font-weight: bold !important;
            font-size:16px !important;
        }}

        /* SLIDER BAR */
        .stSlider > div {{
            background-color: #000000 !important;
            color: white !important;
            border: 2px solid black !important;
            border-radius: 6px !important;
            padding: 8px !important;
        }}
        .stSlider .st-emotion-cache-1e0x3sz {{
            background-color: #333333 !important; 
        }}
        .stSlider .st-emotion-cache-1vzeuhh {{
            background-color: #ff5500 !important; 
            border: 2px solid white !important;
            height: 20px !important;
            width: 20px !important;
        }}
        .stSlider label {{
            color: white !important;
            font-weight: bold !important;
            font-size:16px !important;
        }}

        /* Model Accuracy */
        .stMetric > div {{
            background-color: black !important;
            color: white !important;
            border: 2px solid white !important;
            border-radius: 6px !important;
            padding: 10px !important;
        }}
        .stMetric label, .stMetric div {{
            color: white !important;
            font-weight: bold !important;
        }}

        /* DOWNLOAD BUTTON */
        .stDownloadButton > button {{
            background-color: black !important;
            color: white !important;
            border: 2px solid white !important;
            border-radius: 6px !important;
            font-weight: bold !important;
            padding: 6px 12px !important;
        }}
        .stDownloadButton > button:hover {{
            background-color: #222222 !important;
            color: #00ccff !important;
            border-color: #00ccff !important;
        }}

        /* SIDEBAR */
        .stSidebar {{
            background-color: rgba(0, 0, 0, 0.85) !important;
        }}
        .stSidebar .stMarkdown, .stSidebar p {{
            color: white !important;
            font-weight: 500 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background('bg.jpeg')

#Loading data with caching to improve performance
@st.cache_data
def load_data():
    df = pd.read_csv('airline-passengers.csv', parse_dates=['Month'], index_col='Month')
    df = df.rename(columns={'#Passengers':'Passengers'})
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    return df

data = load_data()

# Sidebar navigation
menu = st.sidebar.selectbox(
    "📋 Navigation",
    ["Data Overview", "Pattern Analysis", "Seasons Changes", "Forecasting"]
)

# 1- Data Overview
if menu == "Data Overview":
    st.header("📊 Raw Data & Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(data)
    
    with col2:
        st.subheader("Data Description")
        st.dataframe(data.describe())

    # Line chart
    fig = px.line(data, y='Passengers', title='Total Passenger Trend Over Time', markers=True)
    st.plotly_chart(fig, use_container_width=True)

# 2- Pattern Analysis
elif menu == "Pattern Analysis":
    st.header("🔍 Discovering Patterns")
    
    # Seasonal Plot
    st.subheader("Seasonal Plot")
    fig_season = px.line(data, x='Month', y='Passengers', color='Year',
                         title='Monthly Pattern Across All Years', markers=True)
    st.plotly_chart(fig_season, use_container_width=True)

    # Periodogram
    st.subheader("Periodogram: Detecting Cycles")
    fs = 12
    frequencies, spectrum = periodogram(data['Passengers'], fs=fs, detrend='linear')
    
    fig_period = go.Figure()
    fig_period.add_trace(go.Scatter(x=frequencies, y=spectrum, mode='lines', fill='tozeroy'))
    fig_period.update_layout(
        title='Periodogram',
        xaxis=dict(
            title='Cycle Length',
            type='log',
            tickvals=[1,2,4,6,12],
            ticktext=['1 Year','6 Months','3 Months','2 Months','1 Month']
        ),
        yaxis_title="Strength of Pattern"
    )
    st.plotly_chart(fig_period, use_container_width=True)
    st.info("✅ We see strongest pattern at 1 Year, followed by 6 Months")

# 3-Seasonal Animations
elif menu == "Seasons Changes":
    st.header("🎬 Animated Charts")
    
    # Animated Seasonal Plot
    st.subheader("Seasonal Pattern Evolution")
    data['Year_str'] = data['Year'].astype(str)
    fig_anim = px.line(data, 
                       x='Month', 
                       y='Passengers', 
                       color='Year_str',
                       animation_frame='Year_str',
                       title='How Passenger Patterns Changed Year by Year',
                       markers=True, height=500)
    min_val = data['Passengers'].min() * 0.9
    max_val = data['Passengers'].max() * 1.1
    fig_anim.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ))
    fig_anim.update_layout(yaxis=dict(range=[min_val, max_val]))
    st.plotly_chart(fig_anim, use_container_width=False)

# 4- Forecasting
elif menu == "Forecasting":
    st.header("🔮 Future Passenger Forecast")
    if isinstance(data.index, pd.PeriodIndex):
        data.index = data.index.to_timestamp()

    data.index = pd.DatetimeIndex(data.index)
        # st.success("Data loaded successfully! Building forecasting model...")
    
    # Create features
    fourier = CalendarFourier(freq='A', order=2)
    dp = DeterministicProcess(
        index=data.index,
        constant=True,
        order=1,
        seasonal=False,
        additional_terms=[fourier],
        drop=True
    )
    X = dp.in_sample()
    y = data['Passengers']

    # Train model
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index=X.index)

    # Create future dates
    forecast_months = st.slider("Select how many months to forecast:", 12, 60, 24)
    future_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), 
                                 periods=forecast_months, freq='M')
    X_future = dp.out_of_sample(steps=forecast_months,
                                forecast_index=future_index)
    y_forecast = pd.Series(model.predict(X_future), index=future_index)

    # Plot actual, fitted, and forecasted values
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=data.index, y=y, name='Actual', line=dict(color='blue')))
    fig_forecast.add_trace(go.Scatter(x=y_pred.index, y=y_pred, name='Fitted', line=dict(color='green', dash='dot')))
    fig_forecast.add_trace(go.Scatter(x=y_forecast.index, y=y_forecast, name='Forecast', line=dict(color='red', width=3)))
    
    fig_forecast.update_layout(
        title=f'Passenger Forecast: Next {forecast_months} Months',
        xaxis_title='Date',
        yaxis_title='Number of Passengers',
        hovermode='x unified'
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Show metrics
    mae = mean_absolute_error(y, y_pred)
    st.metric(label="Model Accuracy (MAE)", value=f"{mae:.2f} passengers")

    # Download forecast data
    forecast_df = pd.DataFrame({'Date':y_forecast.index, 'Forecasted_Passengers':y_forecast.values})
    st.download_button(
        label="📥 Download Forecast Data as CSV",
        data=forecast_df.to_csv(index=False),
        file_name='passenger_forecast.csv',
        mime='text/csv'
    )

st.sidebar.info("I hope you find this project insightful and enjoyable. Feel free to explore the data, discover patterns. Though the Data is a bit Old 🤣")
st.sidebar.info("Your feedback is always welcome!")
st.sidebar.info("Lets connect! 👇")
# st.sidebar.info("Contact: [📧email](mailto:robinsonraphael148@gmail.com)")
# st.sidebar.info("WhatsApp: [🔗WhatsApp](https://wa.link/g3kiwf)")
st.sidebar.info("GitHub: [💻GitHub](https://github.com/RB148-code)")