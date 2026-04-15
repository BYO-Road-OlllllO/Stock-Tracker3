import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import date

# Set the timeframe for historical data
START_DATE = "2026-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# App Layout & Title
st.set_page_config(page_title="Portfolio Tracker & Projector", layout="wide")
st.title('📈 Portfolio Tracker & Market Projector')

# Pre-populated portfolio 
portfolio = ['AI',	'BAC',	'BCTK',	'CSX',	'DAN',	'FSTA',	'FTEC',	'GOOG',	'GOOGL',	'JEPQ',	'LUV',	'MITT',	'PSKY',	'RIVN',	'Ronb',	'SCHD',	'SCHY',	'TSLA',	'VOO',	'WBD',	'XOVR',	'AAL',	'AFL',	'AKA',	'ARQQ',	'ASTI',	'BBAI',	'BCTX',	'BITF',	'BKYI',	'Bud',	'CAVA',	'CRNT',	'FDIS',	'FHLC',	'GME',	'HIMS',	'Hive',	'HOOD',	'HSDT',	'IHRT',	'IONQ',	'IWM',	'JNJ',	'JPM',	'K',	'KULR',	'LLY',	'MARA',	'MSTR',	'NFLX',	'NXST',	'OXY',	'PHIO',	'PLTR',	'PSEC',	'QBTS',	'QQQ',	'QUBT',	'RCAT',	'RGTI',	'RIME',	'SBET',	'SERV',	'SFTBY',	'SIDU',	'SIRI',	'SMR',	'SOFI',	'SOUN',	'SPY',	'TSM',	'UUUU',	'VRSN',	'VTI',	'VXUS',	'WKEY',	'ZONE',]

# Sidebar controls
st.sidebar.header("Configuration")
selected_stock = st.sidebar.selectbox('Select an asset to view:', portfolio)
n_years = st.sidebar.slider('Years of projection:', 1, 5)
period = n_years * 365

# Function to fetch data via yfinance
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START_DATE, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading market data...')
data = load_data(selected_stock)
data_load_state.text('Market data loaded successfully!')

# Section 1: Historical Data
st.subheader(f'Historical Market Data for {selected_stock}')
st.dataframe(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open Price", line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='#ff7f0e')))
    fig.layout.update(title_text='Historical Time Series with Range Slider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    
plot_raw_data()

# Section 2: Projections & Forecasting
st.subheader('🔮 Market Trend Projections')
st.write('Calculating projections based on historical momentum and seasonal trends...')

# Prepare the data frame for Prophet
df_train = pd.DataFrame()
df_train['ds'] = data['Date']

# 1. Safely extract the 'Close' column (handles yfinance's new nested format)
close_data = data['Close']
if isinstance(close_data, pd.DataFrame):
    close_data = close_data.iloc[:, 0] # Grab the first column if it's a nested DataFrame

# 2. Force the data to be numeric
df_train['y'] = pd.to_numeric(close_data, errors='coerce')

# 3. Strip timezone data from the dates
if df_train['ds'].dt.tz is not None:
    df_train['ds'] = df_train['ds'].dt.tz_localize(None)

# 4. Drop any missing values (Prophet will crash on fit() if it sees NaNs)
df_train = df_train.dropna()

# Initialize and fit the Prophet model
m = Prophet(daily_seasonality=True)
m.fit(df_train)
# Create future dates and predict
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display Projection Data
st.write(f'**Projection values for the next {n_years} year(s):**')
# Displaying date, predicted value (yhat), and the lower/upper confidence intervals
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the interactive forecast chart
st.write('**Interactive Projection Chart:**')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1, use_container_width=True)

# Plot the individual trend components
st.write("**Projection Components (Overall Trend vs. Yearly/Weekly Seasonality):**")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)
