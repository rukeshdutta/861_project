# Libraries required
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pypfopt
from datetime import datetime
import streamlit as st 
from pypfopt import risk_models
from pypfopt import EfficientFrontier
from pypfopt import DiscreteAllocation
import plotly.graph_objects as go
import plotly_express as px
import seaborn as sns

# APP HEADINGS
st.header('TEAM HARBOR')
st.subheader('PORTFOLIO OPTIMIZER')
# USER INPUT AREAS
tickers = st.multiselect('Choose Tickers you want in your portfolio',["^GSPC", "^GSPTSE", "^FTSE", "^DJI", "^N225", "^BSESN", "BND",'^BSESN','^N225','SPY',
  "GLD", "ETH-USD", "^IXIC", "^FVX","MSFT", "AMZN", "KO", "MA", "COST", "LUV", "XOM", "PFE", "JPM", "UNH", "ACN", "DIS", "GILD", "F", "TSLA"],
  default=['SPY','BND','GLD','^BSESN','^N225'])
# datefunctionalities
default_start_date = datetime(2019, 1, 1)
today = datetime.today()
# value = today
col1,col2 = st.columns(2)

with col1:
    start_date = st.date_input(label='Pick start date',value=default_start_date)
with col2:
    end_date = st.date_input(label='Pick end date',value='2024-03-28')


# col1,col2 = st.columns(2)
# with col1:
#     minimum_weight = st.slider(label='Select min weight',min_value=(0,1))
# with col2:
#     max_weight = st.slider(label='Select max weight',max_value=(0,1))
max_weight = st.number_input('Insert max weight you want to assign to an asset',min_value=0.01,max_value=1.00,value=0.25,help='Maximum exposure to each asset')
investment_amount = st.number_input('Investment amount in $ Millions',min_value=1,value=10,help='Min investment is $ 1M')

ohlc = yf.download(tickers, start = start_date,end = end_date)
prices = ohlc["Adj Close"].dropna(how="all") 


S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
ef = EfficientFrontier(None, S, weight_bounds=(0,max_weight))
ef.min_volatility()
weights = ef.clean_weights()
weights_df = pd.DataFrame( weights ,index=['Optimal Weights']).melt()
weights_df = weights_df.rename(columns={'variable':'Ticker','value':'Weight'})
# st.write(weights_df)
palette = sns.color_palette('Paired')
colors_hex = [f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}' for color in palette]
fig = px.pie(weights_df ,values='Weight', names='Ticker', color_discrete_sequence =colors_hex, title='Optimal Allocation of Selected Assets')
pie_chart = st.plotly_chart(fig)
annual_volatility = ef.portfolio_performance(verbose=True)[1]
# Volaitily lune
st.markdown(f'## :blue[Annual volatility is minimized at **{"{:.2%}".format(annual_volatility)}**]')
# pie_chart
# plt.show()
# st.write(pd.DataFrame(weights,index=['Optimum weights']))
# st.write(ef.portfolio_performance(verbose=True))


latest_prices = prices.iloc[-1].fillna(0) 
 # prices as of the day you are allocating
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=investment_amount*1_000_000, short_ratio=0)
alloc, leftover = da.lp_portfolio()
allocation_left = leftover/(investment_amount*1_000_000)
allocation_left = '{:.2%}'.format(allocation_left)
st.write(f":blue[Discrete allocation performed with] :green[__$**{leftover:.2f}**__ leftover]:blue[ which is] :green[{allocation_left} ]:blue[ of invested amount.]")
final_allocation = pd.DataFrame(alloc,index=['Units']).T
allocation_table = final_allocation.join(pd.DataFrame(latest_prices))
allocation_table_explained = """### :blue[Buy these amount of stocks at the latest prices.]"""
st.markdown(allocation_table_explained)
st.dataframe(allocation_table)
