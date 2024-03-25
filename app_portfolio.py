import streamlit as st
import yfinance as yf,pandas as pd,numpy as np
import scipy
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
# st.checkbox('yes')
# st.button('Click')
# st.radio('Pick your gender',['Male','Female'])
# st.selectbox('Pick your gender',['Male','Female'])
# st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
# st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
# st.slider('Pick a number', 0,50)

# if 'tickers' not in st.session_state:
#     st.session_state['tickers'] = ['SPY','GLD']

st.header('TEAM HARBOR')
st.subheader('PORTFOLIO OPTIMIZER')
tickers = st.multiselect('Choose Tickers you want in your portfolio',['SPY','BND','GLD','^BSESN','^N225',"MSFT", "AMZN", "KO", "MA", "COST", 
           "LUV", "XOM", "PFE", "JPM", "UNH", 
           "ACN", "DIS", "GILD", "F", "TSLA"],default=['SPY','BND'])

end_date = st.date_input(label='Pick end date')
start_date = end_date - timedelta(days =5*365)
adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start = start_date,end = end_date)
    adj_close_df[ticker] = data['Adj Close']
log_returns = np.log(1+adj_close_df.pct_change()).dropna()
cov_matrix = log_returns.cov()*252
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)
risk_free_rate = .02

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.5) for _ in range(len(tickers))]


        

# if len(tickers)==0:
#     st.write('select atleast one ticker')
# else:
#     
initial_weights= np.array([1/len(tickers)]*len(tickers))

optimized_results = scipy.optimize.minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
optimal_weights = optimized_results.x

# print("Optimal Weights:")
# for ticker, weight in zip(tickers, optimal_weights):
#     print(f"{ticker}: {weight:.4f}")

    # f"Column name is **{columnName}**")

optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)    

# fig,ax = plt.subplots()
# plt.figure(figsize=(10, 6))
# ax.plot(tickers, optimal_weights)

fig = go.Figure([go.Bar(x=tickers, y=optimal_weights)])
fig.update_layout(xaxis_title="Assets", yaxis_title="Weights",yaxis_tickformat='.2%')
st.plotly_chart(fig)
# plt.show()
for ticker, weight in zip(tickers, optimal_weights):
    st.write(f'Optimal weights for **{ticker}**:**{"{:.2%}".format(weight)}**')   