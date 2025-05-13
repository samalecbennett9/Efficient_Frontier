import ef_functions as ef
import streamlit as st
import cvxpy as cp
import numpy as np
import pandas as pd

#title
st.title("Efficient Frontier")

#info
st.write("The following code calculates an Efficient Frontier for up to six assets. The user can choose which assets they would like to incorporate into their portfolio. Options include: S&P 500, Fixed Income, Gold, Private Credit, Real Estate, and Private Equity. The user can also select the time frame of the historical data used in the calculation.")

#user selects assets to include
assets = ['S&P 500', 'Fixed Income', 'Gold', 'Private Credit', 'Real Estate', 'Private Equity']
selected_assets = st.multiselect(
    "Choose 2 to 6 assets you would like to include in your Efficient Frontier",
    options=assets,
    default=assets  
)

#user selects historical time frame
time_frame = st.slider("Choose your desired time frame in years (1-10):", 1, 10)

#Read data
asset_data = ef.read_data("EF_Data_Summary.csv")

#Filter data to just include selected assets
asset_data_filtered = asset_data[selected_assets]

#Filter data to just include time frame
included_rows = int(time_frame)*12
asset_data_filtered_twice = asset_data_filtered.tail(included_rows)

#add "Weight" to selected assets for later use
selected_assets_with_weight = ef.rename(selected_assets)

#make the data into monthly returns
monthly_return_data = ef.monthly_return(asset_data_filtered_twice)

#make a series with the annualized returns of each asset
expected_annualized_returns = ef.calculate_avg_returns(monthly_return_data)

#make the cov matrix
cov_matrix = asset_data_filtered_twice.cov() * 12

#Primary Calculations
n = len(cov_matrix)
weights = cp.Variable(n)

#Calculate Target Returns
target_returns = np.linspace(
expected_annualized_returns.min(),
expected_annualized_returns.max(),
10 # Number of points on the frontier
)

#optimize
efficient_portfolios = ef.optimize(target_returns, weights, cov_matrix, expected_annualized_returns)

# Convert to DataFrame
efficient_frontier = pd.DataFrame(efficient_portfolios)
efficient_frontier['Standard Deviation'] = np.sqrt(efficient_frontier['Variance'])

# Remove duplicates (if any)
efficient_frontier = efficient_frontier.drop_duplicates(['Return', 'Variance'])

#make the portfolio table
table = ef.clean_and_table(efficient_frontier, selected_assets_with_weight)

#graph the EF
graph = ef.graph(table)

#show table df
st.dataframe(table)

#show EF graph
st.pyplot(graph)


















        